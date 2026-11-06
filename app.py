import asyncio
import base64
import json
import os
import re
from typing import Optional

import anthropic
import gpxpy
import gpxpy.gpx
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

anthropic_client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
ORS_API_KEY = os.getenv("ORS_API_KEY")
ORS_HEADERS = {
    "Authorization": f"Bearer {ORS_API_KEY}",
    "Content-Type": "application/json",
}
QUEBEC_GEOCODE_URL = "https://geoegl.msp.gouv.qc.ca/apis/icherche/geocode"


class Stop(BaseModel):
    address: str
    lat: Optional[float] = None
    lng: Optional[float] = None


class RoundTrip(BaseModel):
    pickup: Stop
    dropoff: Stop
    return_to: Stop


class RouteRequest(BaseModel):
    checkpoints: list[Stop]
    round_trip: Optional[RoundTrip] = None
    start_lat: Optional[float] = None
    start_lng: Optional[float] = None
    city: str = ""


class GeocodeRequest(BaseModel):
    address: str
    city: str = ""


@app.post("/extract")
async def extract_addresses(image: UploadFile = File(...)):
    image_bytes = await image.read()
    media_type = image.content_type or "image/jpeg"
    image_data = base64.standard_b64encode(image_bytes).decode("utf-8")

    message = await anthropic_client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": image_data},
                },
                {
                    "type": "text",
                    "text": (
                        "Extract all locations from this bike race manifest and return a JSON object with two keys:\n\n"
                        "1. \"checkpoints\": array of regular checkpoint addresses (the section that can be done in any order). "
                        "For proper street addresses use them as written. "
                        "For landmarks without a street address, combine the checkpoint NAME and description: 'Name - Description' "
                        "(e.g. 'Forbidden McGill dip - Piscine secrète'). "
                        "Include duplicate addresses as separate entries.\n\n"
                        "2. \"round_trip\": if there is an ordered delivery run (aller-retour / round trip), an object with "
                        "\"pickup\", \"dropoff\", and \"return_to\" string addresses. Otherwise null.\n\n"
                        "Only return the JSON object, no other text.\n"
                        "Example:\n"
                        '{"checkpoints": ["5514 Rue Saint-Denis", "Forbidden McGill dip - Piscine secrète"], '
                        '"round_trip": {"pickup": "101 Rue Villeneuve Ouest", "dropoff": "2523 Rue Sheppard", "return_to": "101 Rue Villeneuve Ouest"}}'
                    ),
                },
            ],
        }],
    )

    text = message.content[0].text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Could not parse manifest: {e}")

    return data


def clean_address(address: str) -> str:
    """Strip parenthetical notes like '(par la ruelle)' before geocoding."""
    return re.sub(r'\s*\(.*?\)', '', address).strip()


async def geocode(address: str, city: str, client: httpx.AsyncClient) -> tuple[float, float]:
    # 1. Quebec government API — accurate for QC street addresses
    muni = city.replace(", QC", "").replace(", Quebec", "").strip()
    query = f"{clean_address(address)}, {muni}" if muni else clean_address(address)
    try:
        resp = await client.get(
            QUEBEC_GEOCODE_URL,
            params={"q": query, "limit": 1, "type": "adresse", "geometry": "centroid"},
            timeout=10,
        )
        resp.raise_for_status()
        features = resp.json().get("features", [])
        if features and "geometry" in features[0]:
            lng, lat = features[0]["geometry"]["coordinates"]
            return float(lat), float(lng)
    except Exception:
        pass

    # 2. Claude with web search — for landmarks, colloquial names, unofficial spots
    return await geocode_with_llm(address, city)


async def geocode_with_llm(address: str, city: str) -> tuple[float, float]:
    location = f"{address}, {city}" if city else address
    prompt = (
        f'Find the GPS coordinates of this location: "{location}"\n'
        "This is a stop in a Montreal alleycat bike messenger race. "
        "It may be a colloquial name, local nickname, or landmark known to Montreal cyclists. "
        "Search the web to find accurate coordinates. "
        "A rough location is far better than nothing. "
        'Reply with only a JSON object: {"lat": 45.5017, "lng": -73.5673}\n'
        "Only return null if you have absolutely no information about this location."
    )
    messages = [{"role": "user", "content": prompt}]

    response = await anthropic_client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=messages,
    )

    # web_search is server-side so results come back in the same response (stop_reason=end_turn)
    # but handle tool_use loop defensively in case of changes
    while response.stop_reason == "tool_use":
        messages.append({"role": "assistant", "content": response.content})
        tool_results = [
            {"type": "tool_result", "tool_use_id": b.id, "content": ""}
            for b in response.content if b.type == "tool_use"
        ]
        messages.append({"role": "user", "content": tool_results})
        response = await anthropic_client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=messages,
        )

    # Extract the last text block (after any interleaved tool results)
    text = next((b.text for b in reversed(response.content) if hasattr(b, "text")), "").strip()

    if not text or text.lower() == "null":
        raise HTTPException(status_code=422, detail=f'Could not locate: "{address}"')

    # Pull out the JSON object even if Claude added prose around it
    m = re.search(r'\{[^{}]*"lat"[^{}]*"lng"[^{}]*\}', text)
    if not m:
        m = re.search(r'\{[^{}]*"lng"[^{}]*"lat"[^{}]*\}', text)
    if not m:
        raise HTTPException(status_code=422, detail=f'Could not parse coordinates for: "{address}"')
    try:
        coords = json.loads(m.group())
        return float(coords["lat"]), float(coords["lng"])
    except (json.JSONDecodeError, KeyError):
        raise HTTPException(status_code=422, detail=f'Could not parse coordinates for: "{address}"')


@app.post("/geocode")
async def geocode_single(req: GeocodeRequest):
    async with httpx.AsyncClient() as client:
        lat, lng = await geocode(req.address, req.city, client)
    return {"lat": lat, "lng": lng}


@app.post("/route")
async def generate_route(req: RouteRequest):
    # --- 1. Geocode everything (skip stops that already have coordinates) ---
    all_stops: list[Stop] = req.checkpoints.copy()
    rt_offset = len(all_stops)

    if req.round_trip:
        all_stops += [req.round_trip.pickup, req.round_trip.dropoff, req.round_trip.return_to]

    geocoded_all: list[dict] = []
    async with httpx.AsyncClient() as client:
        for stop in all_stops:
            if stop.lat is not None and stop.lng is not None:
                geocoded_all.append({"address": stop.address, "lat": stop.lat, "lng": stop.lng})
                continue
            lat, lng = await geocode(stop.address, req.city, client)
            geocoded_all.append({"address": stop.address, "lat": lat, "lng": lng})

    geocoded_checkpoints = geocoded_all[:rt_offset]

    # --- 2. Start position ---
    if req.start_lat is not None and req.start_lng is not None:
        start = [req.start_lng, req.start_lat]
    else:
        start = [
            sum(g["lng"] for g in geocoded_all) / len(geocoded_all),
            sum(g["lat"] for g in geocoded_all) / len(geocoded_all),
        ]

    # --- 3. Build Vroom optimization request ---
    # Regular checkpoints: free jobs (any order)
    jobs = [
        {"id": i + 1, "location": [g["lng"], g["lat"]]}
        for i, g in enumerate(geocoded_checkpoints)
    ]

    # ID lookup so we can reconstruct order after optimization
    id_to_stop: dict[int, dict] = {i + 1: g for i, g in enumerate(geocoded_checkpoints)}

    vehicle: dict = {"id": 1, "profile": "cycling-regular", "start": start}
    shipments = []

    if req.round_trip:
        g_pickup  = geocoded_all[rt_offset]
        g_dropoff = geocoded_all[rt_offset + 1]
        g_return  = geocoded_all[rt_offset + 2]

        pickup_id   = rt_offset + 1
        delivery_id = rt_offset + 2

        # Shipment enforces pickup → dropoff ordering; other checkpoints intersperse freely
        shipments = [{
            "pickup":   {"id": pickup_id,   "location": [g_pickup["lng"],  g_pickup["lat"]]},
            "delivery": {"id": delivery_id, "location": [g_dropoff["lng"], g_dropoff["lat"]]},
        }]
        id_to_stop[pickup_id]   = {**g_pickup,  "label": "PICKUP"}
        id_to_stop[delivery_id] = {**g_dropoff, "label": "DROPOFF"}

        # Vehicle ends at the return address — Vroom routes there automatically
        vehicle["end"] = [g_return["lng"], g_return["lat"]]

    opt_body: dict = {"jobs": jobs, "vehicles": [vehicle]}
    if shipments:
        opt_body["shipments"] = shipments

    async with httpx.AsyncClient() as client:
        opt_resp = await client.post(
            "https://api.openrouteservice.org/optimization",
            json=opt_body,
            headers=ORS_HEADERS,
            timeout=30,
        )
        if not opt_resp.is_success:
            raise HTTPException(status_code=502, detail=f"ORS optimization failed: {opt_resp.text}")
        opt_data = opt_resp.json()

    unassigned = opt_data.get("unassigned", [])
    if unassigned:
        addrs = [id_to_stop[u["id"]]["address"] for u in unassigned if u["id"] in id_to_stop]
        raise HTTPException(status_code=422, detail=f"Could not route these stops: {addrs}")

    # --- 4. Reconstruct ordered stops from Vroom steps ---
    steps = opt_data["routes"][0]["steps"]
    ordered_waypoints: list[list[float]] = []  # [lng, lat] for BRouter
    ordered_stops: list[dict] = []             # for GPX waypoints

    for step in steps:
        stype = step["type"]
        if stype == "start":
            ordered_waypoints.append(start)
        elif stype in ("job", "pickup", "delivery"):
            sid = step["id"]
            g = id_to_stop[sid]
            ordered_waypoints.append([g["lng"], g["lat"]])
            ordered_stops.append(g)
        elif stype == "end" and req.round_trip:
            g = geocoded_all[rt_offset + 2]  # return address
            ordered_waypoints.append([g["lng"], g["lat"]])
            ordered_stops.append({**g, "label": "RETURN"})

    # --- 5. BRouter fastbike for full road geometry ---
    lonlats = "|".join(f"{lng},{lat}" for lng, lat in ordered_waypoints)

    async with httpx.AsyncClient() as client:
        brouter_resp = await client.get(
            "https://brouter.de/brouter",
            params={"lonlats": lonlats, "profile": "fastbike", "alternativeidx": "0", "format": "geojson"},
            timeout=30,
        )
        if not brouter_resp.is_success:
            raise HTTPException(status_code=502, detail=f"BRouter failed: {brouter_resp.text}")
        brouter_data = brouter_resp.json()

    route_coords = brouter_data["features"][0]["geometry"]["coordinates"]

    # --- 6. Build GPX ---
    gpx = gpxpy.gpx.GPX()
    gpx.name = "Alleycat Route"

    for i, stop in enumerate(ordered_stops):
        label = stop.get("label", "")
        prefix = f"[{label}] " if label else f"{i + 1}. "
        gpx.waypoints.append(gpxpy.gpx.GPXWaypoint(
            latitude=stop["lat"],
            longitude=stop["lng"],
            name=f"{prefix}{stop['address']}",
        ))

    track = gpxpy.gpx.GPXTrack(name="Alleycat Route")
    segment = gpxpy.gpx.GPXTrackSegment()
    for lng, lat, *_ in route_coords:
        segment.points.append(gpxpy.gpx.GPXTrackPoint(latitude=lat, longitude=lng))
    track.segments.append(segment)
    gpx.tracks.append(track)

    return Response(
        content=gpx.to_xml(),
        media_type="application/gpx+xml",
        headers={"Content-Disposition": "attachment; filename=alleycat_route.gpx"},
    )


# Serve index.html with Mapbox token injected from environment
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    mapbox_token = os.getenv("MAPBOX_TOKEN", "")
    with open("static/index.html") as f:
        html = f.read()
    html = html.replace("__MAPBOX_TOKEN__", mapbox_token)
    return HTMLResponse(content=html)

# Static files last — catches everything not matched above
app.mount("/static", StaticFiles(directory="static"), name="static")
