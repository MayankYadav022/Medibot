import math
import requests
from config import LOCATIONIQ_API_KEY
from utils.logger import get_logger
log = get_logger(__name__)
_URGENT_PHRASES = (
    "severe",
    "immediate",
    "emergency",
    "urgent",
    "call 911",
    "trouble breathing",
    "shortness of breath",
    "chest pain",
    "fainting",
    "unconscious",
    "stroke",
    "seizure",
    "suicidal",
    "overdose",
    "severe headache",
    "worst headache",
)


def is_urgent_case(query: str, response: str) -> bool:
    text = f"{query} {response}".lower()
    return any(phrase in text for phrase in _URGENT_PHRASES)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(a))


def _geocode_location(location: str) -> dict | None:
    if not LOCATIONIQ_API_KEY or not location.strip():
        return None

    response = requests.get(
        "https://us1.locationiq.com/v1/search.php",
        params={
            "key": LOCATIONIQ_API_KEY,
            "q": location,
            "format": "json",
            "limit": 1,
        },
        timeout=20,
    )
    response.raise_for_status()

    data = response.json()
    if not data:
        return None

    first = data[0]
    return {
        "lat": float(first["lat"]),
        "lon": float(first["lon"]),
        "display_name": first.get("display_name", location),
    }


def _auto_detect_location() -> dict | None:
    """Best-effort approximate location from the user's public IP."""
    try:
        response = requests.get("https://ipapi.co/json/", timeout=10)
        response.raise_for_status()

        data = response.json()
        city = (data.get("city") or "").strip()
        region = (data.get("region") or "").strip()
        country = (data.get("country_name") or data.get("country") or "").strip()
        lat = data.get("latitude")
        lon = data.get("longitude")

        label_parts = [part for part in (city, region, country) if part]
        display_name = ", ".join(label_parts) if label_parts else "Approximate IP location"

        if lat is not None and lon is not None:
            return {
                "lat": float(lat),
                "lon": float(lon),
                "display_name": display_name,
            }

        if display_name and display_name != "Approximate IP location":
            return _geocode_location(display_name)

        return None
    except Exception as exc:
        log.warning("Automatic location detection failed: %s", exc)
        return None


def resolve_location(location: str | None = None) -> dict | None:
    """Resolve a user-provided location or fall back to IP-based detection."""
    if isinstance(location, dict):
        lat = location.get("lat")
        lon = location.get("lon")
        if lat is not None and lon is not None:
            return {
                "lat": float(lat),
                "lon": float(lon),
                "display_name": location.get("display_name", "Browser location"),
            }

    if location and location.strip():
        return _geocode_location(location.strip())

    return _auto_detect_location()
def find_nearby_hospitals(location: str | dict | None = None, limit: int = 3, radius_m: int = 10000) -> list[dict]:
    if not LOCATIONIQ_API_KEY:
        log.warning("LOCATIONIQ_API_KEY is not configured")
        return []

    try:
        center = resolve_location(location)
        if not center:
            return []

        response = requests.get(
            "https://us1.locationiq.com/v1/nearby.php",
            params={
                "key": LOCATIONIQ_API_KEY,
                "lat": center["lat"],
                "lon": center["lon"],
                "tag": "amenity:hospital",
                "radius": radius_m,
                "format": "json",
            },
            timeout=20,
        )
        response.raise_for_status()

        hospitals = []
        for item in response.json()[:limit]:
            item_lat = item.get("lat")
            item_lon = item.get("lon")
            distance_km = None
            if item_lat is not None and item_lon is not None:
                distance_km = _haversine_km(
                    center["lat"],
                    center["lon"],
                    float(item_lat),
                    float(item_lon),
                )

            address = item.get("display_name") or item.get("name") or "Address unavailable"
            hospitals.append(
                {
                    "name": item.get("name") or item.get("display_name") or "Hospital",
                    "address": address,
                    "distance_km": distance_km,
                }
            )

        return hospitals
    except Exception as exc:
        log.warning("LocationIQ lookup failed: %s", exc)
        return []