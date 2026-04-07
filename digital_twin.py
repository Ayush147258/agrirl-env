# digital_twin.py
"""
Digital Twin — Real-World Weather Grounding
-------------------------------------------
Fetches live meteorological data from the Open-Meteo API (free, no key needed)
and maps real-world conditions onto the simulation's environment parameters.

This transforms the simulation from a static game into a Digital Twin:
  Real-World Sensor Data  →  Environment Scaling  →  Agent Adapts

Supported farming regions (lat/lon presets):
  "punjab"      → Ludhiana, Punjab, India
  "california"  → Central Valley, California, USA
  "midwest"     → Iowa, USA
  "maharashtra" → Nashik, Maharashtra, India
"""

import urllib.request
import json
from dataclasses import dataclass
from typing import Optional, Literal

# ── Region Registry ────────────────────────────────────────────────────────────

REGIONS: dict[str, tuple[float, float]] = {
    "punjab":      (30.9010, 75.8573),
    "california":  (36.7783, -119.4179),
    "midwest":     (41.8780, -93.0977),
    "maharashtra": (19.9975, 73.7898),
}

RegionKey = Literal["punjab", "california", "midwest", "maharashtra"]


# ── Real-World Weather Snapshot ────────────────────────────────────────────────

@dataclass
class RealWeatherSnapshot:
    region: str
    temperature_c: float
    precipitation_mm: float
    humidity_pct: float
    wind_kmh: float
    raw_condition: str          # "clear" | "rain" | "clouds" | "extreme"

    # ── Derived Scaling Factors ────────────────────────────────────────────
    @property
    def evaporation_multiplier(self) -> float:
        """
        Higher temp → crops lose moisture faster.
        Scale: 1.0 at 25°C, up to 2.0 at 45°C, down to 0.5 at 5°C.
        """
        return max(0.5, min(2.0, 1.0 + (self.temperature_c - 25) / 20))

    @property
    def rain_bonus(self) -> float:
        """Extra soil moisture added by real-world precipitation (per step)."""
        return min(15.0, self.precipitation_mm * 0.8)

    @property
    def sim_weather(self) -> str:
        """Maps real condition onto simulation WeatherType."""
        if self.temperature_c >= 38:
            return "heatwave"
        if self.temperature_c <= 4:
            return "frost"
        if self.precipitation_mm >= 5:
            return "rainy"
        if self.precipitation_mm >= 0.5 or self.humidity_pct >= 70:
            return "cloudy"
        return "sunny"

    def summary(self) -> str:
        return (
            f"[RealWeather:{self.region}] "
            f"{self.temperature_c:.1f}°C | "
            f"Rain:{self.precipitation_mm:.1f}mm | "
            f"Humidity:{self.humidity_pct:.0f}% | "
            f"→ sim_weather={self.sim_weather} | "
            f"evap×{self.evaporation_multiplier:.2f} | "
            f"rain_bonus+{self.rain_bonus:.1f}"
        )


# ── Digital Twin Client ────────────────────────────────────────────────────────

class DigitalTwin:
    """
    Fetches live weather from Open-Meteo and exposes scaling factors
    that the Executor applies to modify the simulation's physics.

    Usage:
        twin    = DigitalTwin(region="punjab")
        weather = twin.fetch()          # RealWeatherSnapshot
        obs     = twin.apply(obs)       # patch observation in-place
    """

    OPEN_METEO_URL = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,precipitation,relative_humidity_2m,wind_speed_10m"
        "&forecast_days=1"
    )

    def __init__(self, region: RegionKey = "punjab", timeout: int = 5):
        if region not in REGIONS:
            raise ValueError(f"Unknown region '{region}'. Choose from: {list(REGIONS)}")
        self.region   = region
        self.lat, self.lon = REGIONS[region]
        self.timeout  = timeout
        self._cache: Optional[RealWeatherSnapshot] = None

    def fetch(self, force: bool = False) -> RealWeatherSnapshot:
        """
        Fetch current weather from Open-Meteo.
        Results are cached for the session (one fetch per episode).
        Pass force=True to refresh mid-episode.
        """
        if self._cache and not force:
            return self._cache

        url = self.OPEN_METEO_URL.format(lat=self.lat, lon=self.lon)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AgriRL-DigitalTwin/1.0"})
            import requests
            resp = requests.get(url, timeout=self.timeout, verify=False)
            data = resp.json()
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data    = json.loads(resp.read())
                current = data["current"]
                snapshot = RealWeatherSnapshot(
                    region           = self.region,
                    temperature_c    = float(current.get("temperature_2m", 25.0)),
                    precipitation_mm = float(current.get("precipitation",   0.0)),
                    humidity_pct     = float(current.get("relative_humidity_2m", 50.0)),
                    wind_kmh         = float(current.get("wind_speed_10m",   10.0)),
                    raw_condition    = _classify_condition(
                        float(current.get("temperature_2m",  25.0)),
                        float(current.get("precipitation",    0.0)),
                    ),
                )
        except Exception as e:
            print(f"[DigitalTwin] API fetch failed ({e}), using neutral defaults.")
            snapshot = RealWeatherSnapshot(
                region="fallback", temperature_c=25.0, precipitation_mm=0.0,
                humidity_pct=50.0, wind_kmh=10.0, raw_condition="clear",
            )

        self._cache = snapshot
        return snapshot

    def apply(self, obs, snapshot: Optional[RealWeatherSnapshot] = None):
        """
        Patch an observation with real-world scaling factors.
        - Overrides obs.weather with the real-world mapped weather type.
        - Adjusts each crop's moisture based on evaporation and rain bonus.
        Returns the (mutated) obs.
        """
        if snapshot is None:
            snapshot = self.fetch()

        # Override sim weather with real-world equivalent
        try:
            obs.weather = snapshot.sim_weather
        except Exception:
            pass  # read-only obs — skip weather override

        # Apply evaporation drain and rain bonus to every crop
        for crop in getattr(obs, "crops", []):
            try:
                delta = snapshot.rain_bonus - (snapshot.evaporation_multiplier - 1.0) * 3.0
                crop.moisture = max(0.0, min(100.0, crop.moisture + delta))
            except Exception:
                pass

        return obs


# ── Internal Helpers ───────────────────────────────────────────────────────────

def _classify_condition(temp: float, precip: float) -> str:
    if temp >= 38 or temp <= 2:
        return "extreme"
    if precip >= 5:
        return "rain"
    if precip >= 0.5:
        return "clouds"
    return "clear"