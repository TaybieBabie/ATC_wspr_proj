# adsb_tracker.py
import requests
import json
import time
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import math
from utils import config


class Aircraft:
    """Represents an aircraft with its tracking data"""

    def __init__(self, icao24: str, callsign: str, latitude: float,
                 longitude: float, altitude: float, track: float,
                 ground_speed: float, vertical_rate: float,
                 on_ground: bool, timestamp: datetime):
        self.icao24 = icao24
        self.callsign = callsign.strip() if callsign else ""
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude  # in feet
        self.track = track  # heading in degrees
        self.ground_speed = ground_speed  # in knots
        self.vertical_rate = vertical_rate  # in feet/min
        self.on_ground = on_ground
        self.timestamp = timestamp
        self.distance_from_airport = None
        self.bearing_from_airport = None

    def calculate_distance_and_bearing(self, ref_lat: float, ref_lon: float):
        """Calculate distance and bearing from reference point"""
        R = 3440.065  # Earth radius in nautical miles

        lat1, lon1 = math.radians(ref_lat), math.radians(ref_lon)
        lat2, lon2 = math.radians(self.latitude), math.radians(self.longitude)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        # Distance
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))
        self.distance_from_airport = R * c

        # Bearing
        y = math.sin(dlon) * math.cos(lat2)
        x = (math.cos(lat1) * math.sin(lat2) -
             math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
        bearing = math.degrees(math.atan2(y, x))
        self.bearing_from_airport = (bearing + 360) % 360

        return self.distance_from_airport, self.bearing_from_airport

    def to_dict(self) -> dict:
        """
        Serializes the Aircraft object to a dictionary for passing
        through a queue to the GUI.
        """
        return {
            'icao24': self.icao24,
            'callsign': self.callsign,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'altitude': self.altitude,
            'track': self.track,
            'ground_speed': self.ground_speed,
            'vertical_rate': self.vertical_rate,
            'on_ground': self.on_ground,
            'timestamp': self.timestamp.isoformat(),
            'distance_from_airport': self.distance_from_airport,
            'bearing_from_airport': self.bearing_from_airport
        }

    def __str__(self):
        return (f"{self.callsign or self.icao24}: "
                f"{self.altitude}ft @ {self.distance_from_airport:.1f}nm "
                f"{self.bearing_from_airport:.0f}°")


class ADSBDataSource(ABC):
    """Abstract base class for ADS-B data sources"""

    @abstractmethod
    def get_aircraft_in_area(self, lat: float, lon: float,
                             radius_nm: float) -> List[Aircraft]:
        """Get all aircraft within radius of given point"""
        pass


class OpenSkySource(ADSBDataSource):
    """OpenSky Network API data source"""

    def __init__(self, credentials_file: str = None):
        self.base_url = "https://opensky-network.org/api"
        self.token_url = (
            "https://auth.opensky-network.org/auth/realms/opensky-network/"
            "protocol/openid-connect/token"
        )
        self.credentials = None
        self.token = None
        self.token_expiry = 0
        if credentials_file:
            try:
                with open(credentials_file, "r") as f:
                    creds = json.load(f)
                client_id = creds.get("client_id") or creds.get("clientId")
                client_secret = (
                    creds.get("client_secret") or creds.get("clientSecret")
                )
                if not client_id or not client_secret:
                    raise ValueError(
                        "OpenSky credentials must include non-empty "
                        "'client_id'/'clientId' and 'client_secret'/'clientSecret'"
                    )
                self.credentials = {
                    "client_id": client_id,
                    "client_secret": client_secret,
                }
                if "scope" in creds:
                    self.credentials["scope"] = creds["scope"]
            except FileNotFoundError:
                print(
                    f"OpenSky credentials file not found: {credentials_file}"
                )
        self.last_request_time = 0
        self.rate_limit = 5 if self.credentials else 10  # seconds between requests

    def _get_access_token(self) -> Optional[str]:
        """Retrieve or refresh OAuth2 token"""
        if not self.credentials:
            return None
        if self.token and time.time() < self.token_expiry - 60:
            return self.token

        client_id = self.credentials.get("client_id", "")
        client_secret = self.credentials.get("client_secret", "")
        scope = self.credentials.get("scope")

        data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }
        if scope:
            data["scope"] = scope

        try:
            resp = requests.post(
                self.token_url,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data=data,
                timeout=10,
            )
            try:
                resp.raise_for_status()
            except requests.HTTPError as http_err:
                if resp.status_code == 400:
                    fallback_data = {"grant_type": "client_credentials"}
                    if scope:
                        fallback_data["scope"] = scope
                    try:
                        resp = requests.post(
                            self.token_url,
                            headers={"Content-Type": "application/x-www-form-urlencoded"},
                            data=fallback_data,
                            auth=(client_id, client_secret),
                            timeout=10,
                        )
                        resp.raise_for_status()
                    except requests.HTTPError as http_err2:
                        print(
                            f"Error obtaining OpenSky token: {http_err2} - {resp.text}"
                        )
                        return None
                    except requests.RequestException as e2:
                        print(f"Error obtaining OpenSky token: {e2}")
                        return None
                else:
                    print(
                        f"Error obtaining OpenSky token: {http_err} - {resp.text}"
                    )
                    return None

            data = resp.json()
            self.token = data.get("access_token")
            self.token_expiry = time.time() + data.get("expires_in", 0)
            return self.token
        except requests.RequestException as e:
            print(f"Error obtaining OpenSky token: {e}")
            return None

    def get_aircraft_in_area(self, lat: float, lon: float,
                             radius_nm: float) -> List[Aircraft]:
        """Get aircraft from OpenSky API"""
        # Rate limiting
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.rate_limit:
            time.sleep(self.rate_limit - time_since_last)

        # Bounding box for initial query
        lat_delta = radius_nm / 60.0
        lon_delta = radius_nm / (60.0 * math.cos(math.radians(lat)))
        params = {
            'lamin': lat - lat_delta,
            'lamax': lat + lat_delta,
            'lomin': lon - lon_delta,
            'lomax': lon + lon_delta
        }

        try:
            headers = {}
            token = self._get_access_token()
            if token:
                headers["Authorization"] = f"Bearer {token}"
            response = requests.get(
                f"{self.base_url}/states/all",
                params=params,
                headers=headers,
                timeout=15
            )
            self.last_request_time = time.time()
            response.raise_for_status()

            data = response.json()
            aircraft_list = []

            if data.get('states'):
                for state in data['states']:
                    if state[5] is not None and state[6] is not None:
                        aircraft = Aircraft(
                            icao24=state[0],
                            callsign=state[1] or "",
                            latitude=state[6],
                            longitude=state[5],
                            altitude=state[13] * 3.28084 if state[13] is not None else 0,
                            track=state[10] or 0,
                            ground_speed=state[9] * 1.94384 if state[9] is not None else 0,
                            vertical_rate=state[11] * 196.85 if state[11] is not None else 0,
                            on_ground=state[8],
                            timestamp=datetime.fromtimestamp(state[3] or time.time())
                        )
                        aircraft.calculate_distance_and_bearing(lat, lon)

                        if aircraft.distance_from_airport <= radius_nm:
                            aircraft_list.append(aircraft)
            return aircraft_list

        except requests.exceptions.RequestException as e:
            print(f"Error fetching OpenSky data: {e}")
        return []


class ADSBExchangeSource(ADSBDataSource):
    """ADS-B Exchange API data source"""

    def __init__(self, api_key: str = None):
        self.base_url = "https://adsbexchange.com/api/aircraft/v2"
        self.api_key = api_key
        self.last_request_time = 0
        self.rate_limit = 1

    def get_aircraft_in_area(self, lat: float, lon: float,
                             radius_nm: float) -> List[Aircraft]:
        return [] # Placeholder


class LocalADSBSource(ADSBDataSource):
    """Local dump1090/dump978 data source"""

    def __init__(self, dump1090_url: str = "http://localhost:8080"):
        self.dump1090_url = dump1090_url

    def get_aircraft_in_area(self, lat: float, lon: float,
                             radius_nm: float) -> List[Aircraft]:
        try:
            response = requests.get(
                f"{self.dump1090_url}/data/aircraft.json",
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            aircraft_list = []
            for ac in data.get('aircraft', []):
                if 'lat' in ac and 'lon' in ac:
                    aircraft = Aircraft(
                        icao24=ac.get('hex', ''),
                        callsign=ac.get('flight', ''),
                        latitude=ac['lat'],
                        longitude=ac['lon'],
                        altitude=ac.get('alt_baro', ac.get('alt_geom', 0)),
                        track=ac.get('track', 0),
                        ground_speed=ac.get('gs', 0),
                        vertical_rate=ac.get('vert_rate', 0),
                        on_ground=ac.get('alt_baro', 1000) < 100,
                        timestamp=datetime.now()
                    )
                    aircraft.calculate_distance_and_bearing(lat, lon)
                    if aircraft.distance_from_airport <= radius_nm:
                        aircraft_list.append(aircraft)
            return aircraft_list
        except requests.exceptions.RequestException as e:
            print(f"Error fetching local ADS-B data: {e}")
        return []


class ADSBTracker:
    """Main ADS-B tracking coordinator"""

    def __init__(self, data_source: ADSBDataSource = None):
        self.data_source = data_source or OpenSkySource(
            config.OPENSKY_CREDENTIALS_FILE
        )
        self.aircraft_history = {}
        self.current_aircraft = {}

    def update_aircraft_positions(self):
        """Fetch current aircraft positions"""
        aircraft_list = self.data_source.get_aircraft_in_area(
            config.AIRPORT_LAT, config.AIRPORT_LON, config.SEARCH_RADIUS_NM
        )
        self.current_aircraft = {ac.icao24: ac for ac in aircraft_list}
        return aircraft_list

    def find_aircraft_by_callsign(self, callsign: str) -> Optional[Aircraft]:
        """Find aircraft by callsign (handles variations)"""
        callsign = callsign.upper().strip()
        for aircraft in self.current_aircraft.values():
            if aircraft.callsign == callsign:
                return aircraft
        return None

    def get_aircraft_at_altitude(self, altitude: int,
                                 tolerance: int = 500) -> List[Aircraft]:
        """Find aircraft at specific altitude ± tolerance"""
        results = []
        for aircraft in self.current_aircraft.values():
            if abs(aircraft.altitude - altitude) <= tolerance:
                results.append(aircraft)
        return results

    def get_aircraft_by_position(self, bearing: float, distance: float,
                                 bearing_tolerance: float = 30,
                                 distance_tolerance: float = 5) -> List[Aircraft]:
        """Find aircraft by position relative to airport"""
        results = []
        for aircraft in self.current_aircraft.values():
            if (abs(aircraft.bearing_from_airport - bearing) <= bearing_tolerance and
                    abs(aircraft.distance_from_airport - distance) <= distance_tolerance):
                results.append(aircraft)
        return results
