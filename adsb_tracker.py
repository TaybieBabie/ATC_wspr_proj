# adsb_tracker.py
import requests
import json
import time
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import math
from config import (AIRPORT_LAT, AIRPORT_LON, SEARCH_RADIUS_NM,
                    OPENSKY_USERNAME, OPENSKY_PASSWORD)


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

    def __init__(self, username: str = None, password: str = None):
        self.base_url = "https://opensky-network.org/api"
        self.auth = (username, password) if username and password else None
        self.last_request_time = 0
        self.rate_limit = 5 if self.auth else 10  # seconds between requests

    def get_aircraft_in_area(self, lat: float, lon: float,
                             radius_nm: float) -> List[Aircraft]:
        """Get aircraft from OpenSky API"""
        # Rate limiting
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.rate_limit:
            time.sleep(self.rate_limit - time_since_last)

        # Convert radius to degrees (approximate)
        lat_delta = radius_nm / 60.0
        lon_delta = radius_nm / (60.0 * math.cos(math.radians(lat)))

        # Bounding box
        params = {
            'lamin': lat - lat_delta,
            'lamax': lat + lat_delta,
            'lomin': lon - lon_delta,
            'lomax': lon + lon_delta
        }

        try:
            response = requests.get(
                f"{self.base_url}/states/all",
                params=params,
                auth=self.auth,
                timeout=15
            )
            self.last_request_time = time.time()

            if response.status_code == 200:
                data = response.json()
                aircraft_list = []

                for state in data.get('states', []):
                    # Parse OpenSky state vector
                    if state[5] and state[6]:  # Has position
                        aircraft = Aircraft(
                            icao24=state[0],
                            callsign=state[1] or "",
                            latitude=state[6],
                            longitude=state[5],
                            altitude=state[13] * 3.28084 if state[13] else 0,  # m to ft
                            track=state[10] or 0,
                            ground_speed=state[9] * 1.94384 if state[9] else 0,  # m/s to kts
                            vertical_rate=state[11] * 196.85 if state[11] else 0,  # m/s to ft/min
                            on_ground=state[8],
                            timestamp=datetime.fromtimestamp(state[3] or time.time())
                        )
                        aircraft.calculate_distance_and_bearing(lat, lon)

                        # Filter by actual distance
                        if aircraft.distance_from_airport <= radius_nm:
                            aircraft_list.append(aircraft)

                return aircraft_list

        except Exception as e:
            print(f"Error fetching OpenSky data: {e}")

        return []


class ADSBExchangeSource(ADSBDataSource):
    """ADS-B Exchange API data source"""

    def __init__(self, api_key: str = None):
        self.base_url = "https://adsbexchange.com/api/aircraft/v2"
        self.api_key = api_key
        self.last_request_time = 0
        self.rate_limit = 1  # seconds between requests

    def get_aircraft_in_area(self, lat: float, lon: float,
                             radius_nm: float) -> List[Aircraft]:
        """Get aircraft from ADS-B Exchange"""
        # Rate limiting
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.rate_limit:
            time.sleep(self.rate_limit - time_since_last)

        try:
            # ADS-B Exchange uses different endpoint structure
            params = {
                'lat': lat,
                'lon': lon,
                'dist': radius_nm
            }

            headers = {}
            if self.api_key:
                headers['api-auth'] = self.api_key

            response = requests.get(
                f"{self.base_url}/lat/{lat}/lon/{lon}/dist/{radius_nm}/",
                headers=headers,
                timeout=15
            )
            self.last_request_time = time.time()

            # Parse response...
            # (Implementation depends on specific API version)

        except Exception as e:
            print(f"Error fetching ADS-B Exchange data: {e}")

        return []


class LocalADSBSource(ADSBDataSource):
    """Local dump1090/dump978 data source"""

    def __init__(self, dump1090_url: str = "http://localhost:8080"):
        self.dump1090_url = dump1090_url

    def get_aircraft_in_area(self, lat: float, lon: float,
                             radius_nm: float) -> List[Aircraft]:
        """Get aircraft from local dump1090"""
        try:
            response = requests.get(
                f"{self.dump1090_url}/data/aircraft.json",
                timeout=5
            )

            if response.status_code == 200:
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

        except Exception as e:
            print(f"Error fetching local ADS-B data: {e}")

        return []


class ADSBTracker:
    """Main ADS-B tracking coordinator"""

    def __init__(self, data_source: ADSBDataSource = None):
        self.data_source = data_source or OpenSkySource(
            OPENSKY_USERNAME, OPENSKY_PASSWORD
        )
        self.aircraft_history = {}  # Store historical positions
        self.current_aircraft = {}  # Current aircraft in area

    def update_aircraft_positions(self):
        """Fetch current aircraft positions"""
        aircraft_list = self.data_source.get_aircraft_in_area(
            AIRPORT_LAT, AIRPORT_LON, SEARCH_RADIUS_NM
        )

        # Update current aircraft dict
        self.current_aircraft = {
            ac.callsign: ac for ac in aircraft_list if ac.callsign
        }

        # Also store by ICAO24 for non-callsign lookups
        for ac in aircraft_list:
            self.current_aircraft[ac.icao24] = ac

        # Update history
        timestamp = datetime.now()
        for ac in aircraft_list:
            if ac.callsign not in self.aircraft_history:
                self.aircraft_history[ac.callsign] = []
            self.aircraft_history[ac.callsign].append({
                'timestamp': timestamp,
                'aircraft': ac
            })

        return aircraft_list

    def find_aircraft_by_callsign(self, callsign: str) -> Optional[Aircraft]:
        """Find aircraft by callsign (handles variations)"""
        callsign = callsign.upper().strip()

        # Direct match
        if callsign in self.current_aircraft:
            return self.current_aircraft[callsign]

        # Try without spaces
        callsign_no_space = callsign.replace(" ", "")
        if callsign_no_space in self.current_aircraft:
            return self.current_aircraft[callsign_no_space]

        # Partial match
        for call, aircraft in self.current_aircraft.items():
            if callsign in call or call in callsign:
                return aircraft

        return None

    def get_aircraft_at_altitude(self, altitude: int,
                                 tolerance: int = 500) -> List[Aircraft]:
        """Find aircraft at specific altitude ± tolerance"""
        results = []
        for aircraft in self.current_aircraft.values():
            if isinstance(aircraft, Aircraft):
                if abs(aircraft.altitude - altitude) <= tolerance:
                    results.append(aircraft)
        return results

    def get_aircraft_by_position(self, bearing: float, distance: float,
                                 bearing_tolerance: float = 30,
                                 distance_tolerance: float = 5) -> List[Aircraft]:
        """Find aircraft by position relative to airport"""
        results = []
        for aircraft in self.current_aircraft.values():
            if isinstance(aircraft, Aircraft):
                if (abs(aircraft.bearing_from_airport - bearing) <= bearing_tolerance and
                        abs(aircraft.distance_from_airport - distance) <= distance_tolerance):
                    results.append(aircraft)
        return results