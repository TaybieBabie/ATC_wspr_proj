import tkinter
import tkinter.font
import queue
import math
from tkintermapview import TkinterMapView
from collections import deque
from datetime import datetime, timedelta
from utils.config import AIRPORT_LAT, AIRPORT_LON


class MapApp(tkinter.Tk):
    """
    The main application class for the map GUI.
    This runs on the main application thread.
    """
    APP_NAME = "ATC ADS-B Monitor"
    WIDTH = 1200
    HEIGHT = 800

    def __init__(self, atc_monitor, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title(self.APP_NAME)
        self.geometry(f"{self.WIDTH}x{self.HEIGHT}")
        self.minsize(self.WIDTH, self.HEIGHT)

        # Store a reference to the background worker object
        self.atc_monitor = atc_monitor

        # Set up a queue for thread-safe communication
        self.update_queue = queue.Queue()
        self.atc_monitor.set_gui_queue(self.update_queue)

        # Dictionaries to keep track of map objects
        self.aircraft_markers = {}
        self.aircraft_trails = {}  # Store trail paths for each aircraft
        self.aircraft_history = {}  # Store position history for each aircraft
        self.non_transponder_paths = {}

        # Configuration
        self.trail_length = 10  # Number of positions to keep in trail
        self.trail_max_age = 120  # Maximum age of trail points in seconds

        # --- Layout ---
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Map widget
        self.map_widget = TkinterMapView(self, corner_radius=0)
        self.map_widget.grid(row=0, column=0, sticky="nsew")

        # Control panel
        self.control_frame = tkinter.Frame(self)
        self.control_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)

        # Trail toggle
        self.show_trails = tkinter.BooleanVar(value=True)
        self.trail_checkbox = tkinter.Checkbutton(
            self.control_frame,
            text="Show Trails",
            variable=self.show_trails,
            command=self.toggle_trails
        )
        self.trail_checkbox.pack(side="left", padx=5)

        # --- Initial Setup ---
        self.map_widget.set_position(AIRPORT_LAT, AIRPORT_LON)
        self.map_widget.set_zoom(10)

        # Airport marker with custom icon
        self.map_widget.set_marker(
            AIRPORT_LAT, AIRPORT_LON,
            text="Airport",
            marker_color_circle="#FF0000",
            marker_color_outside="#8B0000"
        )

        # Handle window closing
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start the queue processor
        self.process_queue()

    def create_aircraft_icon(self, heading):
        """
        Create a directional triangle/arrow shape for aircraft.
        Returns a list of points forming a triangle pointing in the heading direction.
        """
        # Triangle size in degrees (adjust based on zoom level)
        size = 0.003

        # Convert heading to radians (0° is North, clockwise)
        angle_rad = math.radians(heading - 90)  # Adjust so 0° points up

        # Define triangle points (pointing right initially)
        points = [
            (size, 0),  # Nose
            (-size / 2, size / 2),  # Top wing
            (-size / 2, -size / 2),  # Bottom wing
        ]

        # Rotate points by heading
        rotated_points = []
        for x, y in points:
            rx = x * math.cos(angle_rad) - y * math.sin(angle_rad)
            ry = x * math.sin(angle_rad) + y * math.cos(angle_rad)
            rotated_points.append((rx, ry))

        return rotated_points

    def update_aircraft_marker(self, aircraft_data: dict):
        """Adds or updates an aircraft marker on the map with directional icon."""
        icao = aircraft_data['icao24']
        lat = aircraft_data['latitude']
        lon = aircraft_data['longitude']
        callsign = aircraft_data.get('callsign', 'N/A').strip()
        altitude = aircraft_data.get('altitude', 0)
        heading = aircraft_data.get('track', 0)
        ground_speed = aircraft_data.get('ground_speed', 0)
        vertical_rate = aircraft_data.get('vertical_rate', 0)
        on_ground = aircraft_data.get('on_ground', False)

        # Update position history
        if icao not in self.aircraft_history:
            self.aircraft_history[icao] = deque(maxlen=self.trail_length)

        # Add new position with timestamp
        self.aircraft_history[icao].append({
            'lat': lat,
            'lon': lon,
            'time': datetime.now()
        })

        # Clean old positions
        self._clean_old_positions(icao)

        # Update trail if enabled
        if self.show_trails.get():
            self._update_trail(icao)

        # Determine marker color based on status
        if on_ground:
            marker_color = "#808080"  # Gray for ground
        elif vertical_rate > 100:
            marker_color = "#00FF00"  # Green for climbing
        elif vertical_rate < -100:
            marker_color = "#FF8C00"  # Orange for descending
        else:
            marker_color = "#0080FF"  # Blue for level flight

        # Create marker text with more info
        marker_text = f"{callsign}\n{int(altitude)}ft\n{int(ground_speed)}kts"

        # Update or create marker
        if icao in self.aircraft_markers:
            marker = self.aircraft_markers[icao]
            marker.set_position(lat, lon)
            marker.set_text(marker_text)
        else:
            # Create new marker with custom appearance
            new_marker = self.map_widget.set_marker(
                lat, lon,
                text=marker_text,
                font=tkinter.font.Font(size=8, weight="bold"),
                marker_color_circle=marker_color,
                marker_color_outside="#000000"
            )
            self.aircraft_markers[icao] = new_marker

    def _clean_old_positions(self, icao):
        """Remove positions older than trail_max_age seconds."""
        if icao in self.aircraft_history:
            current_time = datetime.now()
            history = self.aircraft_history[icao]

            # Remove old positions
            while history and (current_time - history[0]['time']).total_seconds() > self.trail_max_age:
                history.popleft()

    def _update_trail(self, icao):
        """Update the trail path for an aircraft."""
        if icao not in self.aircraft_history or len(self.aircraft_history[icao]) < 2:
            return

        # Remove old trail if exists
        if icao in self.aircraft_trails:
            self.aircraft_trails[icao].delete()

        # Create position list from history
        positions = [(pos['lat'], pos['lon']) for pos in self.aircraft_history[icao]]

        # Create trail with gradient effect (older positions more transparent)
        trail = self.map_widget.set_path(
            position_list=positions,
            color="#4169E1",  # Royal blue
            width=2
        )
        self.aircraft_trails[icao] = trail

    def toggle_trails(self):
        """Toggle trail visibility."""
        if self.show_trails.get():
            # Re-create all trails
            for icao in self.aircraft_history:
                self._update_trail(icao)
        else:
            # Remove all trails
            for trail in self.aircraft_trails.values():
                trail.delete()
            self.aircraft_trails.clear()

    def draw_non_transponder_radius(self, lat: float, lon: float, radius_nm: float, text: str, alert_id: str):
        """Draws a circle on the map for a non-transponder alert."""
        if alert_id in self.non_transponder_paths:
            self.non_transponder_paths[alert_id].delete()

        # Convert nautical miles to degrees (approximate)
        radius_deg = radius_nm / 60.0

        # Create circle points
        num_points = 36
        circle_points = []
        for i in range(num_points + 1):
            angle = 2 * math.pi * i / num_points
            point_lat = lat + radius_deg * math.cos(angle)
            point_lon = lon + radius_deg * math.sin(angle) / math.cos(math.radians(lat))
            circle_points.append((point_lat, point_lon))

        path = self.map_widget.set_path(
            position_list=circle_points,
            color="#FF0000",
            width=3
        )
        self.non_transponder_paths[alert_id] = path

        # Add alert marker
        self.map_widget.set_marker(
            lat, lon,
            text=text,
            font=tkinter.font.Font(size=10, weight="bold"),
            marker_color_circle="#FF0000",
            marker_color_outside="#8B0000"
        )

    def clear_all_aircraft(self):
        """Removes all aircraft markers and trails from the map."""
        # Clear markers
        for marker in self.aircraft_markers.values():
            marker.delete()
        self.aircraft_markers.clear()

        # Clear trails
        for trail in self.aircraft_trails.values():
            trail.delete()
        self.aircraft_trails.clear()

        # Clear history
        self.aircraft_history.clear()

        # Clear alert paths
        for path in self.non_transponder_paths.values():
            path.delete()
        self.non_transponder_paths.clear()

    def process_queue(self):
        """
        Process items from the thread-safe queue to update the GUI.
        """
        try:
            while not self.update_queue.empty():
                message = self.update_queue.get_nowait()
                command = message[0]
                data = message[1]

                if command == "update_aircraft":
                    self.update_aircraft_marker(data)
                elif command == "draw_radius":
                    self.draw_non_transponder_radius(**data)
                elif command == "clear":
                    self.clear_all_aircraft()

        except queue.Empty:
            pass
        finally:
            # Reschedule itself to run again after 100ms
            self.after(100, self.process_queue)

    def on_closing(self):
        """Handle window closing event."""
        print("GUI closing, stopping background tasks...")
        # Signal the background thread to stop
        self.atc_monitor.stop_monitoring()
        # Close the GUI window
        self.destroy()