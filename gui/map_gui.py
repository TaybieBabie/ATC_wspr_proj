# map_gui.py
import tkinter
import tkinter.font
import queue
from tkintermapview import TkinterMapView
from utils.config import AIRPORT_LAT, AIRPORT_LON


class MapApp(tkinter.Tk):
    """
    The main application class for the map GUI.
    This runs on the main application thread.
    """
    APP_NAME = "ATC ADS-B Monitor"
    WIDTH = 1000
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
        self.non_transponder_paths = {}

        # --- Layout ---
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Map widget
        self.map_widget = TkinterMapView(self, corner_radius=0)
        self.map_widget.grid(row=0, column=0, sticky="nsew")

        # --- Initial Setup ---
        self.map_widget.set_position(AIRPORT_LAT, AIRPORT_LON)
        self.map_widget.set_zoom(10)
        self.map_widget.set_marker(AIRPORT_LAT, AIRPORT_LON, text="Airport")

        # Handle window closing
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start the queue processor
        self.process_queue()

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

    def update_aircraft_marker(self, aircraft_data: dict):
        """Adds or updates an aircraft marker on the map."""
        icao = aircraft_data['icao24']
        lat = aircraft_data['latitude']
        lon = aircraft_data['longitude']
        callsign = aircraft_data.get('callsign', 'N/A').strip()
        altitude = aircraft_data.get('altitude', 0)

        marker_text = f"{callsign}\n{int(altitude)} ft"

        if icao in self.aircraft_markers:
            marker = self.aircraft_markers[icao]
            marker.set_position(lat, lon)
            marker.set_text(marker_text)
        else:
            new_marker = self.map_widget.set_marker(
                lat, lon, text=marker_text,
                font=tkinter.font.Font(size=9)
            )
            self.aircraft_markers[icao] = new_marker

    def draw_non_transponder_radius(self, lat: float, lon: float, radius_nm: float, text: str, alert_id: str):
        """Draws a circle on the map for a non-transponder alert."""
        if alert_id in self.non_transponder_paths:
            self.non_transponder_paths[alert_id].delete()

        radius_deg = radius_nm / 60.0
        circle_points = self.map_widget.get_circle_points(lat, lon, radius_deg)

        path = self.map_widget.set_path(
            position_list=circle_points,
            color="#FF0000",
            width=2
        )
        self.non_transponder_paths[alert_id] = path
        self.map_widget.set_marker(lat, lon, text=text, text_color="#FF0000")

    def clear_all_aircraft(self):
        """Removes all aircraft markers from the map."""
        for marker in self.aircraft_markers.values():
            marker.delete()
        self.aircraft_markers.clear()

        for path in self.non_transponder_paths.values():
            path.delete()
        self.non_transponder_paths.clear()

    def on_closing(self):
        """Handle window closing event."""
        print("GUI closing, stopping background tasks...")
        # Signal the background thread to stop
        self.atc_monitor.stop_monitoring()
        # Close the GUI window
        self.destroy()
