# test_injection_interactive.py
import tkinter as tk
from tkinter import ttk
import threading
from datetime import datetime
from gui.map_app_webview import OpenSkyMapApp
from utils.console_logger import info, success
from utils.config import AIRPORT_LAT, AIRPORT_LON
import random
import math


class MockATCMonitor:
    """Mock ATC Monitor for testing"""

    def __init__(self):
        self.gui_queue = None
        self.running = False

    def set_gui_queue(self, queue):
        self.gui_queue = queue

    def start_monitoring(self):
        self.running = True

    def stop_monitoring(self):
        self.running = False


class TestControlPanel:
    """Control panel for testing injections"""

    def __init__(self, app):
        self.app = app
        self.root = tk.Tk()
        self.root.title("ATC Injection Test Control")
        self.root.geometry("400x500")

        # Transmission counter
        self.transmission_count = 0

        # Create UI
        self.create_ui()

    def create_ui(self):
        """Create the control panel UI"""
        # Title
        title = ttk.Label(self.root, text="ATC Injection Test Control",
                          font=('Arial', 16, 'bold'))
        title.pack(pady=10)

        # Custom transmission frame
        custom_frame = ttk.LabelFrame(self.root, text="Custom Transmission", padding=10)
        custom_frame.pack(fill='x', padx=10, pady=5)

        # Transcript entry
        ttk.Label(custom_frame, text="Transcript:").pack(anchor='w')
        self.transcript_entry = tk.Text(custom_frame, height=3, width=40)
        self.transcript_entry.pack(fill='x', pady=5)
        self.transcript_entry.insert('1.0', "United 123, turn left heading 270")

        # Callsigns entry
        ttk.Label(custom_frame, text="Callsigns (comma separated):").pack(anchor='w')
        self.callsigns_entry = ttk.Entry(custom_frame, width=40)
        self.callsigns_entry.pack(fill='x', pady=5)
        self.callsigns_entry.insert(0, "UAL123, N456AB")

        # Position controls
        position_frame = ttk.Frame(custom_frame)
        position_frame.pack(fill='x', pady=5)

        ttk.Label(position_frame, text="Distance (nm):").grid(row=0, column=0, padx=5)
        self.distance_var = tk.DoubleVar(value=10)
        distance_scale = ttk.Scale(position_frame, from_=0, to=30,
                                   variable=self.distance_var, orient='horizontal')
        distance_scale.grid(row=0, column=1, padx=5, sticky='ew')

        distance_label = ttk.Label(position_frame, text="10.0")
        distance_label.grid(row=0, column=2, padx=5)

        def update_distance(value):
            distance_label.config(text=f"{float(value):.1f}")

        distance_scale.config(command=update_distance)

        position_frame.columnconfigure(1, weight=1)

        # Send button
        send_btn = ttk.Button(custom_frame, text="Send Custom Transmission",
                              command=self.send_custom_transmission)
        send_btn.pack(pady=10)

        # Preset transmissions
        preset_frame = ttk.LabelFrame(self.root, text="Preset Transmissions", padding=10)
        preset_frame.pack(fill='both', expand=True, padx=10, pady=5)

        presets = [
            ("Approach", "Contact tower 118.1", ["AAL789"]),
            ("Landing", "Cleared to land runway 28 right", ["SWA321"]),
            ("Taxi", "Taxi to runway 10 left via alpha", ["FDX123"]),
            ("Departure", "Contact departure 124.35", ["UAL456"]),
            ("Hold Short", "Hold short runway 10 left", ["UPS789"]),
            ("Go Around", "Go around, fly runway heading", ["DAL123"]),
            ("Weather", "Information charlie now current", []),
            ("Emergency", "Emergency vehicles standing by", ["N123EF"]),
        ]

        for name, transcript, callsigns in presets:
            btn = ttk.Button(preset_frame, text=name,
                             command=lambda t=transcript, c=callsigns: self.send_preset(t, c))
            btn.pack(fill='x', pady=2)

        # Stats
        stats_frame = ttk.LabelFrame(self.root, text="Statistics", padding=10)
        stats_frame.pack(fill='x', padx=10, pady=5)

        self.stats_label = ttk.Label(stats_frame, text="Transmissions sent: 0")
        self.stats_label.pack()

        # Auto-generate controls
        auto_frame = ttk.LabelFrame(self.root, text="Auto Generate", padding=10)
        auto_frame.pack(fill='x', padx=10, pady=5)

        self.auto_var = tk.BooleanVar(value=False)
        auto_check = ttk.Checkbutton(auto_frame, text="Auto-generate transmissions",
                                     variable=self.auto_var, command=self.toggle_auto)
        auto_check.pack()

        self.auto_thread = None

    def send_custom_transmission(self):
        """Send custom transmission"""
        transcript = self.transcript_entry.get('1.0', 'end-1c').strip()
        callsigns_text = self.callsigns_entry.get().strip()
        callsigns = [cs.strip() for cs in callsigns_text.split(',')] if callsigns_text else []

        # Generate position
        distance = self.distance_var.get()
        angle = random.uniform(0, 2 * math.pi)
        lat = AIRPORT_LAT + (distance / 60.0) * math.cos(angle)
        lon = AIRPORT_LON + (distance / 60.0) * math.sin(angle)

        self.send_transmission(transcript, callsigns, lat, lon)

    def send_preset(self, transcript, callsigns):
        """Send preset transmission"""
        # Random position
        distance = random.uniform(5, 20)
        angle = random.uniform(0, 2 * math.pi)
        lat = AIRPORT_LAT + (distance / 60.0) * math.cos(angle)
        lon = AIRPORT_LON + (distance / 60.0) * math.sin(angle)

        self.send_transmission(transcript, callsigns, lat, lon)

    def send_transmission(self, transcript, callsigns, lat, lon):
        """Send transmission to the app"""
        data = {
            'transcript': transcript,
            'callsigns': callsigns,
            'timestamp': datetime.now().isoformat(),
            'lat': lat,
            'lon': lon,
        }

        if self.app.atc_monitor.gui_queue:
            self.app.atc_monitor.gui_queue.put(("atc_transmission", data))
            self.transmission_count += 1
            self.stats_label.config(text=f"Transmissions sent: {self.transmission_count}")
            success(f"Sent: {transcript[:50]}...")

    def toggle_auto(self):
        """Toggle auto-generation"""
        if self.auto_var.get():
            self.start_auto_generate()
        else:
            self.stop_auto_generate()

    def start_auto_generate(self):
        """Start auto-generating transmissions"""

        def auto_generate():
            transmissions = [
                ("United 123, contact tower", ["UAL123"]),
                ("American 789, turn left heading 270", ["AAL789"]),
                ("Southwest 321, descend to 4000", ["SWA321"]),
                ("Delta 456, maintain 10000", ["DAL456"]),
                ("FedEx 123, expedite climb", ["FDX123"]),
            ]

            while self.auto_var.get():
                transcript, callsigns = random.choice(transmissions)
                self.send_preset(transcript, callsigns)
                time.sleep(random.uniform(3, 8))

        self.auto_thread = threading.Thread(target=auto_generate, daemon=True)
        self.auto_thread.start()
        info("Auto-generation started")

    def stop_auto_generate(self):
        """Stop auto-generating"""
        info("Auto-generation stopped")

    def run(self):
        """Run the control panel"""
        self.root.mainloop()


def test_with_control_panel():
    """Run the test with control panel"""

    # Create mock monitor
    mock_monitor = MockATCMonitor()

    # Create the app
    app = OpenSkyMapApp(mock_monitor)

    # Create control panel
    control_panel = TestControlPanel(app)

    # Run control panel in separate thread
    control_thread = threading.Thread(target=control_panel.run, daemon=True)
    control_thread.start()

    # Give control panel time to start
    threading.Thread(
        target=lambda: (time.sleep(2), info("Control panel ready - use it to send test transmissions")),
        daemon=True
    ).start()

    # Run the app
    try:
        info("Starting OpenSky map with test control panel...")
        app.run()
    except KeyboardInterrupt:
        info("Test stopped")
        app.stop()


if __name__ == "__main__":
    import time

    test_with_control_panel()