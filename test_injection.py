# test_map_overlay.py
from gui.map_app_webview import OpenSkyMapApp
import queue
import threading
import time
import random
from datetime import datetime
from utils.config import AIRPORT_LAT, AIRPORT_LON
from utils.console_logger import info, success, warning


class MockMonitor:
    """Mock ATC monitor that simulates transmissions"""

    def __init__(self):
        self.gui_queue = None
        self.running = False
        self.simulation_thread = None

    def set_gui_queue(self, q):
        self.gui_queue = q
        info("GUI queue set, starting transmission simulation...")
        self.start_simulation()

    def start_simulation(self):
        """Start simulating ATC transmissions"""
        self.running = True
        self.simulation_thread = threading.Thread(target=self._simulate_transmissions, daemon=True)
        self.simulation_thread.start()

    def _simulate_transmissions(self):
        """Simulate various ATC transmissions"""
        time.sleep(5)  # Wait for overlay to initialize

        # Sample callsigns and messages
        callsigns = [
            "UAL123", "DAL456", "SWA789", "AAL321", "SKW234",
            "ASA567", "JBU890", "FDX123", "UPS456", "N12345"
        ]

        messages = [
            "Portland Tower, {} requesting taxi to runway 28R",
            "{} heavy, turn left heading 270, descend and maintain 3000",
            "Portland Approach, {} with you at 5000",
            "{}, cleared ILS approach runway 10R",
            "{}, contact ground on 121.9",
            "Portland Tower, {} ready for departure",
            "{}, wind 270 at 10, cleared for takeoff runway 28R",
            "{}, reduce speed to 180 knots",
            "{}, traffic 2 o'clock, 5 miles, opposite direction",
            "{}, roger, maintain visual separation"
        ]

        transmission_count = 0

        while self.running:
            # Random delay between transmissions (2-10 seconds)
            time.sleep(random.uniform(2, 10))

            # Pick random callsign(s)
            num_callsigns = random.randint(1, 2)
            selected_callsigns = random.sample(callsigns, num_callsigns)

            # Pick random message
            message_template = random.choice(messages)
            transcript = message_template.format(selected_callsigns[0])

            # Generate random position near airport (within monitoring radius)
            # Add some randomness to position
            lat_offset = random.uniform(-0.5, 0.5)  # roughly +/- 30 miles
            lon_offset = random.uniform(-0.5, 0.5)

            transmission_data = {
                'transcript': transcript,
                'callsigns': selected_callsigns,
                'timestamp': datetime.now().isoformat(),
                'lat': AIRPORT_LAT + lat_offset,
                'lon': AIRPORT_LON + lon_offset,
                'confidence': random.uniform(0.7, 1.0),
                'source': 'test_simulation'
            }

            if self.gui_queue:
                self.gui_queue.put(("atc_transmission", transmission_data))
                transmission_count += 1
                success(f"Sent transmission #{transmission_count}: {transcript[:50]}...")

                # Log additional details occasionally
                if transmission_count % 5 == 0:
                    info(f"Transmission details - Callsigns: {selected_callsigns}, "
                         f"Position: ({transmission_data['lat']:.4f}, {transmission_data['lon']:.4f})")

    def stop_monitoring(self):
        """Stop the simulation"""
        info("Stopping transmission simulation...")
        self.running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=2)


def test_interactive():
    """Run interactive test with manual control"""
    monitor = MockMonitor()
    app = OpenSkyMapApp(monitor)

    # Create control thread for interactive commands
    def control_thread():
        time.sleep(10)  # Wait for everything to initialize

        info("\n=== ATC Monitor Test Console ===")
        info("Commands:")
        info("  t - Send test transmission")
        info("  b - Send burst of transmissions")
        info("  s - Show status")
        info("  q - Quit")
        info("================================\n")

        while app.running:
            try:
                cmd = input("Command: ").strip().lower()

                if cmd == 'q':
                    info("Shutting down...")
                    app.stop()
                    break

                elif cmd == 't':
                    # Send single test transmission
                    test_data = {
                        'transcript': f"Test transmission at {datetime.now().strftime('%H:%M:%S')}",
                        'callsigns': ['TEST123'],
                        'timestamp': datetime.now().isoformat(),
                        'lat': AIRPORT_LAT,
                        'lon': AIRPORT_LON
                    }
                    if monitor.gui_queue:
                        monitor.gui_queue.put(("atc_transmission", test_data))
                        success("Test transmission sent!")

                elif cmd == 'b':
                    # Send burst of transmissions
                    info("Sending burst of 5 transmissions...")
                    for i in range(5):
                        test_data = {
                            'transcript': f"Burst transmission {i + 1}/5",
                            'callsigns': [f'BURST{i + 1}'],
                            'timestamp': datetime.now().isoformat(),
                            'lat': AIRPORT_LAT + random.uniform(-0.1, 0.1),
                            'lon': AIRPORT_LON + random.uniform(-0.1, 0.1)
                        }
                        if monitor.gui_queue:
                            monitor.gui_queue.put(("atc_transmission", test_data))
                        time.sleep(0.5)
                    success("Burst complete!")

                elif cmd == 's':
                    # Show status
                    info(f"Application running: {app.running}")
                    info(f"Overlay initialized: {app.overlay_initialized}")
                    info(f"Transmissions sent: {app.transmission_count}")

            except KeyboardInterrupt:
                break
            except Exception as e:
                warning(f"Command error: {e}")

    # Start control thread
    control = threading.Thread(target=control_thread, daemon=True)
    control.start()

    try:
        app.run()
    except KeyboardInterrupt:
        info("Interrupted by user")
    finally:
        monitor.stop_monitoring()


def test_automated():
    """Run automated test with predefined scenario"""
    info("Starting automated test scenario...")
    monitor = MockMonitor()
    app = OpenSkyMapApp(monitor)

    # Create automated test scenario
    def test_scenario():
        time.sleep(10)  # Wait for initialization

        info("\n=== Starting Automated Test Scenario ===")

        # Test 1: Single transmission
        info("Test 1: Single transmission")
        test_data = {
            'transcript': "Automated test transmission 1",
            'callsigns': ['AUTO001'],
            'timestamp': datetime.now().isoformat(),
            'lat': AIRPORT_LAT,
            'lon': AIRPORT_LON
        }
        monitor.gui_queue.put(("atc_transmission", test_data))
        time.sleep(3)

        # Test 2: Multiple callsigns
        info("Test 2: Multiple callsigns")
        test_data = {
            'transcript': "Multiple callsign test",
            'callsigns': ['AUTO002', 'AUTO003', 'AUTO004'],
            'timestamp': datetime.now().isoformat(),
            'lat': AIRPORT_LAT + 0.1,
            'lon': AIRPORT_LON - 0.1
        }
        monitor.gui_queue.put(("atc_transmission", test_data))
        time.sleep(3)

        # Test 3: Rapid succession
        info("Test 3: Rapid succession (10 transmissions)")
        for i in range(10):
            test_data = {
                'transcript': f"Rapid test {i + 1}/10",
                'callsigns': [f'RAPID{i + 1:03d}'],
                'timestamp': datetime.now().isoformat(),
                'lat': AIRPORT_LAT + random.uniform(-0.2, 0.2),
                'lon': AIRPORT_LON + random.uniform(-0.2, 0.2)
            }
            monitor.gui_queue.put(("atc_transmission", test_data))
            time.sleep(0.5)

        info("Test 4: Long transcript")
        test_data = {
            'transcript': "This is a very long transmission to test how the system handles extended transcripts. "
                          "Portland Tower, United 123 heavy requesting taxi to runway 28R via taxiway Alpha, "
                          "Bravo, and Charlie. We have information Yankee.",
            'callsigns': ['UAL123'],
            'timestamp': datetime.now().isoformat(),
            'lat': AIRPORT_LAT,
            'lon': AIRPORT_LON
        }
        monitor.gui_queue.put(("atc_transmission", test_data))
        time.sleep(5)

        success("\n=== Automated Test Complete ===")
        info(f"Total transmissions sent: {app.transmission_count}")

        # Keep running for 30 more seconds
        info("Continuing with random transmissions for 30 seconds...")
        time.sleep(30)

        info("Test complete, shutting down...")
        app.stop()

    # Start test scenario
    scenario = threading.Thread(target=test_scenario, daemon=True)
    scenario.start()

    try:
        app.run()
    except KeyboardInterrupt:
        info("Interrupted by user")
    finally:
        monitor.stop_monitoring()


if __name__ == "__main__":
    import sys

    # Check for command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == 'auto':
            test_automated()
        elif mode == 'interactive':
            test_interactive()
        else:
            print("Usage: python test_map_overlay.py [auto|interactive]")
            print("  auto        - Run automated test scenario")
            print("  interactive - Run with manual control")
            print("  (no args)   - Run basic continuous simulation")
    else:
        # Default: run basic simulation
        info("Running basic continuous simulation (use 'auto' or 'interactive' for other modes)")
        monitor = MockMonitor()
        app = OpenSkyMapApp(monitor)
        try:
            app.run()
        except KeyboardInterrupt:
            info("Shutting down...")
        finally:
            monitor.stop_monitoring()