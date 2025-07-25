#!/usr/bin/env python3
"""
ATC Communication Monitor - Main Entry Point

This module serves as the main entry point for the ATC monitoring application.
It handles command-line arguments and orchestrates the application startup,
delegating actual functionality to specialized modules.
"""
import argparse
import threading

from core.monitor import ATCMonitor
from gui.map_gui import MapApp
from utils.console_logger import info
from utils.config import (VAD_THRESHOLD, SILENCE_DURATION, LIVEATC_STREAM_URL)


def main():
    """Main entry point for the ATC monitoring application."""
    parser = argparse.ArgumentParser(description='ATC Communication Monitor')
    parser.add_argument('--monitor', action='store_true', help='Start continuous monitoring')
    parser.add_argument('--duration', type=int, help='Monitoring duration in seconds')
    parser.add_argument('--vad-threshold', type=float, default=VAD_THRESHOLD)
    parser.add_argument('--silence-duration', type=float, default=SILENCE_DURATION)
    parser.add_argument('--stream-url', type=str, default=LIVEATC_STREAM_URL)
    parser.add_argument('--system-audio', action='store_true')
    args = parser.parse_args()

    if args.monitor:
        # 1. Create the business logic object
        monitor_params = vars(args)
        atc_monitor = ATCMonitor(monitor_params)

        # 2. Start the business logic in a background thread
        monitor_thread = threading.Thread(target=atc_monitor.start_monitoring, daemon=True)
        monitor_thread.start()

        # 3. Create and run the GUI in the main thread
        app = MapApp(atc_monitor)
        app.mainloop()

        # This code will run after the GUI window is closed
        info("Application has been closed.")
        # The daemon threads will exit automatically

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
