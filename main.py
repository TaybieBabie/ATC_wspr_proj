import argparse
import threading
from core.monitor import ATCMonitor
from utils.config import VAD_THRESHOLD, SILENCE_DURATION, LIVEATC_STREAM_URL
from utils.console_logger import section


def main():
    parser = argparse.ArgumentParser(description='ATC Communication Monitor')
    parser.add_argument('--monitor', action='store_true', help='Start continuous monitoring')
    parser.add_argument('--duration', type=int, help='Monitoring duration in seconds')
    parser.add_argument('--vad-threshold', type=float, default=VAD_THRESHOLD)
    parser.add_argument('--silence-duration', type=float, default=SILENCE_DURATION)
    parser.add_argument('--stream-url', type=str, default=LIVEATC_STREAM_URL)
    parser.add_argument('--system-audio', action='store_true')
    args = parser.parse_args()

    if args.monitor:
        section("ATC Monitor - OpenSky Integration", emoji="✈️")

        # Create the monitor
        monitor_params = vars(args)
        atc_monitor = ATCMonitor(monitor_params)

        # Start monitoring in background thread
        monitor_thread = threading.Thread(target=atc_monitor.start_monitoring, daemon=True)
        monitor_thread.start()

        from gui.map_app_webview import run_webview_app
        run_webview_app(atc_monitor)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()