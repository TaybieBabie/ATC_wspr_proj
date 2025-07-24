# main.py
import argparse
import os
import time
import threading
import queue
from datetime import datetime
from record_audio import LiveATCRecorder, SystemAudioRecorder
from transcribe import transcribe_audio
from analyze import analyze_transcript
from config import (ATC_FREQUENCY, LIVEATC_STREAM_URL, VAD_THRESHOLD,
                    SILENCE_DURATION, AUDIO_DIR, OPENSKY_USERNAME, OPENSKY_PASSWORD,
                    ENABLE_ADSB, ADSB_SOURCE)
from adsb_tracker import ADSBTracker, OpenSkySource, LocalADSBSource
from correlator import ATCCorrelator
from console_logger import info, success, warning, error, section
from map_gui import MapApp


class ATCMonitor:
    """Main monitoring system that runs in a background thread."""

    def __init__(self, monitor_params: dict):
        self.params = monitor_params
        self.stream_url = monitor_params.get('stream_url') or LIVEATC_STREAM_URL
        self.use_system_audio = monitor_params.get('use_system_audio', False)
        self.vad_threshold = monitor_params.get('vad_threshold', VAD_THRESHOLD)
        self.silence_duration = monitor_params.get('silence_duration', SILENCE_DURATION)

        self.audio_queue = queue.Queue()
        self.is_monitoring = False
        self.gui_queue = None

        # Statistics
        self.stats = {
            'transmissions_recorded': 0,
            'transmissions_transcribed': 0,
            'non_transponder_alerts': 0,
            'callsigns_detected': set(),
            'start_time': None
        }

        # Initialize ADS-B tracking
        self.enable_adsb = ENABLE_ADSB
        if self.enable_adsb:
            if ADSB_SOURCE == 'opensky':
                source = OpenSkySource(OPENSKY_USERNAME, OPENSKY_PASSWORD)
            elif ADSB_SOURCE == 'local':
                source = LocalADSBSource()
            else:
                source = OpenSkySource()
            self.adsb_tracker = ADSBTracker(source)
            self.correlator = ATCCorrelator(self.adsb_tracker)

    def set_gui_queue(self, gui_queue: queue.Queue):
        """Set the queue for communicating with the GUI."""
        self.gui_queue = gui_queue

    def adsb_update_worker(self):
        """Background thread to update ADS-B data."""
        info("ADS-B updater thread started.", emoji="📡")
        while self.is_monitoring:
            try:
                aircraft_list = self.adsb_tracker.update_aircraft_positions()
                info(f"ADS-B Update: {len(aircraft_list)} aircraft in area")

                if self.gui_queue:
                    for aircraft in aircraft_list:
                        self.gui_queue.put(("update_aircraft", aircraft.to_dict()))

                time.sleep(15)
            except Exception as e:
                error(f"ADS-B update error: {e}")
                time.sleep(60)
        info("ADS-B updater thread stopped.")

    def transcription_worker(self):
        """Worker thread for processing audio files."""
        info("Transcription worker thread started.", emoji="📝")
        while self.is_monitoring:
            try:
                audio_file = self.audio_queue.get(timeout=1)
                info(f"Processing: {os.path.basename(audio_file)} [Queue: {self.audio_queue.qsize()}]", emoji="🔄")

                try:
                    result = transcribe_audio(audio_file, show_progress=False)
                    self.stats['transmissions_transcribed'] += 1

                    if result and result.get('text', '').strip():
                        transcript_text = result['text'].strip()
                        info(f"Transcript: \"{transcript_text}\"", emoji="🗣️")
                        transcript_file = self.save_transcript(audio_file, result)
                        analysis = analyze_transcript(transcript_file)
                        self.process_analysis(analysis, transcript_text)
                except Exception as e:
                    error(f"Error processing {audio_file}: {e}")
                finally:
                    self.audio_queue.task_done()
            except queue.Empty:
                continue
        info("Transcription worker stopped.")

    def start_monitoring(self):
        """Start all background monitoring tasks."""
        self.is_monitoring = True
        self.stats['start_time'] = datetime.now()

        section("Starting Background Monitoring", emoji="🚀")

        # Start worker threads
        transcription_thread = threading.Thread(target=self.transcription_worker, daemon=True)
        transcription_thread.start()

        if self.enable_adsb:
            adsb_thread = threading.Thread(target=self.adsb_update_worker, daemon=True)
            adsb_thread.start()

        # This part will block until the recorder finishes or is interrupted
        if self.use_system_audio:
            recorder = EnhancedSystemAudioRecorder(
                vad_threshold=self.vad_threshold,
                silence_duration=self.silence_duration,
                callback=self.recording_callback
            )
            recorder.record_system_audio_with_vad(frequency=ATC_FREQUENCY)
        else:
            recorder = EnhancedLiveATCRecorder(
                stream_url=self.stream_url,
                vad_threshold=self.vad_threshold,
                silence_duration=self.silence_duration,
                callback=self.recording_callback
            )
            recorder.record_with_vad(frequency=ATC_FREQUENCY, max_duration=self.params.get('duration'))

        # This will be reached when the recorder stops
        self.stop_monitoring()

    def stop_monitoring(self):
        """Signal all background threads to stop."""
        if self.is_monitoring:
            self.is_monitoring = False
            info("Stopping monitoring tasks...", emoji="🛑")
            self.print_statistics()

    def recording_callback(self, audio_file):
        """Callback when a new audio file is saved."""
        if self.is_monitoring:
            self.audio_queue.put(audio_file)
            self.stats['transmissions_recorded'] += 1

    def save_transcript(self, audio_file, transcript_result):
        """Save transcript to JSON file."""
        transcript_dir = 'transcripts'
        os.makedirs(transcript_dir, exist_ok=True)
        base_name = os.path.basename(audio_file).replace('.wav', '')
        transcript_file = os.path.join(transcript_dir, f"{base_name}_transcript.json")
        import json
        with open(transcript_file, 'w') as f:
            json.dump(transcript_result, f, indent=2)
        return transcript_file

    def process_analysis(self, analysis, transcript_text):
        """Process analysis results and generate alerts."""
        for callsign in analysis.get('overall_info', {}).get('callsigns', []):
            self.stats['callsigns_detected'].add(callsign)

        if self.enable_adsb:
            correlation = self.correlator.correlate_transcript(transcript_text, datetime.now())
            for alert in correlation['alerts']:
                self.stats['non_transponder_alerts'] += 1
                section(f"🚨 ALERT: {alert['type'].upper()} 🚨", emoji="🚨")
                info(f"Details: {alert}")
                info(f"Transcript: \"{transcript_text}\"")

    def print_statistics(self):
        """Print monitoring session statistics."""
        if self.stats['start_time']:
            duration = datetime.now() - self.stats['start_time']
            section("MONITORING SESSION STATISTICS", emoji="📊")
            info(f"Duration: {str(duration).split('.')[0]}")
            info(f"Transmissions recorded: {self.stats['transmissions_recorded']}")
            info(f"Transmissions transcribed: {self.stats['transmissions_transcribed']}")
            info(f"Potential non-transponder alerts: {self.stats['non_transponder_alerts']}")
            info(f"Unique callsigns detected: {len(self.stats['callsigns_detected'])}")


class EnhancedLiveATCRecorder(LiveATCRecorder):
    def __init__(self, stream_url, vad_threshold=0.01, silence_duration=2.0, callback=None):
        super().__init__(stream_url, vad_threshold, silence_duration)
        self.callback = callback

    def save_audio_segment(self, audio_data, frequency=None):
        filename = super().save_audio_segment(audio_data, frequency)
        if filename and self.callback:
            self.callback(filename)
        return filename


class EnhancedSystemAudioRecorder(SystemAudioRecorder):
    def __init__(self, vad_threshold=0.01, silence_duration=2.0, callback=None):
        super().__init__(vad_threshold, silence_duration)
        self.callback = callback

    def save_audio_segment(self, audio_data, frequency=None):
        filename = super().save_audio_segment(audio_data, frequency)
        if filename and self.callback:
            self.callback(filename)
        return filename


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
