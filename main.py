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
                    SILENCE_DURATION, AUDIO_DIR, OPENSKY_USERNAME, OPENSKY_PASSWORD)
from adsb_tracker import ADSBTracker, OpenSkySource, LocalADSBSource
from correlator import ATCCorrelator

class ATCMonitor:
    """Main monitoring system that coordinates recording, transcription, and analysis"""

    def __init__(self, stream_url=None, use_system_audio=False,
                 vad_threshold=VAD_THRESHOLD, silence_duration=SILENCE_DURATION,
                 enable_adsb=True, adsb_source='opensky'):
        self.stream_url = stream_url or LIVEATC_STREAM_URL
        self.use_system_audio = use_system_audio
        self.vad_threshold = vad_threshold
        self.silence_duration = silence_duration
        self.audio_queue = queue.Queue()
        self.is_monitoring = False
        self.transcription_thread = None

        # Statistics
        self.stats = {
            'transmissions_recorded': 0,
            'transmissions_transcribed': 0,
            'non_transponder_alerts': 0,
            'callsigns_detected': set(),
            'start_time': None
        }

        # Initialize ADS-B tracking
        self.enable_adsb = enable_adsb
        if self.enable_adsb:
            # Select ADS-B data source
            if adsb_source == 'opensky':
                source = OpenSkySource(OPENSKY_USERNAME, OPENSKY_PASSWORD)
            elif adsb_source == 'local':
                source = LocalADSBSource()
            else:
                source = OpenSkySource()  # default

            self.adsb_tracker = ADSBTracker(source)
            self.correlator = ATCCorrelator(self.adsb_tracker)

            # Start ADS-B update thread
            self.adsb_thread = threading.Thread(target=self.adsb_update_worker)
            self.adsb_thread.daemon = True
            self.adsb_thread.start()

    def adsb_update_worker(self):
        """Background thread to update ADS-B data"""
        while self.is_monitoring:
            try:
                aircraft = self.adsb_tracker.update_aircraft_positions()
                print(f"📡 ADS-B Update: {len(aircraft)} aircraft in area")
                time.sleep(30)  # Update every 30 seconds
            except Exception as e:
                print(f"❌ ADS-B update error: {e}")
                time.sleep(60)  # Wait longer on error

    def process_analysis(self, analysis, transcript_text):
        """Enhanced analysis with ADS-B correlation"""
        # Original analysis
        super().process_analysis(analysis, transcript_text)

        # ADS-B correlation
        if self.enable_adsb:
            correlation = self.correlator.correlate_transcript(
                transcript_text, datetime.now()
            )

            # Process correlation results
            if correlation['uncorrelated_callsigns']:
                print(f"⚠️  Uncorrelated callsigns: {', '.join(correlation['uncorrelated_callsigns'])}")

            # Process alerts
            for alert in correlation['alerts']:
                if alert['type'] == 'untracked_aircraft':
                    self.generate_enhanced_alert(alert, transcript_text, correlation)
                elif alert['type'] == 'possible_non_transponder':
                    self.generate_possible_alert(alert, transcript_text, correlation)

            # Update stats
            self.stats['correlation_alerts'] = self.stats.get('correlation_alerts', 0) + len(correlation['alerts'])

    def generate_enhanced_alert(self, alert, transcript, correlation):
        """Enhanced alert with ADS-B correlation data"""
        print("\n" + "=" * 60)
        print("🚨 ALERT: UNTRACKED AIRCRAFT DETECTED 🚨")
        print("=" * 60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Callsign: {alert['callsign']}")
        print(f"Reported altitude: {alert.get('altitude', 'Unknown')} ft")
        print(f"Transcript: {transcript}")
        print("\n📡 ADS-B Correlation:")
        print(f"  - Aircraft in area: {len(self.adsb_tracker.current_aircraft)}")
        print(f"  - No ADS-B match found for {alert['callsign']}")

        # List nearby aircraft
        if alert.get('altitude'):
            nearby = self.adsb_tracker.get_aircraft_at_altitude(
                alert['altitude'], tolerance=1000
            )
            if nearby:
                print(f"\n  Nearby aircraft at similar altitude:")
                for ac in nearby[:5]:  # Limit to 5
                    print(f"    - {ac}")

        print("=" * 60 + "\n")

        # Log to file
        self.log_alert(alert, transcript, correlation)

    def recording_callback(self, audio_file):
        """Callback when a new audio file is saved"""
        self.audio_queue.put(audio_file)
        self.stats['transmissions_recorded'] += 1

    def transcription_worker(self):
        """Worker thread for processing audio files"""
        while self.is_monitoring or not self.audio_queue.empty():
            try:
                # Wait for audio files with timeout
                audio_file = self.audio_queue.get(timeout=1)

                print(f"\n🔄 Processing: {os.path.basename(audio_file)}")

                # Transcribe
                try:
                    result = transcribe_audio(audio_file)
                    self.stats['transmissions_transcribed'] += 1

                    if result and result.get('text', '').strip():
                        print(f"📝 Transcript: {result['text']}")

                        # Save transcript
                        transcript_file = self.save_transcript(audio_file, result)

                        # Analyze
                        analysis = analyze_transcript(transcript_file)

                        # Process analysis results
                        self.process_analysis(analysis, result['text'])
                    else:
                        print("❌ No speech detected in transmission")
                        # Optionally delete empty recordings
                        # os.remove(audio_file)

                except Exception as e:
                    print(f"❌ Error processing {audio_file}: {e}")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Worker thread error: {e}")

    def save_transcript(self, audio_file, transcript_result):
        """Save transcript to JSON file"""
        transcript_dir = 'transcripts'
        if not os.path.exists(transcript_dir):
            os.makedirs(transcript_dir)

        base_name = os.path.basename(audio_file).replace('.wav', '')
        transcript_file = os.path.join(transcript_dir, f"{base_name}_transcript.json")

        import json
        with open(transcript_file, 'w') as f:
            json.dump(transcript_result, f, indent=2)

        return transcript_file

    def process_analysis(self, analysis, transcript_text):
        """Process analysis results and generate alerts"""
        # Update statistics
        for callsign in analysis.get('overall_info', {}).get('callsigns', []):
            self.stats['callsigns_detected'].add(callsign)

        # Check for non-transponder indicators
        non_transponder_keywords = [
            'primary target', 'no transponder', 'primary only',
            'radar contact', 'unidentified', '1200', 'vfr'
        ]

        transcript_lower = transcript_text.lower()
        alert_triggered = False

        for keyword in non_transponder_keywords:
            if keyword in transcript_lower:
                alert_triggered = True
                self.stats['non_transponder_alerts'] += 1
                self.generate_alert(transcript_text, keyword, analysis)
                break

        # Display analysis summary
        if not alert_triggered:
            callsigns = analysis.get('overall_info', {}).get('callsigns', [])
            altitudes = analysis.get('overall_info', {}).get('altitudes', [])
            if callsigns or altitudes:
                print(f"ℹ️  Detected: {len(callsigns)} callsigns, {len(altitudes)} altitudes")

    def generate_alert(self, transcript, keyword, analysis):
        """Generate alert for potential non-transponder aircraft"""
        print("\n" + "=" * 60)
        print("🚨 ALERT: POTENTIAL NON-TRANSPONDER AIRCRAFT DETECTED 🚨")
        print("=" * 60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Keyword detected: '{keyword}'")
        print(f"Full transcript: {transcript}")

        callsigns = analysis.get('overall_info', {}).get('callsigns', [])
        if callsigns:
            print(f"Callsigns mentioned: {', '.join(callsigns)}")

        print("=" * 60 + "\n")

        # Here you could add:
        # - Email notification
        # - SMS alert
        # - Log to file
        # - Play alert sound

    def start_monitoring(self, duration=None):
        """Start the monitoring system"""
        self.is_monitoring = True
        self.stats['start_time'] = datetime.now()

        print(f"\n🚀 Starting ATC Monitor")
        print(f"📻 Frequency: {ATC_FREQUENCY}")
        print(f"🎯 VAD Threshold: {self.vad_threshold}")
        print(f"⏱️  Silence Duration: {self.silence_duration}s")
        print(f"📡 Source: {'System Audio' if self.use_system_audio else self.stream_url}")
        print("\n" + "-" * 60)

        # Start transcription worker thread
        self.transcription_thread = threading.Thread(target=self.transcription_worker)
        self.transcription_thread.start()

        # Enhanced recorder with callback
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

            if duration:
                recorder.record_with_vad(frequency=ATC_FREQUENCY, max_duration=duration)
            else:
                recorder.record_with_vad(frequency=ATC_FREQUENCY)

        # Stop monitoring
        self.is_monitoring = False

        # Wait for transcription thread to finish
        if self.transcription_thread:
            print("\n⏳ Waiting for remaining transcriptions to complete...")
            self.transcription_thread.join()

        self.print_statistics()

    def print_statistics(self):
        """Print monitoring session statistics"""
        if self.stats['start_time']:
            duration = datetime.now() - self.stats['start_time']

            print("\n" + "=" * 60)
            print("📊 MONITORING SESSION STATISTICS")
            print("=" * 60)
            print(f"Duration: {duration}")
            print(f"Transmissions recorded: {self.stats['transmissions_recorded']}")
            print(f"Transmissions transcribed: {self.stats['transmissions_transcribed']}")
            print(f"Non-transponder alerts: {self.stats['non_transponder_alerts']}")
            print(f"Unique callsigns detected: {len(self.stats['callsigns_detected'])}")
            if self.stats['callsigns_detected']:
                print(f"Callsigns: {', '.join(sorted(self.stats['callsigns_detected']))}")
            print("=" * 60)


# Enhanced recorder classes with callbacks
class EnhancedLiveATCRecorder(LiveATCRecorder):
    def __init__(self, stream_url, vad_threshold=0.01, silence_duration=2.0, callback=None):
        super().__init__(stream_url, vad_threshold, silence_duration)
        self.callback = callback

    def save_audio_segment(self, audio_data, frequency=None):
        """Override to add callback"""
        filename = super().save_audio_segment(audio_data, frequency)
        if filename and self.callback:
            self.callback(filename)
        return filename


class EnhancedSystemAudioRecorder(SystemAudioRecorder):
    def __init__(self, vad_threshold=0.01, silence_duration=2.0, callback=None):
        super().__init__(vad_threshold, silence_duration)
        self.callback = callback

    def save_audio_segment(self, audio_data, frequency=None):
        """Override to add callback"""
        filename = super().save_audio_segment(audio_data, frequency)
        if filename and self.callback:
            self.callback(filename)
        return filename


def main():
    parser = argparse.ArgumentParser(
        description='ATC Communication Monitoring, Transcription and Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor LiveATC stream continuously
  python main.py --monitor

  # Monitor for 1 hour with custom VAD threshold
  python main.py --monitor --duration 3600 --vad-threshold 0.03

  # Monitor system audio instead of stream
  python main.py --monitor --system-audio

  # Run legacy single recording
  python main.py --record 60

  # Transcribe a specific file
  python main.py --transcribe audio/transmission_20250114_123456.wav
        """
    )

    # New monitoring mode
    parser.add_argument('--monitor', action='store_true',
                        help='Start continuous monitoring with VAD')
    parser.add_argument('--duration', type=int,
                        help='Monitoring duration in seconds (default: continuous)')
    parser.add_argument('--vad-threshold', type=float, default=VAD_THRESHOLD,
                        help=f'Voice activity detection threshold (default: {VAD_THRESHOLD})')
    parser.add_argument('--silence-duration', type=float, default=SILENCE_DURATION,
                        help=f'Seconds of silence before ending recording (default: {SILENCE_DURATION})')
    parser.add_argument('--stream-url', type=str, default=LIVEATC_STREAM_URL,
                        help='LiveATC stream URL')
    parser.add_argument('--system-audio', action='store_true',
                        help='Record from system audio instead of stream')

    # Legacy modes
    parser.add_argument('--record', type=int,
                        help='Record audio for specified duration in seconds (legacy mode)')
    parser.add_argument('--transcribe', type=str,
                        help='Transcribe specified audio file')
    parser.add_argument('--analyze', type=str,
                        help='Analyze specified transcript file')
    parser.add_argument('--pipeline', type=int,
                        help='Run full pipeline: record, transcribe, and analyze (legacy mode)')

    args = parser.parse_args()

    # New monitoring mode
    if args.monitor:
        monitor = ATCMonitor(
            stream_url=args.stream_url,
            use_system_audio=args.system_audio,
            vad_threshold=args.vad_threshold,
            silence_duration=args.silence_duration
        )

        try:
            monitor.start_monitoring(duration=args.duration)
        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring stopped by user")
            monitor.is_monitoring = False

    # Legacy modes
    elif args.record:
        from record_audio import record_audio
        audio_file = record_audio(duration=args.record, frequency=ATC_FREQUENCY)

    elif args.pipeline:
        print("Running full pipeline...")
        from record_audio import record_audio
        audio_file = record_audio(duration=args.pipeline, frequency=ATC_FREQUENCY)
        transcript = transcribe_audio(audio_file)
        transcript_file = os.path.join('transcripts',
                                       os.path.basename(audio_file).replace('.wav', '_transcript.json'))
        analysis = analyze_transcript(transcript_file)
        print("\nTranscription:")
        print(transcript['text'])
        print("\nAnalysis:")
        print(f"Detected {len(analysis['overall_info']['callsigns'])} callsigns")
        print(f"Detected {len(analysis['overall_info']['altitudes'])} altitude instructions")

    elif args.transcribe:
        transcribe_audio(args.transcribe)

    elif args.analyze:
        analyze_transcript(args.analyze)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()