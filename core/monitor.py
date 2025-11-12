"""
monitor.py
-ATC Monitor Core Logic

This module contains the main business logic for the ATC monitoring system.
It coordinates between different subsystems (audio recording, transcription, 
analysis, and tracking) while maintaining clear separation of concerns.
"""
import os
import time
import threading
import queue
from datetime import datetime

from audio.recorders import EnhancedLiveATCRecorder, EnhancedSystemAudioRecorder
from transcription.transcriber import transcribe_audio
from analysis.analyzer import analyze_transcript
from tracking.adsb_tracker import ADSBTracker, OpenSkySource, LocalADSBSource
from analysis.correlator import ATCCorrelator
from analysis.ollama_correlator import (
    OllamaCorrelator as OllamaLLMCorrelator,
    build_adsb_contacts,
    build_atc_transmission,
)
from utils.console_logger import info, success, warning, error, section
from utils.config import (
    ATC_FREQUENCY,
    LIVEATC_STREAM_URL,
    VAD_THRESHOLD,
    SILENCE_DURATION,
    AUDIO_DIR,
    OPENSKY_CREDENTIALS_FILE,
    ENABLE_ADSB,
    ADSB_SOURCE,
    ENABLE_LLM_CORRELATION,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_REQUEST_TIMEOUT,
    LLM_MAX_ADSB_CONTACTS,
    LLM_MAX_TRANSMISSIONS,
)


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

        self.transmission_lock = threading.Lock()
        self.transmissions_history = []
        self.last_llm_result = None
        self.llm_correlator = None
        self.max_llm_history = max(LLM_MAX_TRANSMISSIONS * 3, LLM_MAX_TRANSMISSIONS or 1)

        # Initialize ADS-B tracking
        self.enable_adsb = ENABLE_ADSB
        if self.enable_adsb:
            if ADSB_SOURCE == 'opensky':
                source = OpenSkySource(OPENSKY_CREDENTIALS_FILE)
            elif ADSB_SOURCE == 'local':
                source = LocalADSBSource()
            else:
                source = OpenSkySource()
            self.adsb_tracker = ADSBTracker(source)
            self.correlator = ATCCorrelator(self.adsb_tracker)

            if ENABLE_LLM_CORRELATION:
                try:
                    self.llm_correlator = OllamaLLMCorrelator(
                        model=OLLAMA_MODEL,
                        base_url=OLLAMA_BASE_URL,
                        max_adsb_contacts=LLM_MAX_ADSB_CONTACTS,
                        max_transmissions=LLM_MAX_TRANSMISSIONS,
                        request_timeout=OLLAMA_REQUEST_TIMEOUT,
                    )
                    self.max_llm_history = max(
                        self.max_llm_history,
                        self.llm_correlator.context_builder.max_tx * 3,
                    )
                except Exception as exc:
                    warning(f"Failed to initialize LLM correlator: {exc}", emoji="⚠️")

    def set_gui_queue(self, gui_queue: queue.Queue):
        """Set the queue for communicating with the GUI."""
        self.gui_queue = gui_queue

    def adsb_update_worker(self):
        """Background thread to update ADS-B data."""
        info("ADS-B updater thread started.", emoji="📡")

        # For non-authenticated OpenSky access, we're limited to 10 seconds between requests
        # But we can make it appear smoother by interpolating positions
        update_interval = 5 if getattr(self.adsb_tracker.data_source, "credentials", None) else 10

        while self.is_monitoring:
            try:
                aircraft_list = self.adsb_tracker.update_aircraft_positions()
                info(f"ADS-B Update: {len(aircraft_list)} aircraft in area")

                if self.gui_queue:
                    for aircraft in aircraft_list:
                        self.gui_queue.put(("update_aircraft", aircraft.to_dict()))

                time.sleep(update_interval)
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

                        # Send to GUI before saving/analyzing
                        if self.gui_queue:
                            self.gui_queue.put(("atc_transmission", {
                                'transcript': transcript_text,
                                'timestamp': datetime.now().isoformat(),
                                'audio_file': audio_file,
                                'transcription_number': self.stats['transmissions_transcribed']
                            }))

                        transcript_file = self.save_transcript(audio_file, result)
                        analysis_result = analyze_transcript(transcript_file)
                        analysis_data = (
                            analysis_result[0]
                            if isinstance(analysis_result, tuple)
                            else analysis_result
                        )
                        self.process_analysis(
                            analysis_data,
                            transcript_text,
                            audio_file,
                            result,
                        )
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

    def process_analysis(self, analysis, transcript_text, audio_file, transcription_result):
        """Process analysis results and generate alerts."""
        overall_info = {}
        if isinstance(analysis, dict):
            overall_info = analysis.get('overall_info', {})

        for callsign in overall_info.get('callsigns', []):
            self.stats['callsigns_detected'].add(callsign)

        if self.enable_adsb and getattr(self, 'correlator', None):
            correlation = self.correlator.correlate_transcript(
                transcript_text,
                datetime.now(),
            )
            for alert in correlation.get('alerts', []):
                self.stats['non_transponder_alerts'] += 1
                section(f"🚨 ALERT: {alert['type'].upper()} 🚨", emoji="🚨")
                info(f"Details: {alert}")

        if self.llm_correlator and transcription_result:
            self.run_llm_correlation(
                transcript_text,
                audio_file,
                transcription_result,
            )

    def run_llm_correlation(self, transcript_text, audio_file, transcription_result):
        """Generate a correlation request to the Ollama LLM."""
        if not self.llm_correlator:
            return

        segments = transcription_result.get('segments', []) if isinstance(transcription_result, dict) else []
        metadata = transcription_result.get('metadata', {}) if isinstance(transcription_result, dict) else {}
        metadata_timestamp = metadata.get('timestamp')

        try:
            recorded_timestamp = os.path.getmtime(audio_file)
        except OSError:
            recorded_timestamp = None

        frequency = str(self.params.get('frequency') or ATC_FREQUENCY)
        channel_name = self.params.get('channel_name') or "Primary Channel"

        transmission = build_atc_transmission(
            transcript_text,
            frequency,
            channel_name,
            segments,
            recorded_timestamp=recorded_timestamp,
            metadata_timestamp=metadata_timestamp,
        )

        with self.transmission_lock:
            self.transmissions_history.append(transmission)
            if len(self.transmissions_history) > self.max_llm_history:
                self.transmissions_history = self.transmissions_history[-self.max_llm_history :]
            recent_transmissions = self.transmissions_history[-self.llm_correlator.context_builder.max_tx :]

        adsb_contacts = []
        if getattr(self, 'adsb_tracker', None):
            adsb_contacts = build_adsb_contacts(self.adsb_tracker.current_aircraft.values())

        result = self.llm_correlator.correlate(adsb_contacts, recent_transmissions)
        self.last_llm_result = result

        if 'error' in result:
            warning(f"LLM correlation error: {result['error']}", emoji="⚠️")
            return

        summary = result.get('summary')
        if summary:
            info(f"LLM Summary: {summary}", emoji="🧠")

        for correlation in result.get('correlations', []):
            tx_id = correlation.get('transmission_id')
            if not isinstance(tx_id, int):
                continue
            if 0 <= tx_id < len(recent_transmissions):
                tx = recent_transmissions[tx_id]
                matched_icao = correlation.get('matched_icao', 'NO_MATCH')
                matched_callsign = correlation.get('matched_callsign', '') or ''
                try:
                    confidence = float(correlation.get('confidence', 0.0))
                except (TypeError, ValueError):
                    confidence = 0.0
                flags = correlation.get('flags', []) or []
                flag_text = f" flags={','.join(flags)}" if flags else ""
                info(
                    f"LLM Match [{tx_id}] {matched_icao} {matched_callsign} (conf {confidence:.2f}){flag_text}",
                    emoji="📎",
                )
                info(f"    Text: {tx.text[:120]}")

        for alert in result.get('alerts', []):
            alert_type = alert.get('type', 'UNKNOWN')
            details = alert.get('details', '')
            severity = alert.get('severity', 'LOW')
            warning(
                f"LLM Alert [{severity}] {alert_type}: {details}",
                emoji="🚨",
            )
            if alert_type == 'NON_TRANSPONDER':
                self.stats['non_transponder_alerts'] += 1
            if self.gui_queue:
                self.gui_queue.put((
                    "alert",
                    {
                        'type': f"LLM {alert_type} ({severity})",
                        'transcript': details or transcript_text,
                    },
                ))

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
