"""
multi_channel_monitor.py - Multi-channel ATC monitoring system
"""
import os
import time
import threading
import queue
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

from audio.recorders import EnhancedLiveATCRecorder
from transcription.transcriber import GPUWhisperTranscriber
from analysis.analyzer import analyze_transcript
from tracking.adsb_tracker import ADSBTracker, OpenSkySource
from analysis.correlator import ATCCorrelator
from analysis.ollama_correlator import (
    OllamaCorrelator as OllamaLLMCorrelator,
    build_adsb_contacts,
    build_atc_transmission,
)
from utils.console_logger import info, success, warning, error, section
from utils.config import (
    VAD_THRESHOLD,
    SILENCE_DURATION,
    AUDIO_DIR,
    OPENSKY_CREDENTIALS_FILE,
    ENABLE_ADSB,
    ADSB_SOURCE,
    MODEL_SIZE,
    NUM_TRANSCRIPTION_WORKERS,
    ENABLE_LLM_CORRELATION,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_REQUEST_TIMEOUT,
    LLM_MAX_ADSB_CONTACTS,
    LLM_MAX_TRANSMISSIONS,
)


class TranscriptionWorkerPool:
    """Pool of Whisper transcription workers for parallel processing"""

    def __init__(self, num_workers=3, model_size="large", parent_monitor=None):
        self.num_workers = num_workers
        self.model_size = model_size
        self.parent_monitor = parent_monitor  # Add this
        self.workers = []
        self.work_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.running = False

        info(f"Initializing transcription worker pool with {num_workers} workers")

    def start(self):
        """Start all worker threads"""
        self.running = True

        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True,
                name=f"TranscriptionWorker-{i}"
            )
            worker.start()
            self.workers.append(worker)

        success(f"Started {self.num_workers} transcription workers")

    def _worker_loop(self, worker_id):
        """Worker loop that processes transcription tasks"""
        # Each worker gets its own model instance
        transcriber = GPUWhisperTranscriber(
            model_size=self.model_size,
            device="auto",
            optimize_for_radio=True
        )
        transcriber.load_model()

        info(f"Worker {worker_id} ready with {self.model_size} model")

        # Send initial idle status
        if hasattr(self, 'parent_monitor') and self.parent_monitor.gui_queue:
            self.parent_monitor.gui_queue.put(("worker_status", {
                'worker_id': worker_id,
                'status': 'idle'
            }))

        while self.running:
            try:
                # Get work item (audio_file, channel_info, callback)
                work_item = self.work_queue.get(timeout=1)
                if work_item is None:
                    break

                audio_file, channel_info, callback = work_item

                # Send busy status
                if hasattr(self, 'parent_monitor') and self.parent_monitor.gui_queue:
                    self.parent_monitor.gui_queue.put(("worker_status", {
                        'worker_id': worker_id,
                        'status': 'busy',
                        'channel': channel_info['name']
                    }))

                # Process transcription
                start_time = time.time()
                result = transcriber.transcribe_audio_internal(audio_file)
                processing_time = time.time() - start_time

                if result and result.get('text', '').strip():
                    # Add channel info to result
                    result['channel_info'] = channel_info
                    result['processing_time'] = processing_time
                    result['worker_id'] = worker_id

                    # Call the callback with results
                    if callback:
                        callback(audio_file, result, channel_info)

                # Send idle status
                if hasattr(self, 'parent_monitor') and self.parent_monitor.gui_queue:
                    self.parent_monitor.gui_queue.put(("worker_status", {
                        'worker_id': worker_id,
                        'status': 'idle'
                    }))

                self.work_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                error(f"Worker {worker_id} error: {e}")
                # Send idle status on error
                if hasattr(self, 'parent_monitor') and self.parent_monitor.gui_queue:
                    self.parent_monitor.gui_queue.put(("worker_status", {
                        'worker_id': worker_id,
                        'status': 'idle'
                    }))

    def submit(self, audio_file, channel_info, callback):
        """Submit a transcription job to the pool"""
        self.work_queue.put((audio_file, channel_info, callback))

    def stop(self):
        """Stop all workers"""
        self.running = False

        # Send stop signals
        for _ in range(self.num_workers):
            self.work_queue.put(None)

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)

        info("Transcription worker pool stopped")


class MultiChannelATCMonitor:
    """Main multi-channel monitoring system"""

    def __init__(self, channel_configs, num_transcription_workers=None):
        """
        Initialize multi-channel monitor

        channel_configs: List of channel configurations, each containing:
            - name: Channel name (e.g., "PDX Tower")
            - frequency: Frequency (e.g., "118.7")
            - stream_url: LiveATC stream URL
            - color: Display color for GUI (optional)
        
        num_transcription_workers: Number of parallel transcription workers
            If None, uses NUM_TRANSCRIPTION_WORKERS from config.py
            Can be overridden via command line --workers argument
        """
        self.channel_configs = channel_configs
        self.channels = {}
        self.is_monitoring = False
        self.gui_queue = None

        self.transmission_lock = threading.Lock()
        self.channel_transmissions = {}
        self.last_llm_results = {}
        self.llm_correlator = None
        self.max_llm_history = max(LLM_MAX_TRANSMISSIONS * 3, LLM_MAX_TRANSMISSIONS or 1)

        # Use config value if not specified
        if num_transcription_workers is None:
            num_transcription_workers = NUM_TRANSCRIPTION_WORKERS

        # Create transcription worker pool
        self.transcription_pool = TranscriptionWorkerPool(
            num_workers=num_transcription_workers,
            model_size=MODEL_SIZE,
            parent_monitor=self  # Pass self as parent
        )

        # Statistics
        self.stats = {
            'start_time': None,
            'channels': {}
        }

        # Initialize channels
        for config in channel_configs:
            channel_name = config['name']
            self.channels[channel_name] = {
                'config': config,
                'recorder': None,
                'thread': None,
                'audio_dir': self._create_channel_audio_dir(config),
                'transcript_dir': self._create_channel_transcript_dir(config)
            }

            # Initialize channel stats
            self.stats['channels'][channel_name] = {
                'transmissions_recorded': 0,
                'transmissions_transcribed': 0,
                'callsigns_detected': set(),
                'last_transmission': None,
                'non_transponder_alerts': 0,
            }
            self.channel_transmissions[channel_name] = []

        # Initialize ADS-B tracking if enabled
        self.enable_adsb = ENABLE_ADSB
        if self.enable_adsb:
            if ADSB_SOURCE == 'opensky':
                source = OpenSkySource(OPENSKY_CREDENTIALS_FILE)
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

    def _create_channel_audio_dir(self, config):
        """Create directory for channel audio files"""
        freq_str = config['frequency'].replace('.', 'p')
        channel_dir = os.path.join(AUDIO_DIR, f"{freq_str}_{config['name'].replace(' ', '_')}")
        os.makedirs(channel_dir, exist_ok=True)
        return channel_dir

    def _create_channel_transcript_dir(self, config):
        """Create directory for channel transcripts"""
        freq_str = config['frequency'].replace('.', 'p')
        channel_dir = os.path.join("transcripts", f"{freq_str}_{config['name'].replace(' ', '_')}")
        os.makedirs(channel_dir, exist_ok=True)
        return channel_dir

    def set_gui_queue(self, gui_queue):
        """Set the queue for GUI communication"""
        self.gui_queue = gui_queue

    def start_monitoring(self):
        """Start monitoring all channels"""
        self.is_monitoring = True
        self.stats['start_time'] = datetime.now()

        section("Starting Multi-Channel ATC Monitor", emoji="🎙️")
        info(f"Monitoring {len(self.channels)} channels")

        # Start transcription pool
        self.transcription_pool.start()

        # Start stats updater
        stats_thread = threading.Thread(target=self.stats_update_worker, daemon=True)
        stats_thread.start()

        # Start ADS-B updater if enabled
        if self.enable_adsb:
            adsb_thread = threading.Thread(target=self.adsb_update_worker, daemon=True)
            adsb_thread.start()

        # Start recorder for each channel
        for channel_name, channel_data in self.channels.items():
            config = channel_data['config']
            info(f"Starting {channel_name} on {config['frequency']} MHz")

            # Create recorder with custom callback
            recorder = EnhancedLiveATCRecorder(
                stream_url=config['stream_url'],
                vad_threshold=VAD_THRESHOLD,
                silence_duration=SILENCE_DURATION,
                callback=lambda f, ch=channel_name: self.recording_callback(f, ch)
            )

            # Override save location
            recorder.audio_dir = channel_data['audio_dir']

            channel_data['recorder'] = recorder

            # Start recording thread
            thread = threading.Thread(
                target=self._channel_recording_loop,
                args=(channel_name,),
                daemon=True,
                name=f"Recorder-{channel_name}"
            )
            thread.start()
            channel_data['thread'] = thread

        success(f"All {len(self.channels)} channels started")

        # Update GUI with channel info
        if self.gui_queue:
            self.gui_queue.put(("channels_initialized", {
                'channels': [
                    {
                        'name': ch['config']['name'],
                        'frequency': ch['config']['frequency'],
                        'color': ch['config'].get('color', '#00FF00')
                    }
                    for ch in self.channels.values()
                ]
            }))

    def _channel_recording_loop(self, channel_name):
        """Recording loop for a single channel"""
        channel_data = self.channels[channel_name]
        recorder = channel_data['recorder']
        config = channel_data['config']

        try:
            # This will block and record continuously
            recorder.record_with_vad(frequency=config['frequency'])
        except Exception as e:
            error(f"Error in {channel_name} recording: {e}")
        finally:
            info(f"{channel_name} recording stopped")

    def recording_callback(self, audio_file, channel_name):
        """Callback when audio is recorded on a channel"""
        if not self.is_monitoring:
            return

        # Update stats
        self.stats['channels'][channel_name]['transmissions_recorded'] += 1
        self.stats['channels'][channel_name]['last_transmission'] = datetime.now()

        # Get channel info
        channel_info = {
            'name': channel_name,
            'frequency': self.channels[channel_name]['config']['frequency'],
            'color': self.channels[channel_name]['config'].get('color', '#00FF00'),
            'audio_file': audio_file,
            'timestamp': datetime.now().isoformat()
        }

        # Notify GUI of recording
        if self.gui_queue:
            self.gui_queue.put(("channel_recording", {
                'channel': channel_name,
                'frequency': channel_info['frequency'],
                'recording_count': self.stats['channels'][channel_name]['transmissions_recorded']
            }))

        # Submit to transcription pool
        self.transcription_pool.submit(
            audio_file,
            channel_info,
            self.transcription_callback
        )

    def transcription_callback(self, audio_file, result, channel_info):
        """Callback when transcription is complete"""
        channel_name = channel_info['name']
        transcript_text = result['text'].strip()

        # Update stats
        self.stats['channels'][channel_name]['transmissions_transcribed'] += 1

        info(f"[{channel_name}] Transcript: \"{transcript_text}\"", emoji="📢")

        # Save transcript to channel-specific directory
        self.save_channel_transcript(audio_file, result, channel_info)

        # Send to GUI
        if self.gui_queue:
            self.gui_queue.put(("atc_transmission", {
                'transcript': transcript_text,
                'channel': channel_name,
                'frequency': channel_info['frequency'],
                'color': channel_info['color'],
                'timestamp': result['metadata']['timestamp'],
                'worker_id': result['worker_id'],
                'processing_time': result['processing_time'],
                'transcription_number': self.stats['channels'][channel_name]['transmissions_transcribed']
            }))

        if self.llm_correlator:
            self.run_llm_correlation(channel_info, transcript_text, audio_file, result)

        # Analyze if needed
        # Note: You may want to make analysis optional or async for performance
        # analysis = analyze_transcript(transcript_file)
        # self.process_analysis(analysis, transcript_text, channel_name)

    def save_channel_transcript(self, audio_file, result, channel_info):
        """Save transcript to channel-specific directory"""
        channel_name = channel_info['name']
        transcript_dir = self.channels[channel_name]['transcript_dir']

        base_name = os.path.basename(audio_file).replace('.wav', '')
        transcript_file = os.path.join(transcript_dir, f"{base_name}_transcript.json")

        # Add channel info to result
        result['channel_info'] = channel_info

        import json
        with open(transcript_file, 'w') as f:
            json.dump(result, f, indent=2)

        return transcript_file

    def run_llm_correlation(self, channel_info, transcript_text, audio_file, transcription_result):
        """Run the Ollama correlation workflow for a channel."""
        if not self.llm_correlator:
            return

        channel_name = channel_info.get('name', 'Unknown')
        frequency = str(channel_info.get('frequency', ''))
        segments = transcription_result.get('segments', []) if isinstance(transcription_result, dict) else []
        metadata = transcription_result.get('metadata', {}) if isinstance(transcription_result, dict) else {}
        metadata_timestamp = metadata.get('timestamp')

        try:
            recorded_timestamp = os.path.getmtime(audio_file)
        except OSError:
            recorded_timestamp = None

        transmission = build_atc_transmission(
            transcript_text,
            frequency,
            channel_name,
            segments,
            recorded_timestamp=recorded_timestamp,
            metadata_timestamp=metadata_timestamp,
        )

        with self.transmission_lock:
            history = self.channel_transmissions.setdefault(channel_name, [])
            history.append(transmission)
            if len(history) > self.max_llm_history:
                history = history[-self.max_llm_history:]
                self.channel_transmissions[channel_name] = history
            recent_transmissions = history[-self.llm_correlator.context_builder.max_tx :]

        adsb_contacts = []
        if getattr(self, 'adsb_tracker', None):
            adsb_contacts = build_adsb_contacts(self.adsb_tracker.current_aircraft.values())

        result = self.llm_correlator.correlate(adsb_contacts, recent_transmissions)
        self.last_llm_results[channel_name] = result

        if 'error' in result:
            warning(f"[{channel_name}] LLM correlation error: {result['error']}", emoji="⚠️")
            return

        summary = result.get('summary')
        if summary:
            info(f"[{channel_name}] LLM Summary: {summary}", emoji="🧠")

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
                    f"[{channel_name}] LLM Match [{tx_id}] {matched_icao} {matched_callsign} (conf {confidence:.2f}){flag_text}",
                    emoji="📎",
                )
                info(f"    Text: {tx.text[:120]}")

        for alert in result.get('alerts', []):
            alert_type = alert.get('type', 'UNKNOWN')
            details = alert.get('details', '')
            severity = alert.get('severity', 'LOW')
            warning(
                f"[{channel_name}] LLM Alert [{severity}] {alert_type}: {details}",
                emoji="🚨",
            )
            if alert_type == 'NON_TRANSPONDER':
                self.stats['channels'][channel_name]['non_transponder_alerts'] += 1
            if self.gui_queue:
                self.gui_queue.put((
                    "alert",
                    {
                        'type': f"LLM {alert_type} ({severity})",
                        'transcript': f"[{channel_name}] {details or transcript_text}",
                    },
                ))

    def adsb_update_worker(self):
        """Background thread to update ADS-B data"""
        info("ADS-B updater thread started", emoji="📡")
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

    def stats_update_worker(self):
        """Periodically update GUI with queue statistics"""
        while self.is_monitoring:
            try:
                # Count busy workers
                busy_count = 0
                # This is a simple approximation - you might want to track this more precisely
                queue_size = self.transcription_pool.work_queue.qsize()

                if self.gui_queue:
                    self.gui_queue.put(("stats_update", {
                        'queue_size': queue_size,
                        'workers_busy': min(queue_size, self.transcription_pool.num_workers)  # Approximation
                    }))

                time.sleep(1)  # Update every second
            except Exception as e:
                error(f"Stats update error: {e}")

    def stop_monitoring(self):
        """Stop all monitoring activities"""
        if self.is_monitoring:
            self.is_monitoring = False
            info("Stopping multi-channel monitoring...", emoji="🛑")

            # Stop transcription pool
            self.transcription_pool.stop()

            # Stop all recorders
            for channel_name, channel_data in self.channels.items():
                if channel_data['recorder'] and channel_data['recorder'].ffmpeg_process:
                    channel_data['recorder'].ffmpeg_process.terminate()

            self.print_statistics()

    def print_statistics(self):
        """Print monitoring session statistics"""
        if self.stats['start_time']:
            duration = datetime.now() - self.stats['start_time']
            section("MULTI-CHANNEL MONITORING STATISTICS", emoji="📊")
            info(f"Duration: {str(duration).split('.')[0]}")
            info(f"Channels monitored: {len(self.channels)}")

            total_recorded = 0
            total_transcribed = 0

            for channel_name, channel_stats in self.stats['channels'].items():
                recorded = channel_stats['transmissions_recorded']
                transcribed = channel_stats['transmissions_transcribed']
                total_recorded += recorded
                total_transcribed += transcribed

                info(f"\n{channel_name}:")
                info(f"  Recorded: {recorded}")
                info(f"  Transcribed: {transcribed}")
                info(f"  Unique callsigns: {len(channel_stats['callsigns_detected'])}")

            info(f"\nTotal transmissions recorded: {total_recorded}")
            info(f"Total transmissions transcribed: {total_transcribed}")