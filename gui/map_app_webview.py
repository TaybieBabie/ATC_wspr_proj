import webview
import threading
import queue
import json
import time
from collections import deque
from datetime import datetime
from utils import config
from utils.console_logger import info, success, error, warning


class OpenSkyMapApp:
    """Opensky map injector gui thing to avoid reinventing the wheel to use raw adsb data"""

    def __init__(self, atc_monitor):
        self.atc_monitor = atc_monitor
        self.update_queue = queue.Queue()
        self.atc_monitor.set_gui_queue(self.update_queue)

        self.window = None
        self.running = True
        self.transmission_count = 0
        self.overlay_initialized = False
        self.page_loaded = False
        self.inject_attempts = 0
        self.max_inject_attempts = 60
        self.inject_lock = threading.Lock()
        self.injection_watchdog_interval = 5.0
        self.injection_watchdog_started = False

        # For transcript display
        self.displayed_transcripts = []
        self.max_displayed_transcripts = 10
        self.pending_transmissions = deque()
        self.ui_flush_interval = 0.5
        self.max_pending_transmissions = 100
        self.max_batch_size = 20
        self.last_ui_flush = 0.0

        # Channel-specific counters for multi-channel mode
        self.channel_counters = {}
        for channel_config in atc_monitor.channel_configs:
            self.channel_counters[channel_config['frequency']] = 0
        self.num_workers = getattr(
            atc_monitor,
            'transcription_pool',
            {}
        ).num_workers if hasattr(atc_monitor, 'transcription_pool') else 3

    def run(self):
        """Run the application"""
        window_title = 'Multi-Channel ATC Monitor'
        self.window = webview.create_window(
            window_title,
            f'https://map.opensky-network.org/?lat={config.AIRPORT_LAT}&lon={config.AIRPORT_LON}&zoom=10',
            width=1600,
            height=900
        )

        self.window.events.loaded += self.on_page_loaded
        self.window.events.closed += self.on_closed

        update_thread = threading.Thread(target=self.process_updates, daemon=True)
        update_thread.start()

        webview.start(debug=True)

    def on_page_loaded(self):
        """Called when page is fully loaded"""
        info("Page loaded event received")
        self.page_loaded = True
        self.overlay_initialized = False
        self.inject_attempts = 0

        self._schedule_injection_retry(1.0, "page loaded")
        self._start_injection_watchdog()

    def on_closed(self):
        """Called when the webview window is closed"""
        info("Webview window closed, shutting down monitor")
        self.running = False
        if self.atc_monitor:
            self.atc_monitor.stop_monitoring()

    def inject_monitor(self):
        """Inject the monitoring code once"""
        if not self.running or self.overlay_initialized:
            return

        if not self.page_loaded:
            self._schedule_injection_retry(1.0, "page not loaded yet")
            return

        with self.inject_lock:
            self.inject_attempts += 1
            if self.inject_attempts > self.max_inject_attempts:
                error("Max injection attempts reached; giving up")
                return

            info(f"Injecting ATC monitor (attempt {self.inject_attempts})...")
            self._inject_multi_channel_monitor()

    def _schedule_injection_retry(self, delay, reason=None):
        """Retry injection after a delay."""
        if not self.running or self.overlay_initialized:
            return
        if reason:
            warning(f"Injection retry scheduled in {delay:.1f}s ({reason})")
        threading.Timer(delay, self.inject_monitor).start()

    def _start_injection_watchdog(self):
        """Ensure the injected UI returns after page reloads."""
        if self.injection_watchdog_started:
            return
        self.injection_watchdog_started = True
        threading.Timer(self.injection_watchdog_interval, self._injection_watchdog).start()

    def _injection_watchdog(self):
        if not self.running:
            return

        if self.page_loaded and self.window:
            try:
                status_js = """
                (function() {
                    if (!document || !document.body) {
                        return { ready: false };
                    }
                    const panel = document.getElementById('multi-channel-panel');
                    return {
                        ready: document.readyState === 'complete',
                        hasPanel: !!panel
                    };
                })();
                """
                status = self.window.evaluate_js(status_js)
                if status and status.get("ready") and not status.get("hasPanel"):
                    warning("Injected UI missing; re-injecting")
                    self.overlay_initialized = False
                    self.inject_attempts = 0
                    self._schedule_injection_retry(0.5, "panel missing after reload")
            except Exception as exc:
                warning(f"Injection watchdog error: {exc}")

        threading.Timer(self.injection_watchdog_interval, self._injection_watchdog).start()

    def _inject_multi_channel_monitor(self):
        """Inject multi-channel monitoring interface"""
        # Generate channel list HTML
        channels_html = ""
        for channel_config in self.atc_monitor.channel_configs:
            freq = channel_config['frequency']
            freq_id = freq.replace('.', '_')
            color = channel_config.get('color', '#FFFFFF')
            stream_url = channel_config.get('stream_url', '')
            channels_html += f"""
            <div class="channel-item" data-freq-id="{freq_id}" style="margin-bottom: 6px; padding: 6px 8px; background: rgba(255,255,255,0.03); clip-path: polygon(0 0, calc(100% - 8px) 0, 100% 8px, 100% 100%, 0 100%); border-left: 2px solid {color};">
                <div style="font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;">{channel_config['name']}</div>
                <div style="font-size: 10px; color: #888; margin-top: 3px; font-family: 'Courier New', monospace;">
                    {freq} MHz | TX: <span id="channel-count-{freq_id}" style="color: {color}; font-weight: bold;">0</span>
                    <button id="mute-{freq_id}" class="mute-btn" data-freq-id="{freq_id}" style="margin-left: 8px; padding: 2px 6px; font-size: 9px; background: rgba(255,255,255,0.1); border: 1px solid {color}; color: {color}; cursor: pointer; clip-path: polygon(0 0, calc(100% - 4px) 0, 100% 4px, 100% 100%, 0 100%);">UNMUTE</button>
                    <audio id="audio-{freq_id}" data-stream="{stream_url}" preload="none" style="display:none;"></audio>
                </div>
            </div>
            """

        injection_js = f"""
        (function() {{
            try {{
                if (!document || !document.body || !document.head || document.readyState !== 'complete') {{
                    return false;
                }}
                if (window.multiChannelMonitorInjected === 'complete') {{
                    return true;
                }}

                // Remove any existing panels to prevent duplicates
                const existingPanel = document.getElementById('multi-channel-panel');
                const existingTranscript = document.getElementById('multi-transcript-container');
                if (existingPanel) existingPanel.remove();
                if (existingTranscript) existingTranscript.remove();

                window.multiChannelMonitorInjected = 'in_progress';
                console.log('[ATC] Injecting multi-channel monitor...');

                // Configuration
                const AIRPORT_LAT = {config.AIRPORT_LAT};
                const AIRPORT_LON = {config.AIRPORT_LON};
                const SEARCH_RADIUS_NM = {config.SEARCH_RADIUS_NM};
                const RADIUS_METERS = SEARCH_RADIUS_NM * 1852;

                // Draggable/Resizable utility
                window.makeDraggable = function(element, handle) {{
                    let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
                    handle.style.cursor = 'move';

                    handle.onmousedown = dragMouseDown;

                    function dragMouseDown(e) {{
                        e = e || window.event;
                        e.preventDefault();
                        pos3 = e.clientX;
                        pos4 = e.clientY;
                        document.onmouseup = closeDragElement;
                        document.onmousemove = elementDrag;
                    }}

                    function elementDrag(e) {{
                        e = e || window.event;
                        e.preventDefault();
                        pos1 = pos3 - e.clientX;
                        pos2 = pos4 - e.clientY;
                        pos3 = e.clientX;
                        pos4 = e.clientY;
                        element.style.top = (element.offsetTop - pos2) + "px";
                        element.style.left = (element.offsetLeft - pos1) + "px";
                        element.style.right = 'auto';
                        element.style.bottom = 'auto';
                    }}

                    function closeDragElement() {{
                        document.onmouseup = null;
                        document.onmousemove = null;
                    }}
                }};

                window.makeResizable = function(element) {{
                    const resizer = document.createElement('div');
                    resizer.style.cssText = `
                        position: absolute;
                        right: 0;
                        bottom: 0;
                        width: 20px;
                        height: 20px;
                        cursor: nwse-resize;
                        background: linear-gradient(135deg, transparent 0%, transparent 50%, rgba(255,255,255,0.3) 50%, rgba(255,255,255,0.3) 100%);
                        clip-path: polygon(100% 0, 100% 100%, 0 100%);
                    `;
                    element.appendChild(resizer);

                    let startX, startY, startWidth, startHeight;

                    resizer.addEventListener('mousedown', initResize, false);

                    function initResize(e) {{
                        startX = e.clientX;
                        startY = e.clientY;
                        startWidth = parseInt(document.defaultView.getComputedStyle(element).width, 10);
                        startHeight = parseInt(document.defaultView.getComputedStyle(element).height, 10);
                        document.addEventListener('mousemove', resize, false);
                        document.addEventListener('mouseup', stopResize, false);
                    }}

                    function resize(e) {{
                        element.style.width = (startWidth + e.clientX - startX) + 'px';
                        element.style.height = (startHeight + e.clientY - startY) + 'px';
                    }}

                    function stopResize() {{
                        document.removeEventListener('mousemove', resize, false);
                        document.removeEventListener('mouseup', stopResize, false);
                    }}
                }};

                // Create main control panel
                const panel = document.createElement('div');
                panel.id = 'multi-channel-panel';
                panel.style.cssText = `
                    position: fixed;
                    top: 80px;
                    right: 20px;
                    background: linear-gradient(135deg, rgba(10, 10, 15, 0.95) 0%, rgba(20, 20, 30, 0.95) 100%);
                    color: #00ff00;
                    padding: 0;
                    font-family: 'Courier New', monospace;
                    z-index: 10000;
                    box-shadow: 0 0 0 2px #00ff00, 0 0 20px rgba(0,255,0,0.3), inset 0 0 20px rgba(0,255,0,0.05);
                    width: 340px;
                    max-height: 700px;
                    overflow: hidden;
                    clip-path: polygon(0 0, calc(100% - 20px) 0, 100% 20px, 100% 100%, 0 100%);
                `;

                panel.innerHTML = `
                    <div id="panel-header" style="background: rgba(0,255,0,0.1); padding: 12px 15px; border-bottom: 1px solid #00ff00; cursor: move; user-select: none; clip-path: polygon(0 0, calc(100% - 15px) 0, 100% 15px, 100% 100%, 0 100%);">
                        <div style="font-size: 14px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase;">
                            ◢ ATC MONITOR ◣
                        </div>
                        <div style="font-size: 9px; color: #00aa00; margin-top: 3px; letter-spacing: 1px;">
                            MULTI-CHANNEL SURVEILLANCE
                        </div>
                    </div>

                    <div style="padding: 15px; max-height: 580px; overflow-y: auto;" id="panel-content">
                        <div style="margin-bottom: 15px; padding: 10px; background: rgba(0,255,0,0.05); border: 1px solid rgba(0,255,0,0.3); clip-path: polygon(0 0, calc(100% - 8px) 0, 100% 8px, 100% 100%, 0 100%);">
                            <div style="font-size: 10px; margin-bottom: 5px; color: #00ff00;">
                                <span style="opacity: 0.7;">STATUS:</span> 
                                <span id="monitor-status" style="color: #00ff00;">◉ ACTIVE</span>
                            </div>
                            <div style="font-size: 10px; margin-bottom: 5px; color: #00ff00;">
                                <span style="opacity: 0.7;">AREA:</span> {config.LOCATION_NAME} ({config.SEARCH_RADIUS_NM}NM)
                            </div>
                            <div style="font-size: 10px; margin-bottom: 5px; color: #00ff00;">
                                <span style="opacity: 0.7;">TOTAL TX:</span> 
                                <span id="total-transmissions" style="font-weight: bold;">0</span>
                            </div>
                            <div style="font-size: 10px; color: #00ff00;">
                                <span style="opacity: 0.7;">QUEUE:</span> 
                                <span id="queue-size">0</span> | 
                                <span style="opacity: 0.7;">WORKERS:</span> 
                                <span id="workers-busy">0</span>/{self.num_workers}
                            </div>
                        </div>

                        <div style="margin-bottom: 4px; font-size: 11px; font-weight: 700; letter-spacing: 1px; color: #00ff00; opacity: 0.8;">
                            ▸ CHANNELS [{len(self.atc_monitor.channel_configs)}]
                        </div>
                        <div id="channel-list" style="margin-bottom: 15px;">
                            {channels_html}
                        </div>

                        <div style="margin-bottom: 4px; font-size: 11px; font-weight: 700; letter-spacing: 1px; color: #00ff00; opacity: 0.8;">
                            ▸ WORKERS
                        </div>
                        <div id="worker-status" style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px;">
                            {"".join(f'''<div id="worker-{i}" class="worker-box" style="padding: 8px 4px; background: rgba(0,255,0,0.1); text-align: center; font-size: 9px; clip-path: polygon(0 0, calc(100% - 6px) 0, 100% 6px, 100% 100%, 0 100%); border: 1px solid rgba(0,255,0,0.3);">
                                <div style="font-weight: bold;">W{i}</div>
                                <div id="worker-{i}-status" style="color: #00aa00; margin-top: 2px;">IDLE</div>
                            </div>''' for i in range(self.num_workers))}
                        </div>
                    </div>
                `;

                document.body.appendChild(panel);

                // Make panel draggable and resizable
                const panelHeader = document.getElementById('panel-header');
                window.makeDraggable(panel, panelHeader);
                window.makeResizable(panel);

                // Toggle audio playback
                window.toggleMute = function(freqId) {{
                    const audioEl = document.getElementById('audio-' + freqId);
                    const btnEl = document.getElementById('mute-' + freqId);
                    if (!audioEl || !btnEl) return;

                    if (audioEl.paused) {{
                        const streamUrl = audioEl.getAttribute('data-stream');
                        if (audioEl.src !== streamUrl) {{
                            audioEl.src = streamUrl;
                            audioEl.load();
                        }}
                        audioEl.play();
                        btnEl.textContent = 'MUTE';
                        btnEl.style.background = 'rgba(255,0,0,0.2)';
                    }} else {{
                        audioEl.pause();
                        audioEl.removeAttribute('src');
                        audioEl.load();
                        btnEl.textContent = 'UNMUTE';
                        btnEl.style.background = 'rgba(255,255,255,0.1)';
                    }}
                }};

                // Attach mute button handlers
                document.querySelectorAll('.mute-btn').forEach(btn => {{
                    btn.addEventListener('click', function() {{
                        window.toggleMute(this.getAttribute('data-freq-id'));
                    }});
                }});

                // Create transcript display
                const transcriptContainer = document.createElement('div');
                transcriptContainer.id = 'multi-transcript-container';
                transcriptContainer.style.cssText = `
                    position: fixed;
                    bottom: 20px;
                    left: 20px;
                    width: 900px;
                    max-width: calc(100vw - 400px);
                    background: linear-gradient(135deg, rgba(10, 10, 15, 0.95) 0%, rgba(20, 20, 30, 0.95) 100%);
                    color: #00ff00;
                    padding: 0;
                    font-family: 'Courier New', monospace;
                    z-index: 9999;
                    box-shadow: 0 0 0 2px #00ff00, 0 0 20px rgba(0,255,0,0.3), inset 0 0 20px rgba(0,255,0,0.05);
                    max-height: 280px;
                    overflow: hidden;
                    clip-path: polygon(0 0, calc(100% - 20px) 0, 100% 20px, 100% 100%, 0 100%);
                `;

                transcriptContainer.innerHTML = `
                    <div id="transcript-header" style="background: rgba(0,255,0,0.1); padding: 10px 15px; border-bottom: 1px solid #00ff00; cursor: move; user-select: none; clip-path: polygon(0 0, calc(100% - 15px) 0, 100% 15px, 100% 100%, 0 100%);">
                        <div style="font-size: 12px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase;">
                            ◢ LIVE TRANSMISSIONS ◣
                        </div>
                    </div>
                    <div id="transcript-content" style="padding: 12px 15px; max-height: 200px; overflow-y: auto;">
                        <div style="text-align: center; opacity: 0.5; font-size: 11px;">AWAITING TRANSMISSION DATA...</div>
                    </div>
                `;

                document.body.appendChild(transcriptContainer);

                // Make transcript draggable and resizable
                const transcriptHeader = document.getElementById('transcript-header');
                window.makeDraggable(transcriptContainer, transcriptHeader);
                window.makeResizable(transcriptContainer);

                // Add custom styles
                const style = document.createElement('style');
                style.textContent = `
                    #panel-content::-webkit-scrollbar,
                    #transcript-content::-webkit-scrollbar {{
                        width: 8px;
                    }}

                    #panel-content::-webkit-scrollbar-track,
                    #transcript-content::-webkit-scrollbar-track {{
                        background: rgba(0,255,0,0.05);
                    }}

                    #panel-content::-webkit-scrollbar-thumb,
                    #transcript-content::-webkit-scrollbar-thumb {{
                        background: rgba(0,255,0,0.3);
                        border: 1px solid rgba(0,255,0,0.5);
                    }}

                    #panel-content::-webkit-scrollbar-thumb:hover,
                    #transcript-content::-webkit-scrollbar-thumb:hover {{
                        background: rgba(0,255,0,0.5);
                    }}

                    .worker-box {{
                        transition: all 0.2s ease;
                    }}

                    .channel-item {{
                        transition: background 0.3s ease;
                    }}

                    @keyframes pulse {{
                        0%, 100% {{ opacity: 1; }}
                        50% {{ opacity: 0.6; }}
                    }}

                    @keyframes slideIn {{
                        from {{
                            opacity: 0;
                            transform: translateX(-20px);
                        }}
                        to {{
                            opacity: 1;
                            transform: translateX(0);
                        }}
                    }}

                    .transcript-item {{
                        animation: slideIn 0.3s ease-out;
                    }}
                `;
                document.head.appendChild(style);

                // Initialize monitoring circle overlay
                let overlayInitialized = false;
                let initAttempts = 0;

                function tryInitOverlay() {{
                    initAttempts++;
                    console.log('[ATC] Overlay initialization attempt #' + initAttempts);

                    let map = null;
                    if (typeof OLMap !== 'undefined') {{
                        map = OLMap;
                    }} else if (window.OLMap) {{
                        map = window.OLMap;
                    }} else {{
                        const mapCanvas = document.querySelector('#map_canvas');
                        if (mapCanvas && mapCanvas._olMap) {{
                            map = mapCanvas._olMap;
                        }}
                    }}

                    if (!map && initAttempts < 30) {{
                        setTimeout(tryInitOverlay, 1000);
                        return;
                    }}

                    if (map) {{
                        console.log('[ATC] Map found! Creating monitoring radius overlay...');

                        try {{
                            const mapContainer = document.querySelector('#map_container');
                            if (!mapContainer) {{
                                console.error('[ATC] Map container not found');
                                return;
                            }}

                            // Remove existing overlay if present
                            const existingOverlay = document.getElementById('atc-overlay-canvas');
                            if (existingOverlay) existingOverlay.remove();

                            const overlayCanvas = document.createElement('canvas');
                            overlayCanvas.id = 'atc-overlay-canvas';
                            overlayCanvas.style.cssText = `
                                position: absolute;
                                top: 0;
                                left: 0;
                                width: 100%;
                                height: 100%;
                                pointer-events: none;
                                z-index: 500;
                            `;

                            const mapCanvas = mapContainer.querySelector('#map_canvas');
                            if (mapCanvas) {{
                                mapCanvas.appendChild(overlayCanvas);
                            }}

                            function drawMonitoringCircle() {{
                                const canvas = document.getElementById('atc-overlay-canvas');
                                if (!canvas || !map) return;

                                canvas.width = canvas.offsetWidth;
                                canvas.height = canvas.offsetHeight;

                                const ctx = canvas.getContext('2d');
                                ctx.clearRect(0, 0, canvas.width, canvas.height);

                                const centerCoords = ol.proj.fromLonLat([AIRPORT_LON, AIRPORT_LAT]);
                                const centerPixel = map.getPixelFromCoordinate(centerCoords);

                                if (!centerPixel) return;

                                const resolution = map.getView().getResolution();
                                const radiusPixels = RADIUS_METERS / resolution;

                                // Draw the circle
                                ctx.strokeStyle = 'rgba(0, 255, 0, 0.8)';
                                ctx.lineWidth = 3;
                                ctx.setLineDash([10, 10]);

                                ctx.beginPath();
                                ctx.arc(centerPixel[0], centerPixel[1], radiusPixels, 0, 2 * Math.PI);
                                ctx.stroke();

                                ctx.fillStyle = 'rgba(0, 255, 0, 0.05)';
                                ctx.fill();
                            }}

                            drawMonitoringCircle();

                            map.on('moveend', drawMonitoringCircle);
                            map.on('postrender', drawMonitoringCircle);
                            map.getView().on('change:resolution', drawMonitoringCircle);
                            map.getView().on('change:center', drawMonitoringCircle);

                            overlayInitialized = true;
                            console.log('[ATC] Monitoring radius overlay initialized!');

                            window.atcOverlay = {{
                                map: map,
                                canvas: overlayCanvas,
                                draw: drawMonitoringCircle,
                                initialized: true
                            }};

                        }} catch (error) {{
                            console.error('[ATC] Error creating overlay:', error);
                            if (initAttempts < 30) {{
                                setTimeout(tryInitOverlay, 1000);
                            }}
                        }}
                    }}
                }}

                window.atcGetStatus = function() {{
                    return {{
                        initialized: overlayInitialized,
                        attempts: initAttempts
                    }};
                }};

                tryInitOverlay();

                window.multiChannelMonitorInjected = 'complete';
                return true;
            }} catch (error) {{
                console.error('[ATC] Injection error:', error);
                window.multiChannelMonitorInjected = false;
                return false;
            }}
        }})();
        """

        try:
            result = self.window.evaluate_js(injection_js)
            if result:
                self.overlay_initialized = True
                success("Multi-channel monitor interface injected")
                self.check_initialization_status()
            else:
                warning("Multi-channel monitor injection returned false; retrying soon")
                self._schedule_injection_retry(2.0, "injection returned false")
        except Exception as e:
            error(f"Error injecting monitor: {e}")
            self._schedule_injection_retry(2.0, "injection exception")

    def process_updates(self):
        """Process updates from the monitor"""
        while self.running:
            try:
                message = self.update_queue.get(timeout=0.1)
                command = message[0]
                data = message[1]

                if command == "atc_transmission":
                    self._enqueue_transmission(data)
                elif command == "update_aircraft":
                    self.update_aircraft(data)
                elif command == "recording_started":
                    self.show_recording_status(data)
                elif command == "alert":
                    self.show_alert(data)
                elif command == "channel_recording":
                    self.flash_channel(data['frequency'])
                elif command == "worker_status":
                    self.update_worker_status(data)
                elif command == "stats_update":
                    self.update_statistics(data)

            except queue.Empty:
                pass
            except Exception as e:
                error(f"Error processing update: {e}")

            self._flush_pending_transmissions()

    def _enqueue_transmission(self, data):
        """Queue transmissions for batched UI updates."""
        self.pending_transmissions.append(data)
        while len(self.pending_transmissions) > self.max_pending_transmissions:
            self.pending_transmissions.popleft()

    def _flush_pending_transmissions(self):
        """Apply queued transmissions to the UI on a cadence."""
        if not self.pending_transmissions or not self.window or not self.overlay_initialized:
            return

        now = time.monotonic()
        if now - self.last_ui_flush < self.ui_flush_interval:
            return

        batch = []
        while self.pending_transmissions and len(batch) < self.max_batch_size:
            batch.append(self.pending_transmissions.popleft())

        if batch:
            self._apply_transmission_batch(batch)
            self.last_ui_flush = now

    def add_multi_channel_transmission(self, data):
        """Add transmission for multi-channel mode"""
        self._apply_transmission_batch([data])

    def _apply_transmission_batch(self, batch):
        """Apply a batch of transmissions to the UI - FIXED to prevent layering"""
        if not self.window or not self.overlay_initialized:
            return

        updated_freqs = set()
        for data in batch:
            freq = data.get('frequency', '')
            if freq in self.channel_counters:
                self.channel_counters[freq] += 1
                updated_freqs.add(freq)

            self.displayed_transcripts.append(data)

        while len(self.displayed_transcripts) > self.max_displayed_transcripts:
            self.displayed_transcripts.pop(0)

        # Build transcript HTML
        transcript_html = ""
        for trans in self.displayed_transcripts:
            color = trans.get('color', '#00ff00')
            timestamp = trans.get('timestamp', datetime.now().isoformat())
            time_str = timestamp.split('T')[1][:8] if 'T' in timestamp else timestamp[:8]

            transcript_text = trans.get('transcript', '')[:200]
            if len(trans.get('transcript', '')) > 200:
                transcript_text += '...'

            transcript_html += f"""
            <div class="transcript-item" style="margin-bottom: 8px; padding: 8px 10px; border-left: 2px solid {color}; background: rgba(0,255,0,0.03); clip-path: polygon(0 0, calc(100% - 8px) 0, 100% 8px, 100% 100%, 0 100%);">
                <div style="font-size: 9px; color: #00aa00; margin-bottom: 4px; display: flex; justify-content: space-between;">
                    <span>[{time_str}] <span style="color: {color}; font-weight: bold;">{trans.get('channel', 'UNKNOWN')}</span></span>
                    <span style="opacity: 0.7;">W{trans.get('worker_id', '?')}</span>
                </div>
                <div style="font-size: 11px; color: #00ff00; font-family: monospace;">
                    {transcript_text}
                </div>
            </div>
            """

        # Build frequency updates
        freq_updates = {}
        for freq in updated_freqs:
            freq_updates[freq.replace('.', '_')] = self.channel_counters.get(freq, 0)

        # Use JSON to safely pass data
        freq_updates_json = json.dumps(freq_updates)
        transcript_html_escaped = transcript_html.replace('`', '\\`').replace('${', '\\${')

        # FIXED: Update DOM elements directly instead of replacing innerHTML repeatedly
        js_code = f"""
        (function() {{
            try {{
                const updates = {freq_updates_json};

                // Update frequency counters
                for (const [freqId, count] of Object.entries(updates)) {{
                    const counterEl = document.getElementById('channel-count-' + freqId);
                    if (counterEl && counterEl.textContent !== count.toString()) {{
                        counterEl.textContent = count;
                    }}
                }}

                // Update total transmissions
                const totalEl = document.getElementById('total-transmissions');
                const totalCount = {sum(self.channel_counters.values())};
                if (totalEl && totalEl.textContent !== totalCount.toString()) {{
                    totalEl.textContent = totalCount;
                }}

                // Update transcript content (only if changed)
                const transcriptContent = document.getElementById('transcript-content');
                if (transcriptContent) {{
                    const newContent = `{transcript_html_escaped}`;
                    if (transcriptContent.innerHTML !== newContent) {{
                        transcriptContent.innerHTML = newContent;
                        transcriptContent.scrollTop = transcriptContent.scrollHeight;
                    }}
                }}

                return true;
            }} catch (error) {{
                console.error('[ATC] Error updating UI:', error);
                return false;
            }}
        }})();
        """

        try:
            self.window.evaluate_js(js_code)
            latest = batch[-1]
            info(
                f"[{latest.get('channel', 'Unknown')}] TX #{sum(self.channel_counters.values())}: {latest.get('transcript', '')[:50]}...")
        except Exception as e:
            error(f"Error applying transmission batch: {e}")

    def update_aircraft(self, data):
        """Update aircraft position (stub for now)"""
        pass

    def flash_channel(self, frequency):
        """Flash channel indicator when recording"""
        if not self.window or not self.overlay_initialized:
            return

        freq_id = frequency.replace('.', '_')
        js_code = f"""
        (function() {{
            try {{
                const channelItem = document.querySelector('.channel-item[data-freq-id="{freq_id}"]');
                if (channelItem) {{
                    channelItem.style.background = 'rgba(0,255,0,0.2)';
                    setTimeout(() => {{
                        channelItem.style.background = 'rgba(255,255,255,0.03)';
                    }}, 300);
                }}
                return true;
            }} catch (error) {{
                console.error('[ATC] Error flashing channel:', error);
                return false;
            }}
        }})();
        """

        try:
            self.window.evaluate_js(js_code)
        except Exception as e:
            error(f"Error flashing channel: {e}")

    def update_worker_status(self, data):
        """Update worker status display - FIXED to prevent layering"""
        if not self.window or not self.overlay_initialized:
            return

        worker_id = data.get('worker_id', 0)
        status = data.get('status', 'idle')
        color = '#00ff00' if status == 'idle' else '#ff9900'
        bg_color = 'rgba(0,255,0,0.1)' if status == 'idle' else 'rgba(255,153,0,0.2)'
        border_color = 'rgba(0,255,0,0.3)' if status == 'idle' else 'rgba(255,153,0,0.5)'
        status_text = 'IDLE' if status == 'idle' else 'BUSY'
        channel_text = '' if status == 'idle' else f"<div style='font-size: 8px; margin-top: 2px; opacity: 0.8;'>{data.get('channel', '')[:8]}</div>"

        js_code = f"""
        (function() {{
            try {{
                const worker = document.getElementById('worker-{worker_id}');
                const workerStatus = document.getElementById('worker-{worker_id}-status');

                if (worker) {{
                    worker.style.background = '{bg_color}';
                    worker.style.borderColor = '{border_color}';

                    if ('{status}' === 'busy') {{
                        worker.style.animation = 'pulse 1s infinite';
                    }} else {{
                        worker.style.animation = '';
                    }}
                }}

                if (workerStatus) {{
                    workerStatus.innerHTML = '{status_text}{channel_text}';
                    workerStatus.style.color = '{color}';
                }}

                return true;
            }} catch (error) {{
                console.error('[ATC] Error updating worker status:', error);
                return false;
            }}
        }})();
        """

        try:
            self.window.evaluate_js(js_code)
        except Exception as e:
            error(f"Error updating worker status: {e}")

    def update_statistics(self, stats):
        """Update statistics display - FIXED to prevent layering"""
        if not self.window or not self.overlay_initialized:
            return

        queue_size = stats.get('queue_size', 0)
        workers_busy = stats.get('workers_busy', 0)

        js_code = f"""
        (function() {{
            try {{
                const queueEl = document.getElementById('queue-size');
                if (queueEl && queueEl.textContent !== '{queue_size}') {{
                    queueEl.textContent = {queue_size};
                }}

                const workersEl = document.getElementById('workers-busy');
                if (workersEl && workersEl.textContent !== '{workers_busy}') {{
                    workersEl.textContent = {workers_busy};
                }}

                return true;
            }} catch (error) {{
                console.error('[ATC] Error updating statistics:', error);
                return false;
            }}
        }})();
        """

        try:
            self.window.evaluate_js(js_code)
        except Exception as e:
            error(f"Error updating statistics: {e}")

    def show_recording_status(self, data):
        """Show recording status in GUI"""
        if not self.window or not self.overlay_initialized:
            return

        recording_num = data.get('recording_number', 0)

        js_code = f"""
        (function() {{
            try {{
                const statusEl = document.getElementById('monitor-status');
                if (statusEl) {{
                    statusEl.style.color = '#ff9900';
                    statusEl.textContent = '◉ REC #{recording_num}';

                    setTimeout(() => {{
                        statusEl.style.color = '#00ff00';
                        statusEl.textContent = '◉ ACTIVE';
                    }}, 2000);
                }}
                return true;
            }} catch (error) {{
                console.error('[ATC] Error showing recording status:', error);
                return false;
            }}
        }})();
        """

        try:
            self.window.evaluate_js(js_code)
        except Exception as e:
            error(f"Error showing recording status: {e}")

    def show_alert(self, data):
        """Show alert in GUI - FIXED to prevent duplicates"""
        if not self.window or not self.overlay_initialized:
            return

        alert_type = data.get('type', 'Unknown')
        alert_text = data.get('transcript', '')[:100]
        alert_id = f"alert-{int(time.time() * 1000)}"

        # Escape text for safe injection
        alert_text_escaped = alert_text.replace("'", "\\'").replace('"', '\\"').replace('\n', ' ')

        js_code = f"""
        (function() {{
            try {{
                // Remove any existing alerts first
                const existingAlert = document.getElementById('{alert_id}');
                if (existingAlert) {{
                    existingAlert.remove();
                }}

                const alert = document.createElement('div');
                alert.id = '{alert_id}';
                alert.style.cssText = `
                    position: fixed;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    background: linear-gradient(135deg, rgba(255, 0, 0, 0.95) 0%, rgba(200, 0, 0, 0.95) 100%);
                    color: white;
                    padding: 20px 25px;
                    font-family: 'Courier New', monospace;
                    z-index: 10002;
                    box-shadow: 0 0 0 3px #ff0000, 0 0 30px rgba(255,0,0,0.5), inset 0 0 20px rgba(255,255,255,0.1);
                    max-width: 500px;
                    animation: pulse 1s infinite;
                    clip-path: polygon(0 0, calc(100% - 20px) 0, 100% 20px, 100% 100%, 0 100%);
                `;

                alert.innerHTML = `
                    <div style="font-size: 16px; font-weight: 700; letter-spacing: 2px; margin-bottom: 10px; text-transform: uppercase;">
                        ⚠ ALERT: {alert_type} ⚠
                    </div>
                    <div style="font-size: 13px; line-height: 1.5;">
                        {alert_text_escaped}
                    </div>
                `;

                document.body.appendChild(alert);

                setTimeout(() => {{
                    if (alert.parentNode) {{
                        alert.remove();
                    }}
                }}, 10000);

                return true;
            }} catch (error) {{
                console.error('[ATC] Error showing alert:', error);
                return false;
            }}
        }})();
        """

        try:
            self.window.evaluate_js(js_code)
        except Exception as e:
            error(f"Error showing alert: {e}")

    def check_initialization_status(self):
        """Check if the overlay was successfully initialized"""
        if not self.running:
            return

        try:
            status_js = """
            (function() {
                if (window.atcGetStatus) {
                    return window.atcGetStatus();
                }
                return { initialized: false, error: 'Status function not found' };
            })();
            """

            status = self.window.evaluate_js(status_js)
            if status:
                info(f"Initialization status: {status}")

                if status.get('initialized'):
                    self.overlay_initialized = True
                    success("ATC overlay is active!")
                else:
                    if status.get('attempts', 0) < 30:
                        threading.Timer(2.0, self.check_initialization_status).start()
                    else:
                        error("Failed to initialize after maximum attempts")
            else:
                warning("Could not get initialization status")

        except Exception as e:
            error(f"Error checking status: {e}")

    def stop(self):
        """Stop the application"""
        self.running = False
        if self.window:
            self.window.destroy()


def run_webview_app(atc_monitor):
    """Run the webview application"""
    app = OpenSkyMapApp(atc_monitor)

    try:
        app.run()
    except KeyboardInterrupt:
        info("Shutting down...")
        app.stop()
    finally:
        atc_monitor.stop_monitoring()