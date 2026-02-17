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
            color = channel_config.get('color', '#00D4FF')
            stream_url = channel_config.get('stream_url', '')
            channels_html += f"""
            <div class="channel-item" style="margin-bottom: 1px; padding: 8px; background: #0A0A0A; border-left: 2px solid {color};">
                <div style="font-weight: 600; font-size: 11px; letter-spacing: 0.5px; text-transform: uppercase;">{channel_config['name']}</div>
                <div style="font-size: 10px; color: #6B9DB5; margin-top: 4px;">
                    {freq} MHz |
                    <span id="channel-count-{freq_id}" style="color: {color}; font-weight: 600;">0</span> TX
                    <button id="mute-{freq_id}" class="mute-btn" onclick="toggleMute('{freq_id}')">UNMUTE</button>
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
                if (document.getElementById('multi-channel-panel')) {{
                    window.multiChannelMonitorInjected = 'complete';
                    return true;
                }}
                window.multiChannelMonitorInjected = 'in_progress';

                console.log('[ATC] Injecting multi-channel monitor...');

                // Configuration
                const AIRPORT_LAT = {config.AIRPORT_LAT};
                const AIRPORT_LON = {config.AIRPORT_LON};
                const SEARCH_RADIUS_NM = {config.SEARCH_RADIUS_NM};
                const RADIUS_METERS = SEARCH_RADIUS_NM * 1852;

                // Auto-toggle labels (L) and extended labels (O)
                setTimeout(() => {{
                    try {{
                        console.log('[ATC] Attempting to toggle labels...');

                        const lButton = document.getElementById('L');
                        if (lButton) {{
                            if (!lButton.classList.contains('activeButton')) {{
                                lButton.click();
                                console.log('[ATC] Labels (L) toggled ON');
                            }} else {{
                                console.log('[ATC] Labels (L) already active');
                            }}
                        }} else {{
                            console.log('[ATC] Labels button (L) not found');
                        }}

                        const oButton = document.getElementById('O');
                        if (oButton) {{
                            if (!oButton.classList.contains('activeButton')) {{
                                oButton.click();
                                console.log('[ATC] Extended labels (O) toggled ON');
                            }} else {{
                                console.log('[ATC] Extended labels (O) already active');
                            }}
                        }} else {{
                            console.log('[ATC] Extended labels button (O) not found');
                        }}
                    }} catch (e) {{
                        console.error('[ATC] Error toggling labels:', e);
                    }}
                }}, 2000);

                // Draggable and resizable functionality
                function makeDraggable(element, handleSelector) {{
                    const handle = handleSelector ? element.querySelector(handleSelector) : element;
                    if (!handle) return;

                    let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;

                    handle.style.cursor = 'move';
                    handle.onmousedown = dragMouseDown;

                    function dragMouseDown(e) {{
                        e = e || window.event;
                        e.preventDefault();
                        e.stopPropagation();
                        handle.style.cursor = 'move';
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
                        handle.style.cursor = 'move';
                        document.onmouseup = null;
                        document.onmousemove = null;
                    }}
                }}

                function makeResizable(element) {{
                    const resizer = document.createElement('div');
                    resizer.className = 'resizer';
                    resizer.style.cssText = `
                        position: absolute;
                        right: 0;
                        bottom: 0;
                        width: 20px;
                        height: 20px;
                        cursor: nwse-resize;
                        z-index: 10;
                        opacity: 0.3;
                        transition: opacity 0.2s;
                    `;

                    resizer.innerHTML = `
                        <svg width="20" height="20" style="position: absolute; right: 0; bottom: 0;">
                            <line x1="20" y1="10" x2="10" y2="20" stroke="#00D4FF" stroke-width="2"/>
                            <line x1="20" y1="15" x2="15" y2="20" stroke="#00D4FF" stroke-width="2"/>
                            <line x1="20" y1="5" x2="5" y2="20" stroke="#00D4FF" stroke-width="2"/>
                        </svg>
                    `;

                    element.appendChild(resizer);

                    element.addEventListener('mouseenter', () => {{
                        resizer.style.opacity = '0.6';
                    }});
                    element.addEventListener('mouseleave', () => {{
                        if (!isResizing) resizer.style.opacity = '0.3';
                    }});

                    let startX, startY, startWidth, startHeight;
                    let isResizing = false;

                    resizer.addEventListener('mousedown', initResize);

                    function initResize(e) {{
                        e.preventDefault();
                        e.stopPropagation();
                        isResizing = true;
                        resizer.style.opacity = '1';
                        startX = e.clientX;
                        startY = e.clientY;
                        startWidth = parseInt(document.defaultView.getComputedStyle(element).width, 10);
                        startHeight = parseInt(document.defaultView.getComputedStyle(element).height, 10);
                        document.addEventListener('mousemove', resize);
                        document.addEventListener('mouseup', stopResize);
                    }}

                    function resize(e) {{
                        if (!isResizing) return;
                        const width = startWidth + e.clientX - startX;
                        const height = startHeight + e.clientY - startY;
                        element.style.width = Math.max(280, width) + 'px';
                        element.style.height = Math.max(200, height) + 'px';
                        element.style.maxHeight = Math.max(200, height) + 'px';
                    }}

                    function stopResize() {{
                        isResizing = false;
                        resizer.style.opacity = '0.3';
                        document.removeEventListener('mousemove', resize);
                        document.removeEventListener('mouseup', stopResize);
                    }}
                }}

                // Create main panel
                const panel = document.createElement('div');
                panel.id = 'multi-channel-panel';
                panel.style.cssText = `
                    position: fixed;
                    top: 80px;
                    right: 20px;
                    background: #000000;
                    color: #FFFFFF;
                    padding: 0;
                    border: 2px solid #00D4FF;
                    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                    z-index: 10000;
                    box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
                    width: 340px;
                    max-height: 600px;
                    overflow: hidden;
                    display: flex;
                    flex-direction: column;
                `;

                panel.innerHTML = `
                    <div class="drag-handle" style="
                        padding: 12px 16px;
                        background: linear-gradient(90deg, #000000 0%, #001F2B 100%);
                        border-bottom: 2px solid #00D4FF;
                        user-select: none;
                    ">
                        <h3 style="margin: 0; font-size: 14px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: #00D4FF;">
                            ▶ ATC MONITOR
                        </h3>
                    </div>

                    <div style="flex: 1; overflow-y: auto; padding: 0;">
                        <div style="padding: 12px; background: #0A0A0A; border-bottom: 1px solid #1A1A1A;">
                            <div style="display: grid; grid-template-columns: auto 1fr; gap: 8px 12px; font-size: 10px;">
                                <div style="color: #6B9DB5; text-transform: uppercase; letter-spacing: 0.5px;">STATUS</div>
                                <div id="monitor-status" style="color: #00FF7F; font-weight: 600;">◉ ACTIVE</div>

                                <div style="color: #6B9DB5; text-transform: uppercase; letter-spacing: 0.5px;">AREA</div>
                                <div style="color: #FFFFFF;">{config.LOCATION_NAME} | {config.SEARCH_RADIUS_NM} NM</div>

                                <div style="color: #6B9DB5; text-transform: uppercase; letter-spacing: 0.5px;">TOTAL TX</div>
                                <div><span id="total-transmissions" style="color: #00D4FF; font-weight: 600;">0</span></div>

                                <div style="color: #6B9DB5; text-transform: uppercase; letter-spacing: 0.5px;">QUEUE</div>
                                <div>
                                    <span id="queue-size" style="color: #00D4FF; font-weight: 600;">0</span> |
                                    <span style="color: #6B9DB5;">WORKERS:</span>
                                    <span id="workers-busy" style="color: #00D4FF; font-weight: 600;">0</span><span style="color: #6B9DB5;">/{self.num_workers}</span>
                                </div>

                                <div style="color: #6B9DB5; text-transform: uppercase; letter-spacing: 0.5px;">TRACKED</div>
                                <div><span id="tracked-aircraft" style="color: #00D4FF; font-weight: 600;">0</span> <span style="color: #6B9DB5;">AC</span></div>
                            </div>
                        </div>

                        <div style="padding: 12px 16px; background: #000000; border-bottom: 2px solid #00D4FF;">
                            <div style="font-size: 11px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; color: #00D4FF; margin-bottom: 8px;">CHANNELS [{len(self.atc_monitor.channel_configs)}]</div>
                        </div>
                        <div id="channel-list">
                            {channels_html}
                        </div>

                        <div style="padding: 12px 16px; background: #000000; border-top: 1px solid #1A1A1A; border-bottom: 2px solid #00D4FF; margin-top: 1px;">
                            <div style="font-size: 11px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; color: #00D4FF; margin-bottom: 8px;">WORKERS</div>
                        </div>
                        <div id="worker-status" style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px; background: #1A1A1A; padding: 0;">
                            {"".join(f'<div id="worker-{i}" class="worker-box" style="padding: 10px; background: #0A0A0A; text-align: center; font-size: 9px; letter-spacing: 0.5px; border: 1px solid #1A1A1A;"><div style="color: #6B9DB5; text-transform: uppercase; margin-bottom: 4px;">W{i}</div><div style="color: #00FF7F; font-weight: 600;">IDLE</div></div>' for i in range(self.num_workers))}
                        </div>
                    </div>
                `;

                document.body.appendChild(panel);
                makeDraggable(panel, '.drag-handle');
                makeResizable(panel);

                // Toggle audio playback for a channel without affecting recording
                window.toggleMute = function(freqId) {{
                    const audioEl = document.getElementById('audio-' + freqId);
                    const btnEl = document.getElementById('mute-' + freqId);
                    if (!audioEl || !btnEl) {{
                        return;
                    }}
                    if (audioEl.paused) {{
                        const streamUrl = audioEl.getAttribute('data-stream');
                        if (audioEl.src !== streamUrl) {{
                            audioEl.src = streamUrl;
                            audioEl.load();
                        }}
                        audioEl.play();
                        btnEl.textContent = 'MUTE';
                        btnEl.style.background = '#FF4444';
                    }} else {{
                        audioEl.pause();
                        audioEl.removeAttribute('src');
                        audioEl.load();
                        btnEl.textContent = 'UNMUTE';
                        btnEl.style.background = '#1A1A1A';
                    }}
                }};

                // Create transcript display
                const transcriptContainer = document.createElement('div');
                transcriptContainer.id = 'multi-transcript-container';
                transcriptContainer.style.cssText = `
                    position: fixed;
                    bottom: 20px;
                    left: 20px;
                    right: 380px;
                    max-width: 900px;
                    background: #000000;
                    color: #FFFFFF;
                    padding: 0;
                    border: 2px solid #00D4FF;
                    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                    z-index: 9999;
                    box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
                    max-height: 250px;
                    overflow: hidden;
                    display: flex;
                    flex-direction: column;
                `;

                transcriptContainer.innerHTML = `
                    <div class="transcript-drag-handle" style="
                        padding: 10px 16px;
                        background: linear-gradient(90deg, #000000 0%, #001F2B 100%);
                        border-bottom: 2px solid #00D4FF;
                        user-select: none;
                    ">
                        <strong style="font-size: 12px; letter-spacing: 2px; text-transform: uppercase; color: #00D4FF;">▶ TRANSMISSIONS</strong>
                    </div>
                    <div id="transcript-content" style="
                        flex: 1;
                        overflow-y: auto;
                        padding: 16px;
                        text-align: center;
                        color: #6B9DB5;
                        font-size: 11px;
                        letter-spacing: 0.5px;
                    ">AWAITING TRANSMISSION DATA...</div>
                `;

                document.body.appendChild(transcriptContainer);
                makeDraggable(transcriptContainer, '.transcript-drag-handle');
                makeResizable(transcriptContainer);

                // Add custom styles
                const style = document.createElement('style');
                style.textContent = `
                    #multi-channel-panel > div:last-child::-webkit-scrollbar,
                    #transcript-content::-webkit-scrollbar {{
                        width: 8px;
                    }}

                    #multi-channel-panel > div:last-child::-webkit-scrollbar-track,
                    #transcript-content::-webkit-scrollbar-track {{
                        background: #0A0A0A;
                    }}

                    #multi-channel-panel > div:last-child::-webkit-scrollbar-thumb,
                    #transcript-content::-webkit-scrollbar-thumb {{
                        background: #00D4FF;
                    }}

                    #multi-channel-panel > div:last-child::-webkit-scrollbar-thumb:hover,
                    #transcript-content::-webkit-scrollbar-thumb:hover {{
                        background: #00A8CC;
                    }}

                    .worker-box {{
                        transition: all 0.2s ease;
                    }}

                    .mute-btn {{
                        margin-left: 8px;
                        padding: 2px 8px;
                        font-size: 8px;
                        cursor: pointer;
                        background: #1A1A1A;
                        border: 1px solid #00D4FF;
                        color: #00D4FF;
                        font-weight: 600;
                        letter-spacing: 0.5px;
                        transition: all 0.2s;
                    }}

                    .mute-btn:hover {{
                        background: #00D4FF;
                        color: #000000;
                    }}

                    .drag-handle, .transcript-drag-handle {{
                        transition: background 0.2s;
                    }}

                    .drag-handle:hover, .transcript-drag-handle:hover {{
                        background: linear-gradient(90deg, #001F2B 0%, #003A52 100%) !important;
                    }}

                    @keyframes pulse {{
                        0%, 100% {{ opacity: 1; }}
                        50% {{ opacity: 0.6; }}
                    }}

                    @keyframes fadeInUp {{
                        from {{
                            opacity: 0;
                            transform: translateY(10px);
                        }}
                        to {{
                            opacity: 1;
                            transform: translateY(0);
                        }}
                    }}
                `;
                document.head.appendChild(style);

                // -------------------------------------------------------
                // Monitoring circle overlay with radar sweep + OpenSky
                // -------------------------------------------------------
                let overlayInitialized = false;
                let initAttempts = 0;

                // Sweep state (closure-level so the draw loop can access)
                let sweepAngle = -Math.PI / 2;
                const SWEEP_SPEED = 0.0157;   // ~9 RPM at 60 fps (25 % slower)
                let animFrameId = null;

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

                            const existingOverlay = document.getElementById('atc-overlay-canvas');
                            if (existingOverlay) {{
                                existingOverlay.remove();
                            }}

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
                            }} else {{
                                console.error('[ATC] Map canvas not found');
                                return;
                            }}

                            // Canvas size cache to avoid unnecessary resets
                            let lastW = 0, lastH = 0;

                            // ------------------------------------------
                            //  Main draw function (sweep + circle + highlights)
                            // ------------------------------------------
                            function drawMonitoringCircle() {{
                                const canvas = document.getElementById('atc-overlay-canvas');
                                if (!canvas || !map) return;

                                const w = canvas.offsetWidth;
                                const h = canvas.offsetHeight;
                                if (w !== lastW || h !== lastH) {{
                                    canvas.width = w;
                                    canvas.height = h;
                                    lastW = w;
                                    lastH = h;
                                }}

                                const ctx = canvas.getContext('2d');
                                ctx.clearRect(0, 0, canvas.width, canvas.height);

                                const centerCoords = ol.proj.fromLonLat([AIRPORT_LON, AIRPORT_LAT]);
                                const centerPixel = map.getPixelFromCoordinate(centerCoords);
                                if (!centerPixel) return;

                                const resolution = map.getView().getResolution();
                                const radiusPixels = RADIUS_METERS / resolution;
                                const cx = centerPixel[0];
                                const cy = centerPixel[1];

                                // --- Sweep trail (conical fade) ---
                                const trailAng = Math.PI / 2.2;
                                const segs = 60;
                                for (let i = 0; i < segs; i++) {{
                                    const a1 = sweepAngle - trailAng * (i + 1) / segs;
                                    const a2 = sweepAngle - trailAng * i / segs;
                                    const t  = i / segs;
                                    const alpha = Math.pow(1 - t, 2.2) * 0.18;
                                    ctx.fillStyle = 'rgba(190,0,0,' + alpha + ')';
                                    ctx.beginPath();
                                    ctx.moveTo(cx, cy);
                                    ctx.arc(cx, cy, radiusPixels - 1, a1, a2);
                                    ctx.closePath();
                                    ctx.fill();
                                }}

                                // --- Sweep leading edge ---
                                const ex = cx + radiusPixels * Math.cos(sweepAngle);
                                const ey = cy + radiusPixels * Math.sin(sweepAngle);
                                ctx.save();
                                ctx.shadowColor = 'rgba(190,0,0,0.7)';
                                ctx.shadowBlur = 14;
                                const grad = ctx.createLinearGradient(cx, cy, ex, ey);
                                grad.addColorStop(0,   'rgba(190,0,0,0.1)');
                                grad.addColorStop(0.5, 'rgba(190,0,0,0.55)');
                                grad.addColorStop(1,   'rgba(190,0,0,0.95)');
                                ctx.strokeStyle = grad;
                                ctx.lineWidth = 2;
                                ctx.beginPath();
                                ctx.moveTo(cx, cy);
                                ctx.lineTo(ex, ey);
                                ctx.stroke();
                                ctx.restore();

                                // --- Monitoring radius circle (dashed) ---
                                ctx.strokeStyle = 'rgba(190, 0, 0, 0.8)';
                                ctx.lineWidth = 2;
                                ctx.setLineDash([8, 8]);
                                ctx.beginPath();
                                ctx.arc(cx, cy, radiusPixels, 0, 2 * Math.PI);
                                ctx.stroke();
                                ctx.setLineDash([]);

                                ctx.fillStyle = 'rgba(255, 0, 0, 0.05)';
                                ctx.beginPath();
                                ctx.arc(cx, cy, radiusPixels, 0, 2 * Math.PI);
                                ctx.fill();

                                // --- Center glow (red) ---
                                const cg = ctx.createRadialGradient(cx, cy, 0, cx, cy, 12);
                                cg.addColorStop(0, 'rgba(190,0,0,0.5)');
                                cg.addColorStop(1, 'rgba(190,0,0,0)');
                                ctx.fillStyle = cg;
                                ctx.beginPath();
                                ctx.arc(cx, cy, 12, 0, 2 * Math.PI);
                                ctx.fill();

                                // --- Center dot (red) ---
                                ctx.fillStyle = 'rgba(190,0,0,0.9)';
                                ctx.beginPath();
                                ctx.arc(cx, cy, 3, 0, 2 * Math.PI);
                                ctx.fill();

                                // --- Draw highlighted aircraft (OpenSky) ---
                                const highlighted = window.atcHighlightedAircraft || {{}};
                                const now = Date.now();
                                for (const cs in highlighted) {{
                                    const hl = highlighted[cs];
                                    if (now - hl.highlightTime > hl.ttl) {{
                                        delete highlighted[cs];
                                        continue;
                                    }}
                                    const ac = window.atcAircraftCache ? window.atcAircraftCache[cs] : null;
                                    if (!ac) continue;
                                    // Re-project pixel each frame (aircraft moves)
                                    const px = ac.pixel ? ac.pixel[0] : null;
                                    const py = ac.pixel ? ac.pixel[1] : null;
                                    if (px === null || py === null) continue;

                                    const age = (now - hl.highlightTime) / hl.ttl;
                                    const a = 1 - age;

                                    // Highlight ring
                                    ctx.strokeStyle = 'rgba(255,200,0,' + (a * 0.8) + ')';
                                    ctx.lineWidth = 2;
                                    ctx.beginPath();
                                    ctx.arc(px, py, 15, 0, Math.PI * 2);
                                    ctx.stroke();

                                    // Pulsing outer ring
                                    const pulse = 0.5 + 0.5 * Math.sin(now / 200);
                                    ctx.strokeStyle = 'rgba(255,200,0,' + (a * 0.3 * pulse) + ')';
                                    ctx.lineWidth = 1;
                                    ctx.beginPath();
                                    ctx.arc(px, py, 22, 0, Math.PI * 2);
                                    ctx.stroke();

                                    // Callsign label
                                    ctx.fillStyle = 'rgba(255,200,0,' + a + ')';
                                    ctx.font = '10px Consolas, Monaco, monospace';
                                    ctx.textAlign = 'left';
                                    ctx.fillText(cs, px + 18, py - 4);
                                }}
                            }}

                            // ------------------------------------------
                            //  Animation loop
                            // ------------------------------------------
                            function animateRadar() {{
                                sweepAngle += SWEEP_SPEED;
                                if (sweepAngle > Math.PI * 3) sweepAngle -= Math.PI * 2;
                                drawMonitoringCircle();
                                animFrameId = requestAnimationFrame(animateRadar);
                            }}

                            // Pause when tab is hidden to save CPU
                            document.addEventListener('visibilitychange', function() {{
                                if (document.hidden) {{
                                    if (animFrameId) {{
                                        cancelAnimationFrame(animFrameId);
                                        animFrameId = null;
                                    }}
                                }} else if (!animFrameId) {{
                                    animateRadar();
                                }}
                            }});

                            animateRadar();

                            // ==================================================
                            //  OpenSky Aircraft Feature Integration (stubs)
                            // ==================================================
                            window.atcAircraftCache = {{}};
                            window.atcHighlightedAircraft = {{}};
                            window.atcAircraftLastScan = 0;
                            const AIRCRAFT_SCAN_INTERVAL = 3000;  // ms

                            /**
                             * Walk every vector layer on the OL map and cache
                             * aircraft feature properties + projected pixel coords.
                             */
                            function scanOpenSkyFeatures() {{
                                if (!map) return {{}};
                                const cache = {{}};
                                try {{
                                    map.getLayers().forEach(function(layer) {{
                                        var src;
                                        try {{ src = layer.getSource && layer.getSource(); }} catch(_) {{ return; }}
                                        if (!src || typeof src.getFeatures !== 'function') return;
                                        src.getFeatures().forEach(function(feature) {{
                                            try {{
                                                const props = feature.getProperties();
                                                const cs = (props.callsign || props.name || props.flight || '').toString().trim();
                                                if (!cs) return;

                                                const geom = feature.getGeometry();
                                                if (!geom || !geom.getCoordinates) return;
                                                const coords = geom.getCoordinates();
                                                if (!coords) return;

                                                const pixel = map.getPixelFromCoordinate(coords);
                                                let lonLat = null;
                                                try {{ lonLat = ol.proj.toLonLat(coords); }} catch(_) {{}}

                                                cache[cs] = {{
                                                    callsign:  cs,
                                                    icao24:    props.icao24 || props.hex || '',
                                                    coords:    coords,
                                                    lonLat:    lonLat,
                                                    pixel:     pixel,
                                                    altitude:  props.altitude  || props.baro_altitude || props.geo_altitude || 0,
                                                    velocity:  props.velocity  || props.speed || 0,
                                                    heading:   props.heading   || props.true_track   || props.track || 0,
                                                    on_ground: !!props.on_ground,
                                                    squawk:    props.squawk || '',
                                                    feature:   feature
                                                }};
                                            }} catch(_) {{}}
                                        }});
                                    }});
                                }} catch (e) {{
                                    console.warn('[ATC] Error scanning features:', e);
                                }}
                                window.atcAircraftCache = cache;
                                window.atcAircraftLastScan = Date.now();

                                // Push count to panel
                                const countEl = document.getElementById('tracked-aircraft');
                                if (countEl) countEl.textContent = Object.keys(cache).length;

                                return cache;
                            }}

                            // Kick off periodic scanning
                            scanOpenSkyFeatures();
                            setInterval(scanOpenSkyFeatures, AIRCRAFT_SCAN_INTERVAL);

                            /**
                             * Highlight an aircraft by callsign on the overlay
                             * canvas.  ttl = how long to keep the highlight (ms).
                             */
                            window.atcHighlightCallsign = function(callsign, ttl) {{
                                const ac = window.atcAircraftCache[callsign];
                                if (!ac || !ac.pixel) {{
                                    console.log('[ATC] Callsign not found in cache: ' + callsign);
                                    return false;
                                }}
                                window.atcHighlightedAircraft[callsign] = {{
                                    callsign:      callsign,
                                    pixel:         ac.pixel,
                                    highlightTime: Date.now(),
                                    ttl:           ttl || 10000
                                }};
                                console.log('[ATC] Highlighting aircraft: ' + callsign);
                                return true;
                            }};

                            /**
                             * Fuzzy-match a transcript string against all cached
                             * callsigns.  Returns an array of matching cache entries.
                             */
                            window.atcMatchTranscript = function(transcript) {{
                                const cache = window.atcAircraftCache;
                                const matches = [];
                                const upper = transcript.toUpperCase().replace(/[^A-Z0-9]/g, '');
                                for (const cs in cache) {{
                                    const norm = cs.toUpperCase().replace(/[^A-Z0-9]/g, '');
                                    if (norm && upper.indexOf(norm) !== -1) {{
                                        matches.push(cache[cs]);
                                    }}
                                }}
                                return matches;
                            }};

                            /** Return the full aircraft cache object. */
                            window.atcGetAircraft = function() {{
                                return window.atcAircraftCache;
                            }};

                            /** Return number of currently-cached aircraft. */
                            window.atcGetAircraftCount = function() {{
                                return Object.keys(window.atcAircraftCache).length;
                            }};

                            overlayInitialized = true;
                            console.log('[ATC] Monitoring radius overlay with sweep initialized!');

                            window.atcOverlay = {{
                                map:              map,
                                canvas:           overlayCanvas,
                                draw:             drawMonitoringCircle,
                                initialized:      true,
                                scanAircraft:     scanOpenSkyFeatures,
                                highlightCallsign: window.atcHighlightCallsign,
                                matchTranscript:  window.atcMatchTranscript,
                                getAircraft:      window.atcGetAircraft,
                                getAircraftCount: window.atcGetAircraftCount
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

    # -----------------------------------------------------------------
    #  Queue / update processing
    # -----------------------------------------------------------------

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
        """Apply a batch of transmissions to the UI."""
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

        transcript_html = ""
        for trans in self.displayed_transcripts:
            color = trans.get('color', '#00D4FF')
            timestamp = trans.get('timestamp', datetime.now().isoformat())
            time_str = timestamp.split('T')[1][:8] if 'T' in timestamp else timestamp

            transcript_html += f"""
            <div style="margin-bottom: 1px; padding: 10px; background: #0A0A0A; border-left: 2px solid {color}; animation: fadeInUp 0.3s ease-out;">
                <div style="font-size: 9px; color: #6B9DB5; margin-bottom: 6px; letter-spacing: 0.5px; text-transform: uppercase;">
                    [{time_str}]
                    <span style="color: {color}; font-weight: 700;">{trans.get('channel', 'Unknown')}</span>
                    <span style="float: right;">W{trans.get('worker_id', '?')}</span>
                </div>
                <div style="font-size: 11px; color: #FFFFFF; line-height: 1.4; font-family: 'Consolas', 'Monaco', monospace;">
                    {trans.get('transcript', '')[:200]}{'...' if len(trans.get('transcript', '')) > 200 else ''}
                </div>
            </div>
            """

        freq_updates = ", ".join(
            [f"'{freq.replace('.', '_')}': {self.channel_counters.get(freq, 0)}" for freq in updated_freqs]
        )

        js_code = f"""
        (function() {{
            const updates = {{{freq_updates}}};
            for (const [freq, count] of Object.entries(updates)) {{
                const counterEl = document.getElementById('channel-count-' + freq);
                if (counterEl) {{
                    counterEl.textContent = count;
                }}
            }}

            const totalEl = document.getElementById('total-transmissions');
            if (totalEl) {{
                totalEl.textContent = {sum(self.channel_counters.values())};
            }}

            const contentEl = document.getElementById('transcript-content');
            if (contentEl) {{
                contentEl.innerHTML = `{transcript_html}`;
                contentEl.scrollTop = contentEl.scrollHeight;
            }}
        }})();
        """

        try:
            self.window.evaluate_js(js_code)
        except Exception as e:
            error(f"Error updating UI: {e}")

        latest = batch[-1]
        info(
            f"[{latest.get('channel', 'Unknown')}] Transmission #{sum(self.channel_counters.values())}: "
            f"{latest.get('transcript', '')[:50]}..."
        )

    # -----------------------------------------------------------------
    #  Aircraft / OpenSky helpers (Python side)
    # -----------------------------------------------------------------

    def update_aircraft(self, data):
        """Update aircraft position and optionally highlight on map"""
        callsign = data.get('callsign', '')
        if callsign:
            self.highlight_aircraft(callsign)

    def get_tracked_aircraft(self):
        """Retrieve cached aircraft data from the OpenSky map layer.

        Returns a dict keyed by callsign with position, altitude, etc.
        """
        if not self.window or not self.overlay_initialized:
            return {}
        try:
            result = self.window.evaluate_js("window.atcGetAircraft();")
            return result or {}
        except Exception as e:
            error(f"Error getting aircraft data: {e}")
            return {}

    def highlight_aircraft(self, callsign, ttl=10000):
        """Highlight an aircraft on the map by callsign.

        The highlight ring persists for *ttl* milliseconds.
        Returns True if the callsign was found in the cache.
        """
        if not self.window or not self.overlay_initialized:
            return False
        try:
            safe_cs = callsign.replace("'", "\\'").replace('"', '\\"')
            return bool(
                self.window.evaluate_js(
                    f"window.atcHighlightCallsign('{safe_cs}', {ttl});"
                )
            )
        except Exception as e:
            error(f"Error highlighting aircraft: {e}")
            return False

    def match_transcript_aircraft(self, transcript):
        """Match a transcript string against visible aircraft callsigns.

        Returns a list of matching aircraft cache entries (dicts).
        """
        if not self.window or not self.overlay_initialized:
            return []
        try:
            safe_text = transcript.replace("'", "\\'").replace('"', '\\"').replace('\n', ' ')
            return self.window.evaluate_js(
                f"window.atcMatchTranscript('{safe_text}');"
            ) or []
        except Exception as e:
            error(f"Error matching transcript: {e}")
            return []

    # -----------------------------------------------------------------
    #  Channel / worker / stats UI updates
    # -----------------------------------------------------------------

    def flash_channel(self, frequency):
        """Flash channel indicator when recording"""
        if not self.window or not self.overlay_initialized:
            return

        freq_id = frequency.replace('.', '_')
        js_code = f"""
        (function() {{
            const channelEl = document.querySelector('#channel-count-{freq_id}');
            if (channelEl) {{
                const parentEl = channelEl.parentElement.parentElement;
                const originalBg = parentEl.style.background;
                parentEl.style.background = '#001F2B';
                parentEl.style.borderLeft = '2px solid #00FF7F';
                setTimeout(() => {{
                    parentEl.style.background = originalBg;
                    parentEl.style.borderLeft = '';
                }}, 300);
            }}
        }})();
        """

        try:
            self.window.evaluate_js(js_code)
        except Exception as e:
            error(f"Error flashing channel: {e}")

    def update_worker_status(self, data):
        """Update worker status display"""
        if not self.window or not self.overlay_initialized:
            return

        worker_id = data.get('worker_id', 0)
        status = data.get('status', 'idle')

        if status == 'idle':
            bg_color = '#0A0A0A'
            text_color = '#00FF7F'
            status_text = 'IDLE'
            border = '1px solid #1A1A1A'
        else:
            bg_color = '#1A1A1A'
            text_color = '#FF9900'
            channel = data.get('channel', '')
            status_text = f'ACTIVE<br><span style="font-size: 8px; color: #6B9DB5;">{channel[:8]}</span>'
            border = '1px solid #FF9900'

        js_code = f"""
        (function() {{
            const worker = document.getElementById('worker-{worker_id}');
            if (worker) {{
                worker.style.background = '{bg_color}';
                worker.style.border = '{border}';
                worker.innerHTML = '<div style="color: #6B9DB5; text-transform: uppercase; margin-bottom: 4px; font-size: 9px; letter-spacing: 0.5px;">W{worker_id}</div><div style="color: {text_color}; font-weight: 600; font-size: 9px;">{status_text}</div>';
                if ('{status}' === 'busy') {{
                    worker.style.animation = 'pulse 1.5s infinite';
                }} else {{
                    worker.style.animation = '';
                }}
            }}
        }})();
        """

        try:
            self.window.evaluate_js(js_code)
        except Exception as e:
            error(f"Error updating worker status: {e}")

    def update_statistics(self, stats):
        """Update statistics display"""
        if not self.window or not self.overlay_initialized:
            return

        queue_size = stats.get('queue_size', 0)
        workers_busy = stats.get('workers_busy', 0)

        js_code = f"""
        (function() {{
            const queueEl = document.getElementById('queue-size');
            if (queueEl) queueEl.textContent = {queue_size};

            const workersEl = document.getElementById('workers-busy');
            if (workersEl) workersEl.textContent = {workers_busy};
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

        js_code = f"""
        (function() {{
            const statusEl = document.getElementById('monitor-status');
            if (statusEl) {{
                statusEl.style.color = '#FF9900';
                statusEl.textContent = '◉ REC #{data.get('recording_number', 0)}';

                setTimeout(() => {{
                    statusEl.style.color = '#00FF7F';
                    statusEl.textContent = '◉ ACTIVE';
                }}, 2000);
            }}
            return true;
        }})();
        """

        try:
            self.window.evaluate_js(js_code)
        except Exception as e:
            error(f"Error showing recording status: {e}")

    def show_alert(self, data):
        """Show alert in GUI"""
        if not self.window or not self.overlay_initialized:
            return

        alert_type = data.get('type', 'Unknown')
        alert_text = data.get('transcript', '')[:100]

        alert_text = alert_text.replace("'", "\\'").replace('"', '\\"').replace('\n', ' ')

        js_code = f"""
        (function() {{
            const alert = document.createElement('div');
            alert.style.cssText = `
                position: fixed;
                top: 300px;
                right: 20px;
                background: #000000;
                color: #FF0000;
                padding: 16px 20px;
                border: 2px solid #FF0000;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                z-index: 10001;
                box-shadow: 0 0 30px rgba(255, 0, 0, 0.5);
                max-width: 400px;
                animation: pulse 1s infinite;
            `;

            alert.innerHTML = `
                <h4 style="margin: 0 0 10px 0; font-size: 12px; letter-spacing: 2px; text-transform: uppercase;">⚠ ALERT: {alert_type.upper()}</h4>
                <p style="margin: 0; font-size: 11px; color: #FFFFFF; line-height: 1.4;">{alert_text}</p>
            `;

            document.body.appendChild(alert);

            setTimeout(() => {{
                alert.remove();
            }}, 10000);

            return true;
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
                    success("✓ ATC overlay is active!")
                else:
                    if status.get('attempts', 0) < 30:
                        threading.Timer(2.0, self.check_initialization_status).start()
                    else:
                        warning("Circle overlay may not be visible - this is usually fine")
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