import webview
import threading
import queue
import json
from datetime import datetime
from utils.config import AIRPORT_LAT, AIRPORT_LON, SEARCH_RADIUS_NM
from utils.console_logger import info, success, error, warning


class OpenSkyMapApp:
    """WebView-based OpenSky map application with ATC overlay"""

    def __init__(self, atc_monitor):
        self.atc_monitor = atc_monitor
        self.update_queue = queue.Queue()
        self.atc_monitor.set_gui_queue(self.update_queue)

        self.window = None
        self.running = True
        self.transmission_count = 0
        self.overlay_initialized = False
        self.page_loaded = False

        # For transcript display
        self.displayed_transcripts = []
        self.max_displayed_transcripts = 10

        # Check if this is multi-channel monitor
        self.is_multi_channel = hasattr(atc_monitor, 'channel_configs')

        # Channel-specific counters for multi-channel mode
        if self.is_multi_channel:
            self.channel_counters = {}
            for config in atc_monitor.channel_configs:
                self.channel_counters[config['frequency']] = 0
            self.num_workers = getattr(atc_monitor, 'transcription_pool', {}).num_workers if hasattr(atc_monitor,
                                                                                                     'transcription_pool') else 3
        else:
            self.channel_counters = {}
            self.num_workers = 1

    def run(self):
        """Run the application"""
        # Create window with the OpenSky map URL
        window_title = 'Multi-Channel ATC Monitor' if self.is_multi_channel else 'ATC Monitor - OpenSky Map'
        self.window = webview.create_window(
            window_title,
            f'https://map.opensky-network.org/?lat={AIRPORT_LAT}&lon={AIRPORT_LON}&zoom=10',
            width=1600 if self.is_multi_channel else 1400,
            height=900
        )

        # Set up event handlers
        self.window.events.loaded += self.on_page_loaded
        self.window.events.closed += self.on_closed

        # Start update thread
        update_thread = threading.Thread(target=self.process_updates, daemon=True)
        update_thread.start()

        # Start webview (blocks until window is closed)
        webview.start(debug=True)

    def on_page_loaded(self):
        """Called when page is fully loaded"""
        info("Page loaded event received")
        self.page_loaded = True

        # Wait a bit for OpenLayers to initialize
        threading.Timer(5.0, self.inject_monitor).start()

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

        info("Injecting ATC monitor...")

        # First inject the draggable functionality
        self._inject_draggable_system()

        if self.is_multi_channel:
            self._inject_multi_channel_monitor()
        else:
            self._inject_single_channel_monitor()

    def _inject_draggable_system(self):
        """Inject the reusable draggable system"""
        draggable_js = """
        (function() {
            if (window.draggableSystemInjected) {
                return true;
            }
            window.draggableSystemInjected = true;

            console.log('[ATC] Injecting draggable system...');

            // Storage key prefix for positions
            const STORAGE_PREFIX = 'atc_panel_pos_';

            // Default positions for panels
            window.defaultPanelPositions = {
                'multi-channel-panel': { top: 80, right: 20, bottom: null, left: null },
                'multi-transcript-container': { top: null, right: null, bottom: 20, left: 20 },
                'atc-transcript-container': { top: null, right: null, bottom: 20, left: 20 },
                'atc-info-panel': { top: 80, right: 20, bottom: null, left: null }
            };

            // Save panel position to localStorage
            window.savePanelPosition = function(panelId, position) {
                try {
                    localStorage.setItem(STORAGE_PREFIX + panelId, JSON.stringify(position));
                } catch (e) {
                    console.warn('[ATC] Could not save panel position:', e);
                }
            };

            // Load panel position from localStorage
            window.loadPanelPosition = function(panelId) {
                try {
                    const saved = localStorage.getItem(STORAGE_PREFIX + panelId);
                    if (saved) {
                        return JSON.parse(saved);
                    }
                } catch (e) {
                    console.warn('[ATC] Could not load panel position:', e);
                }
                return null;
            };

            // Reset panel to default position
            window.resetPanelPosition = function(panelId) {
                const panel = document.getElementById(panelId);
                const defaults = window.defaultPanelPositions[panelId];

                if (panel && defaults) {
                    panel.style.top = defaults.top !== null ? defaults.top + 'px' : 'auto';
                    panel.style.right = defaults.right !== null ? defaults.right + 'px' : 'auto';
                    panel.style.bottom = defaults.bottom !== null ? defaults.bottom + 'px' : 'auto';
                    panel.style.left = defaults.left !== null ? defaults.left + 'px' : 'auto';

                    try {
                        localStorage.removeItem(STORAGE_PREFIX + panelId);
                    } catch (e) {}

                    console.log('[ATC] Reset position for:', panelId);
                }
            };

            // Reset all panels to default positions
            window.resetAllPanelPositions = function() {
                Object.keys(window.defaultPanelPositions).forEach(panelId => {
                    window.resetPanelPosition(panelId);
                });
            };

            // Make an element draggable
            window.makeDraggable = function(element, handleSelector) {
                if (!element) return;

                const handle = handleSelector ? element.querySelector(handleSelector) : element;
                if (!handle) return;

                let isDragging = false;
                let startX, startY;
                let startLeft, startTop;
                let startRight, startBottom;

                // Apply saved position if available
                const savedPos = window.loadPanelPosition(element.id);
                if (savedPos) {
                    if (savedPos.left !== null) {
                        element.style.left = savedPos.left + 'px';
                        element.style.right = 'auto';
                    }
                    if (savedPos.top !== null) {
                        element.style.top = savedPos.top + 'px';
                        element.style.bottom = 'auto';
                    }
                }

                // Style the handle
                handle.style.cursor = 'move';
                handle.style.userSelect = 'none';

                handle.addEventListener('mousedown', function(e) {
                    // Don't drag if clicking on buttons or inputs
                    if (e.target.tagName === 'BUTTON' || e.target.tagName === 'INPUT' || 
                        e.target.classList.contains('no-drag')) {
                        return;
                    }

                    isDragging = true;
                    startX = e.clientX;
                    startY = e.clientY;

                    // Get current position
                    const rect = element.getBoundingClientRect();
                    const computedStyle = window.getComputedStyle(element);

                    // Convert to left/top positioning for dragging
                    startLeft = rect.left;
                    startTop = rect.top;

                    // Switch to left/top positioning
                    element.style.left = startLeft + 'px';
                    element.style.top = startTop + 'px';
                    element.style.right = 'auto';
                    element.style.bottom = 'auto';

                    // Visual feedback
                    element.style.opacity = '0.9';
                    element.style.boxShadow = '0 8px 32px rgba(0,0,0,0.6)';
                    element.style.transition = 'none';

                    e.preventDefault();
                });

                document.addEventListener('mousemove', function(e) {
                    if (!isDragging) return;

                    const deltaX = e.clientX - startX;
                    const deltaY = e.clientY - startY;

                    let newLeft = startLeft + deltaX;
                    let newTop = startTop + deltaY;

                    // Keep within viewport bounds
                    const rect = element.getBoundingClientRect();
                    const maxLeft = window.innerWidth - rect.width;
                    const maxTop = window.innerHeight - rect.height;

                    newLeft = Math.max(0, Math.min(newLeft, maxLeft));
                    newTop = Math.max(0, Math.min(newTop, maxTop));

                    element.style.left = newLeft + 'px';
                    element.style.top = newTop + 'px';
                });

                document.addEventListener('mouseup', function(e) {
                    if (!isDragging) return;

                    isDragging = false;

                    // Reset visual feedback
                    element.style.opacity = '';
                    element.style.boxShadow = '';
                    element.style.transition = '';

                    // Save position
                    const rect = element.getBoundingClientRect();
                    window.savePanelPosition(element.id, {
                        left: rect.left,
                        top: rect.top
                    });

                    console.log('[ATC] Saved position for:', element.id);
                });

                // Prevent text selection while dragging
                handle.addEventListener('selectstart', function(e) {
                    if (isDragging) e.preventDefault();
                });

                console.log('[ATC] Made draggable:', element.id);
            };

            // Add global styles for draggable elements
            const style = document.createElement('style');
            style.textContent = `
                .drag-handle {
                    cursor: move !important;
                    user-select: none;
                }

                .drag-handle:hover {
                    background: rgba(255, 255, 255, 0.1);
                }

                .drag-handle:active {
                    background: rgba(255, 255, 255, 0.15);
                }

                .panel-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin: -20px -20px 15px -20px;
                    padding: 12px 15px;
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 8px 8px 0 0;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                }

                .panel-title {
                    font-size: 16px;
                    font-weight: 600;
                    margin: 0;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }

                .panel-controls {
                    display: flex;
                    gap: 5px;
                }

                .panel-btn {
                    background: rgba(255, 255, 255, 0.1);
                    border: none;
                    color: white;
                    width: 24px;
                    height: 24px;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 12px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    transition: background 0.2s;
                }

                .panel-btn:hover {
                    background: rgba(255, 255, 255, 0.2);
                }

                .panel-btn.reset-btn:hover {
                    background: rgba(255, 100, 100, 0.3);
                }

                .resize-handle {
                    position: absolute;
                    bottom: 0;
                    right: 0;
                    width: 15px;
                    height: 15px;
                    cursor: se-resize;
                    opacity: 0.5;
                }

                .resize-handle::after {
                    content: '';
                    position: absolute;
                    bottom: 3px;
                    right: 3px;
                    width: 8px;
                    height: 8px;
                    border-right: 2px solid rgba(255,255,255,0.5);
                    border-bottom: 2px solid rgba(255,255,255,0.5);
                }
            `;
            document.head.appendChild(style);

            console.log('[ATC] Draggable system ready');
            return true;
        })();
        """

        try:
            self.window.evaluate_js(draggable_js)
            success("Draggable system injected")
        except Exception as e:
            error(f"Error injecting draggable system: {e}")

    def _inject_multi_channel_monitor(self):
        """Inject multi-channel monitoring interface"""
        # Generate channel list HTML
        channels_html = ""
        for config in self.atc_monitor.channel_configs:
            freq = config['frequency']
            freq_id = freq.replace('.', '_')
            color = config.get('color', '#FFFFFF')
            stream_url = config.get('stream_url', '')
            channels_html += f"""
            <div class="channel-item" style="margin-bottom: 8px; padding: 5px; border-left: 3px solid {color};">
                <div style="font-weight: bold; font-size: 12px;">{config['name']}</div>
                <div style="font-size: 11px; color: #888;">
                    {freq} MHz -
                    <span id="channel-count-{freq_id}" style="color: {color};">0</span> transmissions
                    <button id="mute-{freq_id}" class="mute-btn no-drag" onclick="toggleMute('{freq_id}')">Unmute</button>
                    <audio id="audio-{freq_id}" data-stream="{stream_url}" preload="none" style="display:none;"></audio>
                </div>
            </div>
            """

        injection_js = f"""
        (function() {{
            if (window.multiChannelMonitorInjected) {{
                return true;
            }}
            window.multiChannelMonitorInjected = true;

            console.log('[ATC] Injecting multi-channel monitor...');

            // Configuration
            const AIRPORT_LAT = {AIRPORT_LAT};
            const AIRPORT_LON = {AIRPORT_LON};
            const SEARCH_RADIUS_NM = {SEARCH_RADIUS_NM};
            const RADIUS_METERS = SEARCH_RADIUS_NM * 1852;

            // Create main panel
            const panel = document.createElement('div');
            panel.id = 'multi-channel-panel';
            panel.style.cssText = `
                position: fixed;
                top: 80px;
                right: 20px;
                background: rgba(20, 20, 20, 0.95);
                color: white;
                padding: 20px;
                border-radius: 8px;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                z-index: 10000;
                box-shadow: 0 4px 20px rgba(0,0,0,0.5);
                border: 2px solid rgba(255, 255, 255, 0.2);
                width: 300px;
                max-height: 600px;
                overflow-y: auto;
            `;

            panel.innerHTML = `
                <div class="panel-header drag-handle">
                    <h3 class="panel-title">
                        <span>📡</span> Multi-Channel ATC
                    </h3>
                    <div class="panel-controls">
                        <button class="panel-btn reset-btn no-drag" onclick="resetPanelPosition('multi-channel-panel')" title="Reset Position">↺</button>
                    </div>
                </div>

                <div style="margin-bottom: 15px; padding: 10px; background: rgba(255,255,255,0.1); border-radius: 5px;">
                    <div style="font-size: 12px; margin-bottom: 5px;">
                        <strong>Status:</strong> 
                        <span id="monitor-status" style="color: #00ff00;">● Active</span>
                    </div>
                    <div style="font-size: 12px; margin-bottom: 5px;">
                        <strong>Area:</strong> PDX - {SEARCH_RADIUS_NM} NM
                    </div>
                    <div style="font-size: 12px; margin-bottom: 5px;">
                        <strong>Total Transmissions:</strong> 
                        <span id="total-transmissions">0</span>
                    </div>
                    <div style="font-size: 12px;">
                        <strong>Queue:</strong> 
                        <span id="queue-size">0</span> | 
                        <strong>Workers:</strong> 
                        <span id="workers-busy">0</span>/{self.num_workers}
                    </div>
                </div>

                <h4 style="margin: 0 0 10px 0; font-size: 14px;">Channels ({len(self.atc_monitor.channel_configs)})</h4>
                <div id="channel-list">
                    {channels_html}
                </div>

                <h4 style="margin: 15px 0 10px 0; font-size: 14px;">Workers</h4>
                <div id="worker-status" style="display: flex; gap: 10px; flex-wrap: wrap;">
                    {"".join(f'<div id="worker-{i}" class="worker-box" style="flex: 1; min-width: 60px; padding: 5px; background: rgba(0,255,0,0.2); border-radius: 3px; text-align: center; font-size: 11px;">Worker {i}<br><span style="color: #888;">Idle</span></div>' for i in range(self.num_workers))}
                </div>
            `;

            document.body.appendChild(panel);

            // Make the panel draggable by its header
            if (window.makeDraggable) {{
                window.makeDraggable(panel, '.panel-header');
            }}

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
                    btnEl.textContent = 'Mute';
                }} else {{
                    audioEl.pause();
                    audioEl.removeAttribute('src');
                    audioEl.load();
                    btnEl.textContent = 'Unmute';
                }}
            }};

            // Create transcript display
            const transcriptContainer = document.createElement('div');
            transcriptContainer.id = 'multi-transcript-container';
            transcriptContainer.style.cssText = `
                position: fixed;
                bottom: 20px;
                left: 20px;
                width: 600px;
                max-width: calc(100vw - 380px);
                background: rgba(0, 0, 0, 0.9);
                color: white;
                padding: 20px;
                border-radius: 8px;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                z-index: 9999;
                box-shadow: 0 4px 20px rgba(0,0,0,0.5);
                border: 1px solid rgba(255, 255, 255, 0.1);
                max-height: 300px;
                overflow-y: auto;
            `;

            transcriptContainer.innerHTML = `
                <div class="panel-header drag-handle">
                    <h3 class="panel-title">
                        <span>📝</span> Transcripts
                    </h3>
                    <div class="panel-controls">
                        <button class="panel-btn reset-btn no-drag" onclick="resetPanelPosition('multi-transcript-container')" title="Reset Position">↺</button>
                    </div>
                </div>
                <div id="transcript-content" style="margin-top: 10px;">
                    <div style="text-align: center; opacity: 0.5;">Waiting for transmissions...</div>
                </div>
            `;

            document.body.appendChild(transcriptContainer);

            // Make transcript container draggable
            if (window.makeDraggable) {{
                window.makeDraggable(transcriptContainer, '.panel-header');
            }}

            // Add custom scrollbar styles and animations
            const style = document.createElement('style');
            style.textContent = `
                #multi-channel-panel::-webkit-scrollbar,
                #multi-transcript-container::-webkit-scrollbar {{
                    width: 6px;
                }}

                #multi-channel-panel::-webkit-scrollbar-track,
                #multi-transcript-container::-webkit-scrollbar-track {{
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 3px;
                }}

                #multi-channel-panel::-webkit-scrollbar-thumb,
                #multi-transcript-container::-webkit-scrollbar-thumb {{
                    background: rgba(255, 255, 255, 0.3);
                    border-radius: 3px;
                }}

                .worker-box {{
                    transition: all 0.3s ease;
                }}

                .mute-btn {{
                    margin-left: 8px;
                    padding: 1px 6px;
                    font-size: 11px;
                    cursor: pointer;
                }}

                @keyframes pulse {{
                    0% {{ opacity: 1; }}
                    50% {{ opacity: 0.7; }}
                    100% {{ opacity: 1; }}
                }}

                @keyframes fadeInUp {{
                    from {{
                        opacity: 0;
                        transform: translateY(20px);
                    }}
                    to {{
                        opacity: 1;
                        transform: translateY(0);
                    }}
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
                    // Try to find through the map container
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
                        // Create overlay canvas for the monitoring circle
                        const mapContainer = document.querySelector('#map_container');
                        if (!mapContainer) {{
                            console.error('[ATC] Map container not found');
                            return;
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
                        }}

                        // Function to draw the monitoring circle
                        function drawMonitoringCircle() {{
                            const canvas = document.getElementById('atc-overlay-canvas');
                            if (!canvas || !map) return;

                            canvas.width = canvas.offsetWidth;
                            canvas.height = canvas.offsetHeight;

                            const ctx = canvas.getContext('2d');
                            ctx.clearRect(0, 0, canvas.width, canvas.height);

                            // Get center coordinates in pixels
                            const centerCoords = ol.proj.fromLonLat([AIRPORT_LON, AIRPORT_LAT]);
                            const centerPixel = map.getPixelFromCoordinate(centerCoords);

                            if (!centerPixel) return;

                            // Calculate radius in pixels
                            const resolution = map.getView().getResolution();
                            const radiusPixels = RADIUS_METERS / resolution;

                            // Draw the circle
                            ctx.strokeStyle = 'rgba(255, 0, 0, 0.8)';
                            ctx.lineWidth = 3;
                            ctx.setLineDash([10, 10]);

                            ctx.beginPath();
                            ctx.arc(centerPixel[0], centerPixel[1], radiusPixels, 0, 2 * Math.PI);
                            ctx.stroke();

                            // Fill with semi-transparent red
                            ctx.fillStyle = 'rgba(255, 0, 0, 0.05)';
                            ctx.fill();
                        }}

                        // Draw initially
                        drawMonitoringCircle();

                        // Redraw on map events
                        map.on('moveend', drawMonitoringCircle);
                        map.on('postrender', drawMonitoringCircle);
                        map.getView().on('change:resolution', drawMonitoringCircle);
                        map.getView().on('change:center', drawMonitoringCircle);

                        overlayInitialized = true;
                        console.log('[ATC] Monitoring radius overlay initialized!');

                        // Store references
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

            tryInitOverlay();

            return true;
        }})();
        """

        try:
            result = self.window.evaluate_js(injection_js)
            if result:
                self.overlay_initialized = True
                success("Multi-channel monitor interface injected")
        except Exception as e:
            error(f"Error injecting monitor: {e}")

    def _inject_single_channel_monitor(self):
        """Inject single-channel monitoring interface"""
        injection_js = f"""
        (function() {{
            if (window.singleChannelMonitorInjected) {{
                return true;
            }}
            window.singleChannelMonitorInjected = true;

            console.log('[ATC] Injecting single-channel monitor...');

            // Configuration
            const AIRPORT_LAT = {AIRPORT_LAT};
            const AIRPORT_LON = {AIRPORT_LON};
            const SEARCH_RADIUS_NM = {SEARCH_RADIUS_NM};

            // Create info panel
            const infoPanel = document.createElement('div');
            infoPanel.id = 'atc-info-panel';
            infoPanel.style.cssText = `
                position: fixed;
                top: 80px;
                right: 20px;
                background: rgba(20, 20, 20, 0.95);
                color: white;
                padding: 20px;
                border-radius: 8px;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                z-index: 10000;
                box-shadow: 0 4px 20px rgba(0,0,0,0.5);
                border: 2px solid rgba(255, 255, 255, 0.2);
                width: 280px;
            `;

            infoPanel.innerHTML = `
                <div class="panel-header drag-handle">
                    <h3 class="panel-title">
                        <span>📡</span> ATC Monitor
                    </h3>
                    <div class="panel-controls">
                        <button class="panel-btn reset-btn no-drag" onclick="resetPanelPosition('atc-info-panel')" title="Reset Position">↺</button>
                    </div>
                </div>

                <div style="margin-bottom: 15px; padding: 10px; background: rgba(255,255,255,0.1); border-radius: 5px;">
                    <div style="font-size: 12px; margin-bottom: 5px;">
                        <strong>Status:</strong> 
                        <span id="atc-status" style="color: #00ff00;">● Active</span>
                    </div>
                    <div style="font-size: 12px; margin-bottom: 5px;">
                        <strong>Area:</strong> PDX - {SEARCH_RADIUS_NM} NM
                    </div>
                    <div style="font-size: 12px;">
                        <strong>Transmissions:</strong> 
                        <span id="atc-transmission-count">0</span>
                    </div>
                </div>
            `;

            document.body.appendChild(infoPanel);

            // Make info panel draggable
            if (window.makeDraggable) {{
                window.makeDraggable(infoPanel, '.panel-header');
            }}

            // Create transcript container
            const transcriptContainer = document.createElement('div');
            transcriptContainer.id = 'atc-transcript-container';
            transcriptContainer.style.cssText = `
                position: fixed;
                bottom: 20px;
                left: 20px;
                width: 600px;
                max-width: calc(100vw - 340px);
                background: rgba(0, 0, 0, 0.9);
                color: white;
                padding: 20px;
                border-radius: 8px;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                z-index: 9999;
                box-shadow: 0 4px 20px rgba(0,0,0,0.5);
                border: 1px solid rgba(255, 255, 255, 0.1);
                max-height: 250px;
                overflow-y: auto;
                display: none;
            `;

            transcriptContainer.innerHTML = `
                <div class="panel-header drag-handle">
                    <h3 class="panel-title">
                        <span>📝</span> Transcripts
                    </h3>
                    <div class="panel-controls">
                        <button class="panel-btn reset-btn no-drag" onclick="resetPanelPosition('atc-transcript-container')" title="Reset Position">↺</button>
                    </div>
                </div>
                <div id="transcript-content" style="margin-top: 10px;">
                    <div style="text-align: center; opacity: 0.5;">Waiting for transmissions...</div>
                </div>
            `;

            document.body.appendChild(transcriptContainer);

            // Make transcript container draggable
            if (window.makeDraggable) {{
                window.makeDraggable(transcriptContainer, '.panel-header');
            }}

            // Add animation styles
            const style = document.createElement('style');
            style.textContent = `
                @keyframes fadeInUp {{
                    from {{
                        opacity: 0;
                        transform: translateY(20px);
                    }}
                    to {{
                        opacity: 1;
                        transform: translateY(0);
                    }}
                }}

                #atc-transcript-container::-webkit-scrollbar {{
                    width: 6px;
                }}

                #atc-transcript-container::-webkit-scrollbar-track {{
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 3px;
                }}

                #atc-transcript-container::-webkit-scrollbar-thumb {{
                    background: rgba(255, 255, 255, 0.3);
                    border-radius: 3px;
                }}
            `;
            document.head.appendChild(style);

            return true;
        }})();
        """

        try:
            result = self.window.evaluate_js(injection_js)
            if result:
                self.overlay_initialized = True
                success("Single-channel monitor interface injected")
        except Exception as e:
            error(f"Error injecting monitor: {e}")

    def process_updates(self):
        """Process updates from the monitor"""
        while self.running:
            try:
                message = self.update_queue.get(timeout=0.1)
                command = message[0]
                data = message[1]

                if command == "atc_transmission":
                    if self.is_multi_channel:
                        self.add_multi_channel_transmission(data)
                    else:
                        self.add_transmission(data)
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
                continue
            except Exception as e:
                error(f"Error processing update: {e}")

    def add_multi_channel_transmission(self, data):
        """Add transmission for multi-channel mode"""
        if not self.window or not self.overlay_initialized:
            return

        # Update channel counter
        freq = data.get('frequency', '')
        if freq in self.channel_counters:
            self.channel_counters[freq] += 1

        # Add to transcript buffer
        self.displayed_transcripts.append(data)
        if len(self.displayed_transcripts) > self.max_displayed_transcripts:
            self.displayed_transcripts.pop(0)

        # Build transcript HTML
        transcript_html = ""
        for trans in self.displayed_transcripts:
            color = trans.get('color', '#FFFFFF')
            timestamp = trans.get('timestamp', datetime.now().isoformat())
            time_str = timestamp.split('T')[1][:8] if 'T' in timestamp else timestamp

            # Escape any special characters in transcript text
            transcript_text = trans.get('transcript', '').replace('`', '\\`').replace('$', '\\$')

            transcript_html += f"""
            <div style="margin-bottom: 10px; padding: 8px; border-left: 3px solid {color}; background: rgba(255,255,255,0.05); animation: fadeInUp 0.5s ease-out;">
                <div style="font-size: 11px; color: #888; margin-bottom: 3px;">
                    [{time_str}] 
                    <span style="color: {color}; font-weight: bold;">{trans.get('channel', 'Unknown')}</span>
                    <span style="float: right;">Worker {trans.get('worker_id', '?')}</span>
                </div>
                <div style="font-size: 13px; color: #fff;">
                    {transcript_text[:200]}{'...' if len(transcript_text) > 200 else ''}
                </div>
            </div>
            """

        js_code = f"""
        (function() {{
            // Update channel counter
            const freq = '{freq.replace('.', '_')}';
            const counterEl = document.getElementById('channel-count-' + freq);
            if (counterEl) {{
                counterEl.textContent = {self.channel_counters.get(freq, 0)};
            }}

            // Update total counter
            const totalEl = document.getElementById('total-transmissions');
            if (totalEl) {{
                totalEl.textContent = {sum(self.channel_counters.values())};
            }}

            // Update transcript display
            const contentEl = document.getElementById('transcript-content');
            if (contentEl) {{
                contentEl.innerHTML = `{transcript_html}`;
            }}

            // Scroll to bottom
            const container = document.getElementById('multi-transcript-container');
            if (container) {{
                container.scrollTop = container.scrollHeight;
            }}
        }})();
        """

        self.window.evaluate_js(js_code)

        info(
            f"[{data.get('channel', 'Unknown')}] Transmission #{sum(self.channel_counters.values())}: {data.get('transcript', '')[:50]}...")

    def add_transmission(self, data):
        """Add transmission for single-channel mode"""
        if not self.window or not self.overlay_initialized:
            return

        self.transmission_count += 1
        transcript = data.get('transcript', 'No transcript')
        timestamp = datetime.now().strftime('%H:%M:%S')

        # Update counter
        js_update_counter = f"""
        (function() {{
            const el = document.getElementById('atc-transmission-count');
            if (el) {{
                el.textContent = {self.transmission_count};
            }}
            return true;
        }})();
        """
        self.window.evaluate_js(js_update_counter)

        # Add to displayed transcripts
        self.displayed_transcripts.append({
            'id': f'transcript-{self.transmission_count}',
            'text': transcript,
            'timestamp': timestamp
        })

        if len(self.displayed_transcripts) > self.max_displayed_transcripts:
            self.displayed_transcripts.pop(0)

        self.update_transcript_display()
        info(f"Transmission #{self.transmission_count}: {transcript[:50]}...")

    def update_transcript_display(self):
        """Update transcript display for single-channel mode"""
        if not self.window or self.is_multi_channel:
            return

        # Show container on first transcript
        if len(self.displayed_transcripts) == 1:
            show_container_js = """
            (function() {
                const container = document.getElementById('atc-transcript-container');
                if (container) {
                    container.style.display = 'block';
                }
            })();
            """
            self.window.evaluate_js(show_container_js)

        # Build HTML for transcripts
        transcript_html = ""
        for i, trans in enumerate(self.displayed_transcripts):
            opacity = 0.4 + (0.6 * (i + 1) / len(self.displayed_transcripts))
            text = trans['text'].replace('`', '\\`').replace('$', '\\$')
            transcript_html += f"""
            <div class="transcript-item" style="opacity: {opacity}; margin-bottom: 8px; animation: fadeInUp 0.3s ease-out;">
                <span style="color: #888; font-size: 11px;">[{trans['timestamp']}]</span>
                <span style="color: #fff; font-size: 13px;">{text[:150]}{'...' if len(text) > 150 else ''}</span>
            </div>
            """

        js_code = f"""
        (function() {{
            const contentEl = document.getElementById('transcript-content');
            if (contentEl) {{
                contentEl.innerHTML = `{transcript_html}`;
            }}

            const container = document.getElementById('atc-transcript-container');
            if (container) {{
                container.scrollTop = container.scrollHeight;
            }}
            return true;
        }})();
        """

        self.window.evaluate_js(js_code)

    def update_aircraft(self, data):
        """Update aircraft position (stub for now)"""
        pass

    def flash_channel(self, frequency):
        """Flash channel indicator when recording"""
        if not self.window or not self.is_multi_channel:
            return

        freq_id = frequency.replace('.', '_')
        js_code = f"""
        (function() {{
            const channelEl = document.querySelector('#channel-count-{freq_id}');
            if (channelEl) {{
                const parentEl = channelEl.parentElement.parentElement;
                parentEl.style.background = 'rgba(255,255,255,0.2)';
                setTimeout(() => {{
                    parentEl.style.background = '';
                }}, 300);
            }}
        }})();
        """

        self.window.evaluate_js(js_code)

    def update_worker_status(self, data):
        """Update worker status display"""
        if not self.window or not self.is_multi_channel:
            return

        worker_id = data.get('worker_id', 0)
        status = data.get('status', 'idle')
        color = '#00ff00' if status == 'idle' else '#ff9900'
        bg_color = 'rgba(0,255,0,0.2)' if status == 'idle' else 'rgba(255,153,0,0.3)'
        channel = data.get('channel', '').replace("'", "\\'")
        status_text = 'Idle' if status == 'idle' else f"Processing<br>{channel}"

        js_code = f"""
        (function() {{
            const worker = document.getElementById('worker-{worker_id}');
            if (worker) {{
                worker.style.background = '{bg_color}';
                worker.innerHTML = 'Worker {worker_id}<br><span style="color: {color};">{status_text}</span>';
                if ('{status}' === 'busy') {{
                    worker.style.animation = 'pulse 1s infinite';
                }} else {{
                    worker.style.animation = '';
                }}
            }}
        }})();
        """

        self.window.evaluate_js(js_code)

    def update_statistics(self, stats):
        """Update statistics display"""
        if not self.window or not self.is_multi_channel:
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

        self.window.evaluate_js(js_code)

    def show_recording_status(self, data):
        """Show recording status in GUI"""
        if not self.window:
            return

        recording_num = data.get('recording_number', 0)
        js_code = f"""
        (function() {{
            const statusEl = document.getElementById('atc-status') || document.getElementById('monitor-status');
            if (statusEl) {{
                statusEl.style.color = '#ff9900';
                statusEl.textContent = '● Recording #{recording_num}';

                setTimeout(() => {{
                    statusEl.style.color = '#00ff00';
                    statusEl.textContent = '● Active';
                }}, 2000);
            }}
            return true;
        }})();
        """

        self.window.evaluate_js(js_code)

    def show_alert(self, data):
        """Show alert in GUI"""
        if not self.window:
            return

        alert_type = data.get('type', 'Unknown').replace("'", "\\'")
        alert_text = data.get('transcript', '')[:100].replace("'", "\\'").replace("`", "\\`")

        js_code = f"""
        (function() {{
            const alert = document.createElement('div');
            alert.style.cssText = `
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: rgba(255, 0, 0, 0.95);
                color: white;
                padding: 20px 30px;
                border-radius: 12px;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                z-index: 10001;
                box-shadow: 0 8px 32px rgba(255,0,0,0.5);
                border: 2px solid #ff0000;
                max-width: 500px;
                animation: pulse 1s infinite;
            `;

            alert.innerHTML = `
                <h4 style="margin: 0 0 10px 0; font-size: 18px;">⚠️ ALERT: {alert_type.upper()}</h4>
                <p style="margin: 0; font-size: 14px;">{alert_text}</p>
            `;

            document.body.appendChild(alert);

            setTimeout(() => {{
                alert.style.transition = 'opacity 0.5s';
                alert.style.opacity = '0';
                setTimeout(() => alert.remove(), 500);
            }}, 10000);

            return true;
        }})();
        """

        self.window.evaluate_js(js_code)

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