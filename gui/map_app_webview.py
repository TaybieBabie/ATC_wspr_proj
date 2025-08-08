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

        self.displayed_transcripts = []  # Track displayed transcripts
        self.max_displayed_transcripts = 5  # Show last 5 transcripts

    def run(self):
        """Run the application"""
        # Create window with the OpenSky map URL
        self.window = webview.create_window(
            'ATC Monitor - OpenSky Map',
            f'https://map.opensky-network.org/?lat={AIRPORT_LAT}&lon={AIRPORT_LON}&zoom=10',
            width=1400,
            height=900
        )

        # Set up event handlers
        self.window.events.loaded += self.on_page_loaded

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

    def inject_monitor(self):
        """Inject the monitoring code once"""
        if not self.running or self.overlay_initialized:
            return

        info("Injecting ATC monitor...")

        # Completely new approach - find the map through tar1090's structure
        injection_js = """
        // ATC Monitor Injection - tar1090 compatible
        (function() {
            // Prevent multiple injections
            if (window.atcMonitorInjected) {
                console.log('[ATC] Monitor already injected');
                return true;
            }
            window.atcMonitorInjected = true;

            console.log('[ATC] Starting ATC Monitor injection for tar1090...');

            // Configuration
            const AIRPORT_LAT = """ + str(AIRPORT_LAT) + """;
            const AIRPORT_LON = """ + str(AIRPORT_LON) + """;
            const SEARCH_RADIUS_NM = """ + str(SEARCH_RADIUS_NM) + """;
            const RADIUS_METERS = SEARCH_RADIUS_NM * 1852;

            let initAttempts = 0;
            let overlayInitialized = false;

            // Add info panel immediately
            function addInfoPanel() {
                if (document.getElementById('atc-monitor-panel')) {
                    return;
                }

                const panel = document.createElement('div');
                panel.id = 'atc-monitor-panel';
                panel.style.cssText = `
                    position: fixed;
                    top: 80px;
                    right: 20px;
                    background: rgba(20, 20, 20, 0.9);
                    color: white;
                    padding: 15px 20px;
                    border-radius: 8px;
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                    z-index: 10000;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
                    border: 2px solid rgba(255, 0, 0, 0.3);
                    min-width: 200px;
                `;

                panel.innerHTML = `
                    <h3 style="margin: 0 0 10px 0; font-size: 16px; font-weight: 600;">
                        ✈️ ATC Monitor
                    </h3>
                    <div style="font-size: 14px; line-height: 1.6;">
                        <div style="margin-bottom: 5px;">
                            <strong>Airport:</strong> PDX
                        </div>
                        <div style="margin-bottom: 5px;">
                            <strong>Radius:</strong> ${SEARCH_RADIUS_NM} NM
                        </div>
                        <div style="margin-bottom: 5px;">
                            <strong>Status:</strong> 
                            <span id="atc-status" style="color: #ffff00;">● Initializing...</span>
                        </div>
                        <div>
                            <strong>Transmissions:</strong> 
                            <span id="atc-transmission-count">0</span>
                        </div>
                    </div>
                `;

                document.body.appendChild(panel);
                console.log('[ATC] Info panel added');
            }

            function tryInitialize() {
                initAttempts++;
                console.log('[ATC] Initialization attempt #' + initAttempts);

                // Method 1: Look for tar1090's OLMap
                let map = null;

                // Check various possible locations
                if (typeof OLMap !== 'undefined') {
                    map = OLMap;
                    console.log('[ATC] Found map at global OLMap');
                } else if (window.OLMap) {
                    map = window.OLMap;
                    console.log('[ATC] Found map at window.OLMap');
                } else {
                    // Try to find through the map container
                    const mapCanvas = document.querySelector('#map_canvas');
                    if (mapCanvas && mapCanvas._olMap) {
                        map = mapCanvas._olMap;
                        console.log('[ATC] Found map through canvas element');
                    }
                }

                if (!map) {
                    console.log('[ATC] Map not found yet, will retry...');
                    if (initAttempts < 30) {
                        setTimeout(tryInitialize, 1000);
                    }
                    return;
                }

                console.log('[ATC] Map found! Creating overlay...');

                try {
                    // Create a simple canvas overlay instead of using OL geometries
                    const mapContainer = document.querySelector('#map_container');
                    if (!mapContainer) {
                        console.error('[ATC] Map container not found');
                        return;
                    }

                    // Create overlay canvas
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

                    // Add to map container
                    const mapCanvas = mapContainer.querySelector('#map_canvas');
                    if (mapCanvas) {
                        mapCanvas.appendChild(overlayCanvas);
                    }

                    // Function to draw the monitoring circle
                    function drawMonitoringCircle() {
                        const canvas = document.getElementById('atc-overlay-canvas');
                        if (!canvas || !map) return;

                        // Update canvas size
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
                    }

                    // Draw initially
                    drawMonitoringCircle();

                    // Redraw on map events
                    map.on('moveend', drawMonitoringCircle);
                    map.on('postrender', drawMonitoringCircle);
                    map.getView().on('change:resolution', drawMonitoringCircle);
                    map.getView().on('change:center', drawMonitoringCircle);

                    // Update status
                    const statusEl = document.getElementById('atc-status');
                    if (statusEl) {
                        statusEl.style.color = '#00ff00';
                        statusEl.textContent = '● Active';
                    }

                    overlayInitialized = true;
                    console.log('[ATC] Initialization complete!');

                    // Store references
                    window.atcOverlay = {
                        map: map,
                        canvas: overlayCanvas,
                        draw: drawMonitoringCircle,
                        initialized: true
                    };

                } catch (error) {
                    console.error('[ATC] Error during initialization:', error);
                    if (initAttempts < 30) {
                        setTimeout(tryInitialize, 1000);
                    }
                }
            }

            // Add info panel
            addInfoPanel();

            // Start initialization
            tryInitialize();

            // Status function
            window.atcGetStatus = function() {
                return {
                    initialized: overlayInitialized,
                    attempts: initAttempts,
                    hasMap: window.atcOverlay && window.atcOverlay.map ? true : false
                };
            };

            // Initialize empty transcript container
            const transcriptContainer = document.createElement('div');
            transcriptContainer.id = 'atc-transcript-container';
            transcriptContainer.style.cssText = `
                position: fixed;
                bottom: 20px;
                left: 20px;
                right: 20px;
                max-width: 800px;
                margin: 0 auto;
                background: rgba(0, 0, 0, 0.85);
                color: white;
                padding: 15px 20px;
                border-radius: 8px;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                z-index: 9999;
                box-shadow: 0 4px 20px rgba(0,0,0,0.5);
                border: 1px solid rgba(255, 255, 255, 0.1);
                display: none; // Hidden until first transcript
            `;
            transcriptContainer.innerHTML = '<div style="text-align: center; opacity: 0.5;">Waiting for transmissions...</div>';
            document.body.appendChild(transcriptContainer);
    
            return true;
        })();
        """

        try:
            result = self.window.evaluate_js(injection_js)
            if result:
                success("Monitor injection script executed")
                # Check status after a delay
                threading.Timer(2.0, self.check_initialization_status).start()
            else:
                error("Failed to execute injection script")
        except Exception as e:
            error(f"Error injecting monitor: {e}")

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
                    # Check again later
                    if status.get('attempts', 0) < 30:
                        threading.Timer(2.0, self.check_initialization_status).start()
                    else:
                        error("Failed to initialize after maximum attempts")
            else:
                warning("Could not get initialization status")

        except Exception as e:
            error(f"Error checking status: {e}")

    def process_updates(self):
        """Process updates from the monitor"""
        while self.running:
            try:
                message = self.update_queue.get(timeout=0.1)
                command = message[0]
                data = message[1]

                if command == "atc_transmission":
                    self.add_transmission(data)

            except queue.Empty:
                continue
            except Exception as e:
                error(f"Error processing update: {e}")

    def add_transmission(self, data):
        """Add transmission to map and display transcript"""
        if not self.window or not self.overlay_initialized:
            return

        # Update transmission count
        self.transmission_count += 1
        transcript = data.get('transcript', 'No transcript')
        timestamp = datetime.now().strftime('%H:%M:%S')

        # Update counter in info panel
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

        # Add transcript to display
        self.displayed_transcripts.append({
            'id': f'transcript-{self.transmission_count}',
            'text': transcript,
            'timestamp': timestamp
        })

        # Keep only the last N transcripts
        if len(self.displayed_transcripts) > self.max_displayed_transcripts:
            self.displayed_transcripts.pop(0)

        # Update transcript display
        self.update_transcript_display()

        info(f"Transmission #{self.transmission_count}: {transcript[:50]}...")

    def update_transcript_display(self):
        """Update the transcript display overlay"""
        if not self.window:
            return

        # Show container if it was hidden
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

        # Build HTML for all transcripts
        transcript_html = ""
        for i, trans in enumerate(self.displayed_transcripts):
            opacity = 0.4 + (0.6 * (i + 1) / len(self.displayed_transcripts))  # Older = more transparent
            transcript_html += f"""
            <div class="transcript-item" style="opacity: {opacity}; margin-bottom: 8px;">
                <span style="color: #888; font-size: 11px;">[{trans['timestamp']}]</span>
                <span style="color: #fff; font-size: 13px;">{trans['text'][:150]}{'...' if len(trans['text']) > 150 else ''}</span>
            </div>
            """

        js_code = f"""
        (function() {{
            // Create or update transcript container
            let container = document.getElementById('atc-transcript-container');
            if (!container) {{
                container = document.createElement('div');
                container.id = 'atc-transcript-container';
                container.style.cssText = `
                    position: fixed;
                    bottom: 20px;
                    left: 20px;
                    right: 20px;
                    max-width: 800px;
                    margin: 0 auto;
                    background: rgba(0, 0, 0, 0.85);
                    color: white;
                    padding: 15px 20px;
                    border-radius: 8px;
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                    z-index: 9999;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    max-height: 200px;
                    overflow-y: auto;
                `;
                document.body.appendChild(container);
            }}

            // Update content with animation for new items
            const newContent = `{transcript_html}`;
            const isNewTransmission = container.innerHTML !== newContent;

            container.innerHTML = newContent;

            // Animate new transmission
            if (isNewTransmission && container.lastElementChild) {{
                container.lastElementChild.style.animation = 'fadeInUp 0.5s ease-out';
            }}

            // Add animation styles if not already present
            if (!document.getElementById('atc-animations')) {{
                const style = document.createElement('style');
                style.id = 'atc-animations';
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

                    #atc-transcript-container::-webkit-scrollbar-thumb:hover {{
                        background: rgba(255, 255, 255, 0.5);
                    }}
                `;
                document.head.appendChild(style);
            }}

            // Auto-scroll to bottom
            container.scrollTop = container.scrollHeight;

            return true;
        }})();
        """

        try:
            self.window.evaluate_js(js_code)
        except Exception as e:
            error(f"Error updating transcript display: {e}")

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