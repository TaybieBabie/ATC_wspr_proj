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
        """Add transmission to map"""
        if not self.window or not self.overlay_initialized:
            return

        # Update transmission count
        js_code = """
        (function() {
            const el = document.getElementById('atc-transmission-count');
            if (el) {
                const current = parseInt(el.textContent) || 0;
                el.textContent = current + 1;
            }
            return true;
        })();
        """

        self.window.evaluate_js(js_code)

        self.transmission_count += 1
        info(f"Transmission #{self.transmission_count}: {data.get('transcript', 'No transcript')[:50]}...")

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