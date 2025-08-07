# test_openlayers_overlay.py
import webview
import time
from utils.config import AIRPORT_LAT, AIRPORT_LON


def test_openlayers_overlay():
    """Test OpenLayers overlay with a simple circle"""

    def on_loaded():
        print("Page loaded, waiting for map initialization...")
        time.sleep(5)  # Give the map more time to load

        # First, let's find the map instance
        map_search = window.evaluate_js("""
            // Search for the OpenLayers map instance
            let mapInstance = null;
            let searchResults = {
                foundMap: false,
                method: 'none',
                mapInfo: {}
            };

            // Method 1: Check for global map variable
            if (window.map && window.map.addLayer) {
                mapInstance = window.map;
                searchResults.method = 'window.map';
                searchResults.foundMap = true;
            }

            // Method 2: Check for OL map in various places
            if (!mapInstance && window.ol) {
                // Look for map in common variable names
                const possibleNames = ['map', 'olMap', 'OLMap', 'Map', 'osMap'];
                for (let name of possibleNames) {
                    if (window[name] && window[name].addLayer) {
                        mapInstance = window[name];
                        searchResults.method = 'window.' + name;
                        searchResults.foundMap = true;
                        break;
                    }
                }
            }

            // Method 3: Look for map in DOM elements
            if (!mapInstance) {
                const mapDivs = document.querySelectorAll('.ol-viewport');
                if (mapDivs.length > 0) {
                    // Check if parent has map reference
                    const mapContainer = mapDivs[0].parentElement;
                    if (mapContainer && mapContainer._map) {
                        mapInstance = mapContainer._map;
                        searchResults.method = 'DOM element._map';
                        searchResults.foundMap = true;
                    }
                }
            }

            // Method 4: Check all window properties for OpenLayers map
            if (!mapInstance) {
                for (let key in window) {
                    try {
                        if (window[key] && window[key].constructor && 
                            window[key].constructor.name === 'Map' && 
                            window[key].addLayer) {
                            mapInstance = window[key];
                            searchResults.method = 'window.' + key;
                            searchResults.foundMap = true;
                            break;
                        }
                    } catch (e) {}
                }
            }

            if (mapInstance) {
                searchResults.mapInfo = {
                    hasAddLayer: typeof mapInstance.addLayer === 'function',
                    hasGetView: typeof mapInstance.getView === 'function',
                    viewportFound: document.querySelector('.ol-viewport') !== null
                };
            }

            window.testMapInstance = mapInstance;
            JSON.stringify(searchResults);
        """)

        print(f"Map search results: {map_search}")

        # Now try to add a simple circle overlay
        overlay_result = window.evaluate_js("""
            try {
                if (!window.testMapInstance) {
                    throw new Error('No map instance found');
                }

                const map = window.testMapInstance;

                // Create a circle feature
                const airportCoords = ol.proj.fromLonLat([""" + str(AIRPORT_LON) + """, """ + str(AIRPORT_LAT) + """]);

                // Create a circle geometry (30 NM radius = ~55.56 km)
                const circle = new ol.geom.Circle(airportCoords, 55560); // meters

                // Create a feature from the circle
                const circleFeature = new ol.Feature(circle);

                // Create a style for the circle
                const circleStyle = new ol.style.Style({
                    stroke: new ol.style.Stroke({
                        color: 'rgba(255, 0, 0, 0.8)',
                        width: 3
                    }),
                    fill: new ol.style.Fill({
                        color: 'rgba(255, 0, 0, 0.1)'
                    })
                });

                circleFeature.setStyle(circleStyle);

                // Create a vector source and layer
                const vectorSource = new ol.source.Vector({
                    features: [circleFeature]
                });

                const vectorLayer = new ol.layer.Vector({
                    source: vectorSource,
                    zIndex: 100
                });

                // Add the layer to the map
                map.addLayer(vectorLayer);

                // Store reference for later
                window.atcOverlayLayer = vectorLayer;

                'SUCCESS: Circle overlay added';
            } catch (error) {
                'ERROR: ' + error.message;
            }
        """)

        print(f"Overlay result: {overlay_result}")

        # Add a test panel
        panel_result = window.evaluate_js("""
            const panel = document.createElement('div');
            panel.style.cssText = `
                position: fixed;
                top: 100px;
                right: 20px;
                background: rgba(20, 20, 20, 0.95);
                color: white;
                padding: 20px;
                border-radius: 8px;
                width: 300px;
                font-family: Arial, sans-serif;
                z-index: 10000;
                box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            `;
            panel.innerHTML = `
                <h3 style="margin: 0 0 10px 0;">✈️ ATC Monitor</h3>
                <p>Airport: PDX</p>
                <p>Monitoring Radius: 30 NM</p>
                <p style="color: #00ff00;">● System Active</p>
            `;
            document.body.appendChild(panel);
            'Panel added';
        """)

        print(f"Panel result: {panel_result}")

        # Try the lazy approach - wait and check globals
        time.sleep(2)

        lazy_check = window.evaluate_js("""
            const results = [];

            // Check for any global map variable
            for (let key of Object.keys(window)) {
                if (key.toLowerCase().includes('map') && window[key] && typeof window[key] === 'object') {
                    results.push({
                        name: key,
                        type: window[key].constructor ? window[key].constructor.name : 'unknown',
                        hasAddLayer: typeof window[key].addLayer === 'function'
                    });
                }
            }

            JSON.stringify(results);
        """)

        print(f"Lazy check results: {lazy_check}")

    window = webview.create_window(
        'OpenLayers Overlay Test',
        f'https://map.opensky-network.org/?lat={AIRPORT_LAT}&lon={AIRPORT_LON}&zoom=10',
        width=1400,
        height=900
    )

    webview.start(on_loaded, debug=True)


if __name__ == "__main__":
    test_openlayers_overlay()