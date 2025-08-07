# test_circle_overlay.py
import webview
from utils.config import AIRPORT_LAT, AIRPORT_LON, SEARCH_RADIUS_NM


def test_circle_overlay():
    """Test just the circle overlay"""

    window = webview.create_window(
        'Circle Overlay Test',
        f'https://map.opensky-network.org/?lat={AIRPORT_LAT}&lon={AIRPORT_LON}&zoom=10',
        width=1400,
        height=900
    )

    def on_loaded():
        # Simplified circle drawing code
        js_code = f"""
        setTimeout(() => {{
            if (window.OLMap) {{
                const source = new ol.source.Vector();
                const layer = new ol.layer.Vector({{
                    source: source,
                    style: new ol.style.Style({{
                        stroke: new ol.style.Stroke({{
                            color: 'rgba(255, 0, 0, 0.8)',
                            width: 3,
                            lineDash: [10, 10]
                        }})
                    }})
                }});

                window.OLMap.addLayer(layer);

                const center = ol.proj.fromLonLat([{AIRPORT_LON}, {AIRPORT_LAT}]);
                const circle = new ol.geom.Circle(center, {SEARCH_RADIUS_NM * 1852});
                const feature = new ol.Feature(circle);
                source.addFeature(feature);

                console.log('Circle added!');
            }}
        }}, 3000);
        """
        window.evaluate_js(js_code)

    webview.start(on_loaded, debug=True)


if __name__ == "__main__":
    test_circle_overlay()