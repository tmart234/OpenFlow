from utils.get_poly import get_huc8_polygon, simplify_polygon, main  # type: ignore
import pytest
import requests_mock
from shapely.geometry import Polygon
from shapely.validation import make_valid


@pytest.fixture
def mock_response():
    return {
        "features": [
            {
                "geometry": {
                    "rings": [
                        [[0, 0], [1, 0], [1, 1], [0, 1]]
                    ]
                }
            }
        ]
    }

def test_get_huc8_polygon(mock_response):
    with requests_mock.Mocker() as m:
        m.get("https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/4/query", json=mock_response)
        polygon = get_huc8_polygon(37.7749, -122.4194)
        assert polygon == [[0, 0], [1, 0], [1, 1], [0, 1]]

def test_get_huc8_polygon_error():
    with requests_mock.Mocker() as m:
        m.get("https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/4/query", status_code=500)
        polygon = get_huc8_polygon(37.7749, -122.4194)
        assert polygon is None

def test_simplify_polygon(mock_response):
    with requests_mock.Mocker() as m:
        m.get("https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/4/query", json=mock_response)
        polygon = get_huc8_polygon(37.7749, -122.4194)
        simplified = simplify_polygon(polygon)
        assert len(simplified) <= 100

def test_main(mock_response):
    with requests_mock.Mocker() as m:
        m.get("https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/4/query", json=mock_response)
        simplified_polygon = main(37.7749, -122.4194)
        assert simplified_polygon == [[0, 0], [1, 0], [1, 1], [0, 1]]
def calculate_iou(polygon1, polygon2):
    """Calculate the Intersection over Union (IoU) of two polygons."""
    poly1 = Polygon(polygon1)
    poly2 = Polygon(polygon2)
    
    # Check if polygons are valid and try to fix them if they're not
    if not poly1.is_valid:
        poly1 = make_valid(poly1)
    if not poly2.is_valid:
        poly2 = make_valid(poly2)
    
    # If polygons are still invalid, return 0
    if not poly1.is_valid or not poly2.is_valid:
        print("Warning: Unable to fix invalid polygons")
        return 0
    
    try:
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        return intersection / union if union > 0 else 0
    except Exception as e:
        print(f"Error calculating IoU: {str(e)}")
        return 0

@pytest.fixture
def mock_colorado_response():
    # This is a mock response for a Colorado HUC8 polygon
    return {
        "features": [
            {
                "geometry": {
                    "rings": [
                        [
                            [-104.9903, 39.7392],
                            [-104.9903, 39.8392],
                            [-104.8903, 39.8392],
                            [-104.8903, 39.7392],
                            [-104.9903, 39.7392]
                        ] + [[-104.9903 + 0.01*i, 39.7392 + 0.01*i] for i in range(1, 200)]
                    ]
                }
            }
        ]
    }

def test_simplify_colorado_huc8_polygon(mock_colorado_response):
    with requests_mock.Mocker() as m:
        m.get("https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/4/query", json=mock_colorado_response)
        
        # Coordinates for a point in Colorado (Denver area)
        lat, lon = 39.7392, -104.9903

        # Get the full HUC8 polygon
        original_polygon = get_huc8_polygon(lat, lon)
        assert original_polygon is not None, "Failed to fetch HUC8 polygon"

        # Simplify the polygon
        simplified_polygon = simplify_polygon(original_polygon)

        # Check polygon validity
        original_poly = Polygon(original_polygon)
        simplified_poly = Polygon(simplified_polygon)
        
        print(f"Original polygon valid: {original_poly.is_valid}")
        print(f"Simplified polygon valid: {simplified_poly.is_valid}")

        # Calculate IoU
        iou = calculate_iou(original_polygon, simplified_polygon)

        # Check that the simplified polygon retains at least 92% of the original shape
        assert iou >= 0.92, f"Simplified polygon retains less than 92% of the original shape. IoU: {iou:.2%}"

        # Print detailed information about the simplification
        print(f"Original polygon points: {len(original_polygon)}")
        print(f"Simplified polygon points: {len(simplified_polygon)}")
        print(f"Shape similarity (IoU): {iou:.2%}")
        print(f"Point reduction: {(1 - len(simplified_polygon) / len(original_polygon)):.2%}")

if __name__ == "__main__":
    pytest.main([__file__])