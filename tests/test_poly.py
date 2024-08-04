from test_utils import set_ml_pypath
set_ml_pypath()
from get_poly import  *
import pytest
import requests_mock

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