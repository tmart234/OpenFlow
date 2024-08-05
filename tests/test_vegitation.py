from test_utils import set_ml_pypath
set_ml_pypath()
import pytest
import requests_mock
from get_vegdri import get_vegdri_data
import requests
import json
from get_poly import simplify_polygon, get_huc8_polygon

@pytest.fixture
def mock_response():
    with requests_mock.Mocker() as m:
        m.get('https://vegdri.cr.usgs.gov/api/v1/data', json={'test': 'data'})
        yield m

def test_missing_params():
    assert get_vegdri_data(None, None) is None

def test_api_error(mock_response):
    mock_response.get('https://vegdri.cr.usgs.gov/api/v1/data', status_code=500)
    assert get_vegdri_data("40.7128,-74.0060", '2022-07-30') is None

def test_json_decode_error(mock_response):
    mock_response.get('https://vegdri.cr.usgs.gov/api/v1/data', text='Invalid JSON')
    assert get_vegdri_data("40.7128,-74.0060", '2022-07-30') is None

def test_success(mock_response):
    assert get_vegdri_data("40.7128,-74.0060", '2022-07-30') == {'test': 'data'}

def test_invalid_date_format(mock_response):
    assert get_vegdri_data("40.7128,-74.0060", '07/30/2022') is None

def test_out_of_range_latitude(mock_response):
    assert get_vegdri_data("100,-74.0060", '2022-07-30') is None

def test_out_of_range_longitude(mock_response):
    assert get_vegdri_data("40.7128,200", '2022-07-30') is None

def test_non_numeric_latitude(mock_response):
    assert get_vegdri_data('abc,-74.0060', '2022-07-30') is None

def test_non_numeric_longitude(mock_response):
    assert get_vegdri_data("40.7128,abc", '2022-07-30') is None

def test_missing_date(mock_response):
    assert get_vegdri_data("40.7128,-74.0060", None) is None

def test_empty_response(mock_response):
    mock_response.get('https://vegdri.cr.usgs.gov/api/v1/data', text='')
    assert get_vegdri_data("40.7128,-74.0060", '2022-07-30') is None

def test_connection_error(mock_response):
    mock_response.get('https://vegdri.cr.usgs.gov/api/v1/data', exc=requests.exceptions.RequestException)
    assert get_vegdri_data("40.7128,-74.0060", '2022-07-30') is None

def test_polygon_success(mock_response):
    polygon = '{"type": "Polygon", "coordinates": [[-100, 40], [-100, 45], [-90, 45], [-90, 40], [-100, 40]]}'
    assert get_vegdri_data(polygon, '2022-07-30') == {'test': 'data'}

def test_invalid_polygon(mock_response):
    polygon = '{"type": "Point", "coordinates": [40, -100]}'
    assert get_vegdri_data(polygon, '2022-07-30') is None

def test_polygon_missing_type(mock_response):
    polygon = '{"coordinates": [[-100, 40], [-100, 45], [-90, 45], [-90, 40], [-100, 40]]}'
    assert get_vegdri_data(polygon, '2022-07-30') is None

def test_polygon_missing_coordinates(mock_response):
    polygon = '{"type": "Polygon"}'
    assert get_vegdri_data(polygon, '2022-07-30') is None

def test_polygon_invalid_coordinates(mock_response):
    polygon = '{"type": "Polygon", "coordinates": "abc"}'
    assert get_vegdri_data(polygon, '2022-07-30') is None

def test_polygon_invalid_coordinates_list(mock_response):
    polygon = '{"type": "Polygon", "coordinates": [[-100, 40], [-100, 45], [-90, 45], [-90, 40], "abc"]]}'
    assert get_vegdri_data(polygon, '2022-07-30') is None

# integration tests
def test_integration(mock_response):
    lat = 37.7749
    lon = -122.4194
    mock_response.get('https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/4/query', json={
        "features": [
            {
                "geometry": {
                    "rings": [
                        [[-122.4194, 37.7749], [-122.4193, 37.7748], [-122.4192, 37.7747]]
                    ]
                }
            }
        ]
    })
    huc8_polygon = get_huc8_polygon(lat, lon)
    simplified_polygon = simplify_polygon(huc8_polygon)
    geometry = json.dumps({
        "type": "Polygon",
        "coordinates": [simplified_polygon]
    })
    date = "2022-01-01"
    mock_response.get('https://vegdri.cr.usgs.gov/api/v1/data', json={'data': 'some data'})
    data = get_vegdri_data(geometry, date)
    assert data == {'data': 'some data'}