import pytest
import requests_mock
from nasa_moisture import search_smap_data, get_smap_soil_moisture_data, process_smap_data, get_smap_timeseries, main
import pandas as pd

@pytest.fixture
def mock_cmr_response():
    with requests_mock.Mocker() as m:
        url = 'https://cmr.earthdata.nasa.gov/search/granules.json'
        latitude = 45.0
        longitude = -120.0
        bounding_box = f"{longitude-0.1},{latitude-0.1},{longitude+0.1},{latitude+0.1}"
        params = {
            'short_name': 'SPL2SMP_E',
            'version': '006',
            'temporal': '2020-01-01T00:00:00Z,2020-01-31T23:59:59Z',
            'bounding_box': bounding_box,
            'page_size': '100',
            'page_num': '1'
        }
        query_string = '&'.join(f'{k}={v}' for k, v in params.items())
        m.get(f'{url}?{query_string}', json={'feed': {'entry': [{'links': [{'href': 'https://example.com/granule'}]}]}})
        yield m

@pytest.fixture
def mock_granule_response():
    with requests_mock.Mocker() as m:
        m.get('https://example.com/granule', json={
            'Soil_Moisture_Retrieval_Data': {
                'soil_moisture_option3': 'example_data',
                'latitude': 45.0,
                'longitude': -120.0,
                'tb_time_utc': '2020-01-01T00:00:00Z',
                'retrieval_qual_flag': 'example_quality_flag'
            }
        })
        yield m

def test_search_smap_data(mock_cmr_response):
    result = search_smap_data('2020-01-01', '2020-01-31', '-180,-90,180,90', 'bearer_token')
    assert result == {
        'feed': {
            'entry': [
                {'links': [{'href': 'https://example.com/granule'}]}
            ]
        }
    }

def test_get_smap_soil_moisture_data(mock_granule_response):
    result = get_smap_soil_moisture_data('https://example.com/granule', 'bearer_token')
    assert result == {
        'Soil_Moisture_Retrieval_Data': {
            'soil_moisture_option3': 'example_data',
            'latitude': 45.0,
            'longitude': -120.0,
            'tb_time_utc': '2020-01-01T00:00:00Z',
            'retrieval_qual_flag': 'example_quality_flag'
        }
    }

def test_process_smap_data():
    data = {
        'Soil_Moisture_Retrieval_Data': {
            'soil_moisture_option3': 'example_data',
            'latitude': 45.0,
            'longitude': -120.0,
            'tb_time_utc': '2020-01-01T00:00:00Z',
            'retrieval_qual_flag': 'example_quality_flag'
        }
    }
    result = process_smap_data(data)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 5)
    assert result.columns.tolist() == ['date', 'latitude', 'longitude', 'soil_moisture', 'quality_flag']

def test_get_smap_timeseries(mock_cmr_response, mock_granule_response):
    result = get_smap_timeseries('bearer_token', '2020-01-01', '2020-01-31', 45.0, -120.0)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 5)
    assert result.columns.tolist() == ['date', 'latitude', 'longitude', 'soil_moisture', 'quality_flag']

def test_main(mock_cmr_response, mock_granule_response):
    result = main('bearer_token', '2020-01-01', '2020-01-31', 45.0, -120.0)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 5)
    assert result.columns.tolist() == ['date', 'latitude', 'longitude', 'soil_moisture', 'quality_flag']