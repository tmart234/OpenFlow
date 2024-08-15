import pytest
from datetime import datetime, timedelta
from get_CODWR_flow import get_historical_data
import pandas as pd
import requests_mock


# Define the tests for get_daily_flow_data function
# Mock base URL and endpoint as per your script
dwr_base_url = "https://dwr.state.co.us/Rest/GET/api/v2/telemetrystations/telemetrytimeserieshour/"

@pytest.fixture
def mock_dwr_response():
    with requests_mock.Mocker() as m:
        yield m

def test_successful_data_fetch_dwr(mock_dwr_response):
    # Mocking a successful API response
    mock_dwr_response.get(dwr_base_url, json={
        "ResultList": [
            {"measDate": "2022-05-01 00:00:00", "measValue": "10"},
            {"measDate": "2022-05-01 01:00:00", "measValue": "20"},
            # More entries as needed
        ]
    })

    start_date = datetime.now() - timedelta(days=1)
    end_date = datetime.now()

    result = get_historical_data("ARKCANCO", start_date, end_date)
    assert not result.empty
    assert len(result) == 1
    assert result.loc[0, 'Min Discharge'] == 10.0
    assert result.loc[0, 'Max Discharge'] == 20.0

def test_api_error_handling(mock_dwr_response):
    # Mocking a failure scenario
    mock_dwr_response.get(dwr_base_url, status_code=500)
    start_date = datetime.now() - timedelta(days=1)
    end_date = datetime.now()

    result = get_historical_data("ARKCANCO", start_date, end_date)
    assert result.empty

def test_no_data_returned(mock_dwr_response):
    # Mocking no data scenario
    mock_dwr_response.get(dwr_base_url, json={"ResultList": []})
    start_date = datetime.now() - timedelta(days=1)
    end_date = datetime.now()

    result = get_historical_data("ARKCANCO", start_date, end_date)
    assert result.empty

def test_invalid_data_format(mock_dwr_response):
    # Mocking an invalid data format
    mock_dwr_response.get(dwr_base_url, json={
        "ResultList": [
            {"measDate": "This is not a date", "measValue": "Not a number"}
        ]
    })
    start_date = datetime.now() - timedelta(days=1)
    end_date = datetime.now()

    result = get_historical_data("ARKCANCO", start_date, end_date)
    assert result.empty

