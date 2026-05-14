from data.get_flow import get_daily_flow_data
import pytest
from datetime import datetime, timedelta
import requests_mock
import pandas as pd

# Mock base URL for USGS
usgs_base_url = "https://nwis.waterservices.usgs.gov/nwis/iv/"

@pytest.fixture
def mock_response():
    with requests_mock.Mocker() as m:
        yield m

def test_successful_data_fetch_usgs(mock_response):
    # Mocking a successful API response for USGS. The real NWIS RDB format is
    # tab-separated with columns: agency, site, datetime, tz, value, qualifier.
    mock_response.get(usgs_base_url, text=(
        "# Some comment\n"
        "USGS\t09114500\t2022-05-01 00:00\tMST\t15.0\tA\n"
        "USGS\t09114500\t2022-05-01 01:00\tMST\t20.0\tA\n"
    ))

    start_date = datetime.now() - timedelta(days=1)
    end_date = datetime.now()

    result = get_daily_flow_data('09114500', start_date, end_date)
    assert not result.empty
    assert len(result) == 1
    assert result.loc[0, 'Min Flow'] == 15.0
    assert result.loc[0, 'Max Flow'] == 20.0

def test_api_error_handling_usgs(mock_response):
    # Mocking a failure scenario for USGS
    mock_response.get(usgs_base_url, status_code=500)
    start_date = datetime.now() - timedelta(days=1)
    end_date = datetime.now()

    result = get_daily_flow_data('09114500', start_date, end_date)
    assert result.empty

def test_no_data_returned_usgs(mock_response):
    # Mocking no data scenario for USGS
    mock_response.get(usgs_base_url, text='''
    # Some comment
    ''')
    start_date = datetime.now() - timedelta(days=1)
    end_date = datetime.now()

    result = get_daily_flow_data('09114500', start_date, end_date)
    assert result.empty

def test_invalid_data_format_usgs(mock_response):
    # Mocking an invalid data format for USGS
    mock_response.get(usgs_base_url, text='''
    # Some comment
    USGS    09114500    NotADate    NotAFlow
    ''')
    start_date = datetime.now() - timedelta(days=1)
    end_date = datetime.now()

    result = get_daily_flow_data('09114500', start_date, end_date)
    assert result.empty

@pytest.mark.network
def test_flow_data_retrieval():
    """Test normal data retrieval for a known good flow site and valid date range."""
    flow_site_id = '09114500'  # Known USGS site with flow data
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    df = get_daily_flow_data(flow_site_id, start_date, end_date)
    assert not df.empty, "DataFrame should not be empty for valid inputs"
    assert 'Date' in df.columns and 'Min Flow' in df.columns and 'Max Flow' in df.columns, "DataFrame must contain the expected columns"

@pytest.mark.network
def test_empty_data_retrieval():
    """Test data retrieval for a date range with no expected data."""
    flow_site_id = '09114500'  # Use a valid site ID
    start_date = datetime(1900, 1, 1)  # An old date with no data
    end_date = datetime(1900, 1, 2)
    df = get_daily_flow_data(flow_site_id, start_date, end_date)
    assert df.empty, "DataFrame should be empty when there's no data for the date range"

@pytest.mark.network
def test_invalid_site_id():
    """Test data retrieval with an invalid site ID."""
    flow_site_id = '00000000'  # Invalid site ID
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    df = get_daily_flow_data(flow_site_id, start_date, end_date)
    assert df.empty, "DataFrame should be empty for invalid site IDs"
