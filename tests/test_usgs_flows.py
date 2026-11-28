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


def test_parse_rdb_flow_extracts_valid_rows_only():
    from data.get_flow import _parse_rdb_flow
    text = (
        "# comment\n"
        "agency_cd\tsite_no\tdatetime\ttz_cd\tvalue\tqual\n"
        "5s\t15s\t20d\t6s\t14n\t10s\n"                          # RDB format-spec row
        "USGS\t09114500\t2022-05-01 00:00\tMST\t15.0\tA\n"
        "USGS\t09114500\t2022-05-01 00:15\tMST\t\tA\n"          # missing value -> skipped
        "USGS\t09114500\t2022-05-01 00:20\tMST\t-999999\tIce\n"   # negative sentinel -> skipped
        "USGS\t09114500\t2022-05-01 00:30\tMST\t25.0\tP\n"
    )
    assert _parse_rdb_flow(text) == [("2022-05-01", 15.0), ("2022-05-01", 25.0)]


def test_parse_rdb_dv_flow_extracts_daily_mean_column():
    from data.get_flow import _parse_rdb_dv_flow
    text = (
        "# comment\n"
        "agency_cd\tsite_no\tdatetime\t18015_00060_00003\t18015_00060_00003_cd\n"
        "5s\t15s\t20d\t14n\t10s\n"
        "USGS\t09095500\t2024-05-01\t2500\tA\n"
        "USGS\t09095500\t2024-05-02\tIce\tA\n"
        "USGS\t09095500\t2024-05-03\t-999999\tA\n"
        "USGS\t09095500\t2024-05-04\t3100.5\tP\n"
    )
    assert _parse_rdb_dv_flow(text) == [("2024-05-01", 2500.0), ("2024-05-04", 3100.5)]


def test_get_daily_mean_flow_uses_dv_and_tops_up_trailing_days_from_iv(mock_response):
    from data.get_flow import USGS_DV_URL, get_daily_mean_flow

    mock_response.get(USGS_DV_URL, text=(
        "# comment\n"
        "agency_cd\tsite_no\tdatetime\t18015_00060_00003\t18015_00060_00003_cd\n"
        "5s\t15s\t20d\t14n\t10s\n"
        "USGS\t09095500\t2024-05-01\t1000\tA\n"
        "USGS\t09095500\t2024-05-02\t1200\tA\n"
    ))
    mock_response.get(usgs_base_url, text=(
        "# comment\n"
        "USGS\t09095500\t2024-05-03 00:00\tMST\t1400.0\tP\n"
        "USGS\t09095500\t2024-05-03 12:00\tMST\t1600.0\tP\n"
    ))

    df = get_daily_mean_flow('09095500', datetime(2024, 5, 1), datetime(2024, 5, 3))
    assert list(df['Date']) == ['2024-05-01', '2024-05-02', '2024-05-03']
    assert list(df['flow']) == pytest.approx([1000.0, 1200.0, 1500.0])
    # IV was queried ONLY for the missing trailing day (2024-05-03), not May 1-2.
    iv_reqs = [r for r in mock_response.request_history if '/nwis/iv/' in r.url]
    assert len(iv_reqs) == 1
    assert 'startDT=2024-05-03' in iv_reqs[0].url


def test_get_daily_mean_flow_falls_back_to_iv_when_dv_empty(mock_response):
    from data.get_flow import USGS_DV_URL, get_daily_mean_flow

    mock_response.get(USGS_DV_URL, text="# no dv data\n")
    mock_response.get(usgs_base_url, text=(
        "# comment\n"
        "USGS\t09095500\t2024-05-01 00:00\tMST\t500.0\tA\n"
        "USGS\t09095500\t2024-05-01 06:00\tMST\t700.0\tA\n"
    ))

    df = get_daily_mean_flow('09095500', datetime(2024, 5, 1), datetime(2024, 5, 1))
    assert len(df) == 1
    assert df.loc[0, 'flow'] == pytest.approx(600.0)
