import pytest
from datetime import datetime
from data.get_CODWR_flow import get_historical_data
import requests_mock

# get_CODWR_flow now reads the historical daily archive (surfacewatertsday),
# not the telemetry hourly feed, which only retains ~30 days of recent data.
dwr_base_url = "https://dwr.state.co.us/Rest/GET/api/v2/surfacewater/surfacewatertsday/"


@pytest.fixture
def mock_dwr_response():
    with requests_mock.Mocker() as m:
        yield m


def test_successful_data_fetch_dwr(mock_dwr_response):
    mock_dwr_response.get(dwr_base_url, json={
        "PageCount": 1,
        "ResultList": [
            {"measDate": "2022-05-01T00:00:00-06:00", "value": 12.0},
            {"measDate": "2022-05-02T00:00:00-06:00", "value": 15.0},
        ],
    })
    result = get_historical_data("ARKCANCO", datetime(2022, 5, 1), datetime(2022, 5, 2))
    assert not result.empty
    assert len(result) == 2
    # DWR's daily archive carries one value per day, so min == max == value.
    assert result.loc[0, 'Min Flow'] == 12.0
    assert result.loc[0, 'Max Flow'] == 12.0


def test_paginates_through_all_pages(mock_dwr_response):
    mock_dwr_response.get(dwr_base_url, [
        {'json': {"PageCount": 2, "ResultList": [
            {"measDate": "2022-05-01T00:00:00-06:00", "value": 10.0}]}},
        {'json': {"PageCount": 2, "ResultList": [
            {"measDate": "2022-05-02T00:00:00-06:00", "value": 20.0}]}},
    ])
    result = get_historical_data("ARKCANCO", datetime(2022, 5, 1), datetime(2022, 5, 2))
    assert len(result) == 2
    assert set(result['Date']) == {"2022-05-01", "2022-05-02"}


def test_api_error_handling(mock_dwr_response):
    mock_dwr_response.get(dwr_base_url, status_code=500)
    result = get_historical_data("ARKCANCO", datetime(2022, 5, 1), datetime(2022, 5, 2))
    assert result.empty


def test_no_data_returned(mock_dwr_response):
    mock_dwr_response.get(dwr_base_url, json={"PageCount": 1, "ResultList": []})
    result = get_historical_data("ARKCANCO", datetime(2022, 5, 1), datetime(2022, 5, 2))
    assert result.empty


def test_invalid_records_are_skipped(mock_dwr_response):
    mock_dwr_response.get(dwr_base_url, json={
        "PageCount": 1,
        "ResultList": [
            {"measDate": "not a date", "value": "not a number"},
            {"measDate": None, "value": None},
        ],
    })
    result = get_historical_data("ARKCANCO", datetime(2022, 5, 1), datetime(2022, 5, 2))
    assert result.empty
