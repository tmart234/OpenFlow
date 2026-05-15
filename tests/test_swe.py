import pytest
import requests_mock
from datetime import datetime

import data.get_swe as get_swe


@pytest.fixture
def mock_api():
    with requests_mock.Mocker() as m:
        yield m


def test_get_huc_id_extracts_huc8_from_arcgis(mock_api):
    mock_api.get(
        "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/4/query",
        json={"features": [{"attributes": {"huc8": "14010001"}}]},
    )
    assert get_swe.get_huc_id(40.0, -106.0, level=8) == "14010001"


def test_get_huc_id_returns_none_when_no_features(mock_api):
    mock_api.get(
        "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/4/query",
        json={"features": []},
    )
    assert get_swe.get_huc_id(40.0, -106.0) is None


def test_get_huc_id_rejects_unsupported_level():
    with pytest.raises(ValueError):
        get_swe.get_huc_id(40.0, -106.0, level=9)


def test_get_snotel_triplets_filters_records_without_triplet(mock_api):
    mock_api.get(
        "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/stations",
        json=[
            {"stationTriplet": "1030:CO:SNTL", "name": "Arapaho Ridge"},
            {"stationTriplet": "913:CO:SNTL", "name": "Berthoud Summit"},
            {"name": "no triplet -- skipped"},
        ],
    )
    assert get_swe.get_snotel_station_triplets("14010001") == ["1030:CO:SNTL", "913:CO:SNTL"]


def test_get_snotel_triplets_empty_for_empty_huc(mock_api):
    assert get_swe.get_snotel_station_triplets("") == []


def test_get_swe_timeseries_averages_across_stations(mock_api):
    mock_api.get(
        "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data",
        json=[
            {"stationTriplet": "1030:CO:SNTL",
             "data": [{"stationElement": {"elementCode": "WTEQ"},
                       "values": [{"date": "2024-01-01", "value": 6.0},
                                  {"date": "2024-01-02", "value": 8.0}]}]},
            {"stationTriplet": "913:CO:SNTL",
             "data": [{"stationElement": {"elementCode": "WTEQ"},
                       "values": [{"date": "2024-01-01", "value": 10.0},
                                  {"date": "2024-01-02", "value": 12.0}]}]},
        ],
    )
    df = get_swe.get_swe_timeseries(
        ["1030:CO:SNTL", "913:CO:SNTL"],
        datetime(2024, 1, 1), datetime(2024, 1, 2))
    assert len(df) == 2
    assert df.loc[df['Date'] == '2024-01-01', 'SWE'].iloc[0] == 8.0
    assert df.loc[df['Date'] == '2024-01-02', 'SWE'].iloc[0] == 10.0


def test_get_swe_returns_empty_when_huc_lookup_fails(mock_api):
    mock_api.get(
        "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/4/query",
        json={"features": []},
    )
    df = get_swe.get_swe(40.0, -106.0, datetime(2024, 1, 1), datetime(2024, 1, 2))
    assert df.empty


def test_get_swe_returns_empty_when_no_snotel_stations(mock_api):
    mock_api.get(
        "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/4/query",
        json={"features": [{"attributes": {"huc8": "14010001"}}]},
    )
    mock_api.get(
        "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/stations",
        json=[],
    )
    df = get_swe.get_swe(40.0, -106.0, datetime(2024, 1, 1), datetime(2024, 1, 2))
    assert df.empty
