"""
Tests for get_reservoir (USBR RISE storage + release).

The mapping file and the RISE result endpoint are both mocked. We don't hit
data.usbr.gov in the test run; the test asserts the mapping is parsed correctly
and that the JSON:API response shape is normalised into a [Date, value] frame.
"""

import pandas as pd
import pytest
import requests_mock as _rm

from data import get_reservoir


@pytest.fixture
def mock_api():
    with _rm.Mocker() as m:
        yield m


def test_load_mapping_skips_comments_and_blanks(tmp_path):
    path = tmp_path / 'reservoirs.txt'
    path.write_text(
        "# header comment\n"
        "\n"
        "USGS:09163500, Lake Powell, 6126, 6127\n"
        "# inline comment line\n"
        "USGS:09114500, Blue Mesa, 6135,\n"  # no release id
        "  USGS:09070500 , Dillon , 6200 , 6201  \n"  # extra whitespace
    )
    mapping = get_reservoir.load_mapping(str(path))
    assert set(mapping) == {'USGS:09163500', 'USGS:09114500', 'USGS:09070500'}
    assert mapping['USGS:09163500'] == [('Lake Powell', '6126', '6127')]
    # The release id is empty -> None for that slot.
    assert mapping['USGS:09114500'] == [('Blue Mesa', '6135', None)]
    # Whitespace is stripped.
    assert mapping['USGS:09070500'] == [('Dillon', '6200', '6201')]


def test_load_mapping_handles_missing_file(tmp_path):
    missing = tmp_path / 'does_not_exist.txt'
    assert get_reservoir.load_mapping(str(missing)) == {}


def test_load_mapping_skips_malformed_lines(tmp_path):
    path = tmp_path / 'reservoirs.txt'
    path.write_text(
        "USGS:09163500, Lake Powell, 6126, 6127\n"
        "this line has too few fields\n"
        "USGS:09114500, Blue Mesa, 6135\n"
    )
    mapping = get_reservoir.load_mapping(str(path))
    assert set(mapping) == {'USGS:09163500', 'USGS:09114500'}


def test_fetch_rise_series_parses_jsonapi_response(mock_api):
    mock_api.get(
        get_reservoir.RISE_RESULT_URL,
        json={
            'data': [
                {'attributes': {'dateTime': '2024-01-01T00:00:00Z', 'result': 100.0}},
                {'attributes': {'dateTime': '2024-01-02T00:00:00Z', 'result': 200.0}},
                {'attributes': {'dateTime': '2024-01-03T00:00:00Z', 'result': 300.0}},
            ],
            'links': {},
        },
    )
    df = get_reservoir.fetch_rise_series('6126', '2024-01-01', '2024-01-03')
    assert list(df['Date']) == ['2024-01-01', '2024-01-02', '2024-01-03']
    assert list(df['value']) == [100.0, 200.0, 300.0]


def test_fetch_rise_series_collapses_sub_daily_to_daily_mean(mock_api):
    mock_api.get(
        get_reservoir.RISE_RESULT_URL,
        json={
            'data': [
                {'attributes': {'dateTime': '2024-01-01T00:00:00Z', 'result': 100.0}},
                {'attributes': {'dateTime': '2024-01-01T12:00:00Z', 'result': 200.0}},
            ],
            'links': {},
        },
    )
    df = get_reservoir.fetch_rise_series('6126', '2024-01-01', '2024-01-01')
    assert len(df) == 1
    assert df.iloc[0]['value'] == 150.0


def test_fetch_rise_series_returns_empty_on_empty_id(mock_api):
    # No request must be issued when item_id is falsy.
    df = get_reservoir.fetch_rise_series(None, '2024-01-01', '2024-01-03')
    assert df.empty
    assert mock_api.call_count == 0


def test_get_reservoir_returns_empty_for_unmapped_station(tmp_path):
    mapping_path = tmp_path / 'reservoirs.txt'
    mapping_path.write_text("USGS:09163500, Lake Powell, 6126, 6127\n")
    df = get_reservoir.get_reservoir(
        'USGS:UNREGULATED', '2024-01-01', '2024-01-03', mapping_path=str(mapping_path))
    assert df.empty
    assert list(df.columns) == ['Date', 'reservoir_storage', 'reservoir_release']


def test_get_reservoir_sums_multiple_reservoirs(mock_api, tmp_path):
    mapping_path = tmp_path / 'reservoirs.txt'
    mapping_path.write_text(
        "USGS:09163500, ResA, 100, 200\n"
        "USGS:09163500, ResB, 101, 201\n"
    )

    # Different responses by query string -- match on itemId.
    def _by_item(request, context):
        item_id = request.qs.get('itemid', [''])[0]
        value_map = {
            '100': 1000.0,  # ResA storage
            '101': 500.0,   # ResB storage
            '200': 50.0,    # ResA release
            '201': 25.0,    # ResB release
        }
        v = value_map[item_id]
        return {
            'data': [{'attributes': {'dateTime': '2024-01-01T00:00:00Z', 'result': v}}],
            'links': {},
        }

    mock_api.get(get_reservoir.RISE_RESULT_URL, json=_by_item)
    df = get_reservoir.get_reservoir(
        'USGS:09163500', '2024-01-01', '2024-01-01', mapping_path=str(mapping_path))
    assert len(df) == 1
    assert df.iloc[0]['reservoir_storage'] == 1500.0  # 1000 + 500
    assert df.iloc[0]['reservoir_release'] == 75.0    # 50 + 25
