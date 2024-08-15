import pytest
import requests_mock
from utils.get_coordinates import get_usgs_coordinates, get_dwr_coordinates, main

@pytest.fixture
def mock_requests():
    with requests_mock.Mocker() as m:
        yield m

@pytest.fixture
def usgs_mock_response():
    return '''
# comment line
# another comment line
site_no	station_nm	dec_lat_va	dec_long_va
12345678	Test Station	40.123456	-105.654321
07094500	ARKANSAS RIVER AT PARKDALE, CO	38.4872189	-105.373604
'''

@pytest.fixture
def dwr_mock_response():
    return '''
# comment line
# another comment line
abbrev	latitude	longitude
TESTABBR	39.987654	-104.123456
'''

@pytest.fixture
def mock_sys_argv(monkeypatch):
    def _mock_argv(args):
        monkeypatch.setattr('sys.argv', ['script_name'] + args)
    return _mock_argv

def test_get_usgs_coordinates(mock_requests, usgs_mock_response):
    mock_requests.get('https://waterdata.usgs.gov/nwis/inventory', text=usgs_mock_response)

    result = get_usgs_coordinates('12345678')
    expected = {
        'latitude': '40.123456',
        'longitude': '-105.654321'
    }
    assert result == expected

def test_get_usgs_coordinates_specific_station(mock_requests, usgs_mock_response):
    mock_requests.get('https://waterdata.usgs.gov/nwis/inventory', text=usgs_mock_response)

    result = get_usgs_coordinates('07094500')
    expected = {
        'latitude': '38.4872189',
        'longitude': '-105.373604'
    }
    assert result == expected

def test_get_usgs_coordinates_invalid_station(mock_requests):
    mock_requests.get('https://waterdata.usgs.gov/nwis/inventory', text='')

    result = get_usgs_coordinates('invalid_station')
    assert result is None

def test_get_usgs_coordinates_api_error(mock_requests):
    mock_requests.get('https://waterdata.usgs.gov/nwis/inventory', status_code=500)

    result = get_usgs_coordinates('12345678')
    assert result is None

def test_get_dwr_coordinates(mock_requests, dwr_mock_response):
    mock_requests.get('https://dwr.state.co.us/Rest/GET/api/v2/surfacewater/surfacewaterstations', text=dwr_mock_response)

    result = get_dwr_coordinates('TESTABBR')
    expected = {
        'latitude': '39.987654',
        'longitude': '-104.123456'
    }
    assert result == expected

def test_get_dwr_coordinates_invalid_abbrev(mock_requests):
    mock_requests.get('https://dwr.state.co.us/Rest/GET/api/v2/surfacewater/surfacewaterstations', text='')

    result = get_dwr_coordinates('INVALID')
    assert result is None

def test_get_dwr_coordinates_api_error(mock_requests):
    mock_requests.get('https://dwr.state.co.us/Rest/GET/api/v2/surfacewater/surfacewaterstations', status_code=500)

    result = get_dwr_coordinates('TESTABBR')
    assert result is None

def test_main_usgs(mock_sys_argv, monkeypatch, capsys):
    mock_sys_argv(['usgs', '07094500'])
    
    def mock_get_usgs_coordinates(site_number):
        return {'latitude': '38.4872189', 'longitude': '-105.373604'}
    
    monkeypatch.setattr('utils.get_coordinates.get_usgs_coordinates', mock_get_usgs_coordinates)
    
    main()
    
    captured = capsys.readouterr()
    assert captured.out.strip() == "{'latitude': '38.4872189', 'longitude': '-105.373604'}"

def test_main_dwr(mock_sys_argv, monkeypatch, capsys):
    mock_sys_argv(['dwr', 'TESTABBR'])
    
    def mock_get_dwr_coordinates(abbrev):
        return {'latitude': '39.987654', 'longitude': '-104.123456'}
    
    monkeypatch.setattr('utils.get_coordinates.get_dwr_coordinates', mock_get_dwr_coordinates)
    
    main()
    
    captured = capsys.readouterr()
    assert captured.out.strip() == "{'latitude': '39.987654', 'longitude': '-104.123456'}"

def test_main_invalid_source(mock_sys_argv, capsys):
    mock_sys_argv(['invalid', 'TESTABBR'])
    
    with pytest.raises(SystemExit):
        main()
    
    captured = capsys.readouterr()
    assert "Invalid source. Use 'usgs' or 'dwr'." in captured.out

def test_main_missing_arguments(mock_sys_argv, capsys):
    mock_sys_argv([])
    
    with pytest.raises(SystemExit):
        main()
    
    captured = capsys.readouterr()
    assert "error: the following arguments are required: site_type, site_number" in captured.err
    
if __name__ == "__main__":
    pytest.main()