import pytest
from datetime import datetime, timedelta
import data.get_swe as get_swe  # Assuming get_swe is your script with the main function

def test_date_validation():
    """Test that end date must be greater than start date."""
    basin_name = 'South Platte'
    basin_type = 'basin'
    start_date = datetime.now()
    end_date = start_date - timedelta(days=10)  # Intentionally set to be wrong

    with pytest.raises(ValueError):
        get_swe.main(basin_name, basin_type, start_date, end_date)
    
def test_data_retrieval():
    """Test that data retrieval returns a non-empty DataFrame for valid dates."""
    basin_name = 'South Platte'
    basin_type = 'basin'
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    result = get_swe.main(basin_name, basin_type, start_date, end_date)
    assert not result.empty, "DataFrame should not be empty"

# Test for invalid basin type
def test_invalid_basin_type():
    """Test handling of invalid basin types."""
    basin_name = 'South Platte'
    basin_type = 'unknown'  # Invalid type
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    result = get_swe.main(basin_name, basin_type, start_date, end_date)
    assert result.empty, "DataFrame should be empty for invalid basin types"

# check basin vs sub basin
# Using a tuple in parametrize to differentiate between basin types and their names
@pytest.mark.parametrize("basin_type,basin_name", [
    ('basin', 'South Platte'),  # Known basin
    ('subbasin', 'Upper South Platte')  # Known subbasin
])
def test_basin_subbasin_retrieval(basin_type, basin_name):
    """Test data retrieval for both basin and subbasin types with known good inputs."""
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    result = get_swe.main(basin_name, basin_type, start_date, end_date)
    assert not result.empty, f"DataFrame should not be empty for valid {basin_type} type with {basin_name}"

