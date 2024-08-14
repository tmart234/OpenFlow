import asyncio
import xarray as xr
import pandas as pd
from shapely.geometry import Polygon
from pyproj import CRS
import fsspec
import dask
from dask.distributed import Client
import argparse
import logging
from get_poly import get_huc8_polygon, simplify_polygon

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def fetch_soil_moisture_data(polygon: Polygon, date_range: pd.date_range) -> xr.Dataset:
    """
    Asynchronously fetch CPC soil moisture data for a given polygon and date range.
    
    Args:
    polygon (Polygon): Shapely Polygon object defining the area of interest
    date_range (pd.date_range): Date range for data retrieval
    
    Returns:
    xr.Dataset: Dataset containing soil moisture data for the specified region and time
    """
    # CPC soil moisture data is available via Google Cloud Storage
    gcs_url = "gs://noaa-cpc-pds/soil-moisture/"
    
    async with fsspec.open_files(gcs_url + "*.nc", mode='rb') as files:
        datasets = [xr.open_dataset(file, engine='h5netcdf', chunks={'time': 'auto'}) 
                    for file in files if file.start.date() in date_range]
    
    combined_data = xr.concat(datasets, dim='time')
    
    # Ensure CRS matches the polygon CRS (assuming WGS84)
    data_crs = CRS.from_epsg(4326)
    combined_data = combined_data.rio.write_crs(data_crs)
    
    # Clip data to polygon
    clipped_data = combined_data.rio.clip([polygon], all_touched=True)
    
    return clipped_data

async def process_soil_moisture_data(data: xr.Dataset) -> pd.DataFrame:
    """
    Process the retrieved soil moisture data.
    
    Args:
    data (xr.Dataset): Dataset containing soil moisture data
    
    Returns:
    pd.DataFrame: Processed soil moisture data
    """
    # Example processing - adjust as needed
    mean_moisture = data['soilw'].mean(dim=['lat', 'lon'])
    return mean_moisture.to_dataframe()

async def main(lat: float, lon: float, start_date: str, end_date: str):
    # Set up dask client for parallel processing
    client = Client(n_workers=4, threads_per_worker=2, memory_limit='4GB')
    
    # Get HUC8 polygon
    huc8_polygon = get_huc8_polygon(lat, lon)
    if not huc8_polygon:
        logging.error("No HUC8 polygon found")
        await client.close()
        return

    # Simplify the polygon
    simplified_polygon = simplify_polygon(huc8_polygon)
    logging.info(f"Simplified polygon: {simplified_polygon}")

    # Conve
    # rt to Shapely Polygon
    polygon = Polygon(simplified_polygon)

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    data = await fetch_soil_moisture_data(polygon, date_range)
    
    # Process data using dask for parallelization
    processed_data = await dask.compute(process_soil_moisture_data(data))[0]
    
    print(processed_data)
    
    # Clean up dask client
    await client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch soil moisture data for HUC8 polygon based on lat/lon.')
    parser.add_argument('--lat', type=float, required=True, help='Latitude')
    parser.add_argument('--lon', type=float, required=True, help='Longitude')
    parser.add_argument('--start_date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, required=True, help='End date (YYYY-MM-DD)')
    args = parser.parse_args()

    asyncio.run(main(args.lat, args.lon, args.start_date, args.end_date))