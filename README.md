# OpenFlowColorado

![Alt text](SS2.png?raw=true "Home Screen")
![Alt text](SS1.png?raw=true "River View")

# NOAA
The National Oceanic and Atmospheric Administration (NOAA) provides various datasets related to climate, weather, and environmental data. In this project, we use the NOAA Global Historical Climatology Network Daily (GHCND) dataset to fetch temperature data from the nearest weather station.

To see a list of all NOAA stations:
https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt

NCEI Data API Documentation: 
https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation

NCEI Search (datasets) API Documentation: 
https://www.ncei.noaa.gov/support/access-search-service-api-user-documentation

NCEI Management (for metadata) API Documentation: 
https://www.ncei.noaa.gov/support/access-support-service

NCEI Order API Documentation:
https://www.ncei.noaa.gov/support/access-order-service
- you'll need to buy a dataset w/ NOAA account that has the API key
- most datasets are free

NCEI GitHub: 
https://github.com/NCEI-NOAAGov

# USGS
The United States Geological Survey (USGS) provides a wide range of water data, including flows, reservoirs, lakes, and groundwater data. The USGS water data is accessible through various services, such as Instant Value Service, Daily Value Service, and Site Service. These services allow users to retrieve recent and historical values for streamflow and other time-series parameters, daily statistical data, and information about hydrologic sites.
Flows (Historical and Current)

Flow: https://waterdata.usgs.gov/co/nwis/current/?type=flow&group_key=huc_cd

Reservoirs & Lakes: https://waterdata.usgs.gov/co/nwis/current/?type=res&group_key=huc_cd

Groundwater Current: https://waterdata.usgs.gov/co/nwis/current/?type=gw

Groundwater Historical: https://waterdata.usgs.gov/co/nwis/uv/?referred_module=gw

## USGS API services

Instant Value Service Documentation:
https://waterservices.usgs.gov/rest/IV-Service.html
- use this service to retrieve recent and historical values for streamflow as well as data for other regular time-series parameters served by the USGS

Daily Value Service Documentation:
https://waterservices.usgs.gov/rest/DV-Service.html
- use this service to retrieve daily statistical data

Site Service Documentation:
https://waterservices.usgs.gov/rest/Site-Service.html
- use this service to retrieve information about the millions of hydrologic sites

## USGS GitHUB

River DL
https://github.com/USGS-R/river-dl
- intent of this repository was to predict stream temperature and streamflow

FCPG Tools:
https://fcpgtools.readthedocs.io/en/latest/cookbook.html
- produce pre-computed girds of upstream basin characteristics


# USBR
The United States Bureau of Reclamation (USBR) manages water resources in the western United States, including reservoirs and dams. The USBR provides historical and current data for reservoirs, as well as an API called RISE for accessing information about reservoirs, such as coordinates, location tags, and basin information. The RISE API can also be used to fetch historical data.

- https://www.usbr.gov/rsvrWater/HistoricalApp.html
- https://www.usbr.gov/gp/lakes_reservoirs/colorado_lakes.html

## RISE API
- https://data.usbr.gov/rise/api
    - https://data.usbr.gov/rise/api/result/downloadall?query[]=itemId.383.before.2023-05-01.after.2023-04-24.order.ASC&type=json&filename=RISE%20Time%20Series%20Query%20Package%20(2023-05-01)&order=ASC

## pyForecast
https://github.com/usbr/PyForecast
 - useful in predicting monthly and seasonal inflows and streamflows
 - "In this project, we use a combination of APIs and services provided by NOAA, USGS, and USBR to fetch, analyze, and visualize temperature, water flow, and reservoir data. The collected data is then stored in CSV files for further processing and analysis."

# Other Github Repos
https://github.com/AIStream-Peelout/flow-forecast
- LSTM flow forecasting