import requests
import json

def fetch_all_table_names():
    base_url = "https://SDMDataAccess.sc.egov.usda.gov"
    headers = {"Content-Type": "application/json"}

    query_url = f"{base_url}/Tabular/post.rest"
    # Query to fetch all table names from the information schema
    sql_query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES"
    query_data = {
        "SERVICE": "query",
        "REQUEST": "query",
        "QUERY": sql_query,
        "FORMAT": "JSON"
    }
    response = requests.post(query_url, headers=headers, data=json.dumps(query_data))
    
    if response.status_code == 200:
        return response.json()
    else:
        return f"Failed to fetch table names: {response.text}"

# Execute the function to get all table names
result = fetch_all_table_names()
print(result)
