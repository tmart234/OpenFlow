//
//  CoordinatesFetcher.swift
//  WW-app
//
//  Created by Tyler Martin on 3/18/24.
//

import Foundation

struct CoordinatesFetcher {
    static func getCoordinates(siteNumber: String, completion: @escaping (Result<Coordinates, Error>) -> Void) {
        let baseURL = "https://waterdata.usgs.gov/nwis/inventory"
        
        let parameters: [String: String] = [
            "search_site_no": siteNumber,
            "search_site_no_match_type": "exact",
            "group_key": "NONE",
            "format": "sitefile_output",
            "sitefile_output_format": "rdb",
            "column_name": "site_no,station_nm,dec_lat_va,dec_long_va",
            "list_of_search_criteria": "search_site_no"
        ]
        
        var components = URLComponents(string: baseURL)!
        components.queryItems = parameters.map { URLQueryItem(name: $0.key, value: $0.value) }
        
        let url = components.url!
        
        let task = URLSession.shared.dataTask(with: url) { data, response, error in
            if let error = error {
                completion(.failure(error))
                return
            }
            
            guard let data = data else {
                completion(.failure(NSError(domain: "CoordinatesFetcher", code: 0, userInfo: [NSLocalizedDescriptionKey: "No data received"])))
                return
            }
            
            let responseString = String(data: data, encoding: .utf8)!
            let lines = responseString.components(separatedBy: "\n")
            
            // Filter out comment lines and retrieve relevant data
            let dataLines = lines.filter { !$0.starts(with: "#") }
            
            guard dataLines.count >= 3 else {
                completion(.failure(NSError(domain: "CoordinatesFetcher", code: 0, userInfo: [NSLocalizedDescriptionKey: "Invalid response data"])))
                return
            }
            
            let corddata = dataLines[2]
            let fields = corddata.components(separatedBy: "\t")
            
            guard fields.count >= 4 else {
                completion(.failure(NSError(domain: "CoordinatesFetcher", code: 0, userInfo: [NSLocalizedDescriptionKey: "Invalid response data"])))
                return
            }
            
            let siteNo = fields[0]
            let stationName = fields[1]
            let latitude = Double(fields[2]) ?? 0.0
            let longitude = Double(fields[3]) ?? 0.0
            
            let coordinates = Coordinates(siteNo: siteNo, stationName: stationName, latitude: latitude, longitude: longitude)
            completion(.success(coordinates))
        }
        
        task.resume()
    }
}

struct Coordinates {
    let siteNo: String
    let stationName: String
    let latitude: Double
    let longitude: Double
}
