import Foundation

struct Coordinates {
    let siteNo: String
    let stationName: String
    let latitude: Double
    let longitude: Double
}

struct CoordinatesFetcher {
    static func fetchUSGSCoordinates(completion: @escaping (Result<[[String: String]], Error>) -> Void) {
        let urlString = "https://waterdata.usgs.gov/nwis/inventory?state_cd=co&group_key=NONE&format=sitefile_output&sitefile_output_format=rdb&column_name=agency_cd&column_name=site_no&column_name=station_nm&column_name=site_tp_cd&column_name=dec_lat_va&column_name=dec_long_va&list_of_search_criteria=state_cd"
        
        guard let url = URL(string: urlString) else {
            completion(.failure(NSError(domain: "CoordinatesFetcher", code: 0, userInfo: [NSLocalizedDescriptionKey: "Invalid URL"])))
            return
        }
        
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
            
            let dataLines = lines.filter { !$0.hasPrefix("#") }
            
            var coordinates: [[String: String]] = []
            
            for line in dataLines {
                let fields = line.components(separatedBy: "\t")
                
                guard fields.count >= 6 else {
                    continue
                }
                
                let siteNo = fields[1]
                let latitude = fields[4]
                let longitude = fields[5]
                
                let coordinateDict = ["number": siteNo, "lat": latitude, "long": longitude]
                coordinates.append(coordinateDict)
            }
            
            completion(.success(coordinates))
        }
        
        task.resume()
    }
    func fetchUSGSCoordinate(for siteID: String, completion: @escaping (Result<Coordinates, Error>) -> Void) {
        let urlString = "https://waterdata.usgs.gov/nwis/inventory?search_site_no=\(siteID)&search_site_no_match_type=exact&group_key=NONE&format=sitefile_output&sitefile_output_format=rdb&column_name=agency_cd&column_name=site_no&column_name=station_nm&column_name=dec_lat_va&column_name=dec_long_va&list_of_search_criteria=search_site_no"
        
        guard let url = URL(string: urlString) else {
            completion(.failure(NSError(domain: "CoordinatesFetcher", code: 0, userInfo: [NSLocalizedDescriptionKey: "Invalid URL"])))
            return
        }
        
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
            let dataLines = lines.filter { !$0.hasPrefix("#") }
            
            for line in dataLines {
                let fields = line.components(separatedBy: "\t")
                
                guard fields.count >= 5 else {
                    continue
                }
                
                let siteNo = fields[1]
                let stationName = fields[2]
                let latitude = Double(fields[3])
                let longitude = Double(fields[4])
                
                if siteNo == siteID, let lat = latitude, let lon = longitude {
                    let coordinates = Coordinates(siteNo: siteNo, stationName: stationName, latitude: lat, longitude: lon)
                    completion(.success(coordinates))
                    return
                }
            }
            
            completion(.failure(NSError(domain: "CoordinatesFetcher", code: 0, userInfo: [NSLocalizedDescriptionKey: "No coordinates found"])))
        }
        
        task.resume()
    }
}
