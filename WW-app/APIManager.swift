//
//  APIManager.swift
//  WW-app
//
//  Created by Tyler Martin on 3/29/23.
//

import Foundation
import Alamofire
import SwiftyJSON
import SwiftCSV
import CSV

struct StorageData: Codable {
    let date: Date
    let storage: Double
}
class APIManager {
    static let shared = APIManager()
    private let nrcsBaseURL = "https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customSingleStationReport/daily/start_of_period/"
    struct WeatherData {
        let highTemperature: Double
        let lowTemperature: Double
    }
    enum APIError: Error {
        case invalidURL
        case requestFailed
        case noData
        case decodingError
        case invalidSiteID
    }


    func getWeatherData(latitude: Double, longitude: Double, apiKey: String, completionHandler: @escaping (Result<WeatherData, Error>) -> Void) {
        let url = "https://api.openweathermap.org/data/2.5/weather?lat=\(latitude)&lon=\(longitude)&units=imperial&appid=\(apiKey)"
        AF.request(url).validate().responseDecodable(of: WeatherResponse.self) { response in
            switch response.result {
            case .success(let weatherResponse):
                let highTemperature = weatherResponse.main.temp_max
                let lowTemperature = weatherResponse.main.temp_min
                
                let weatherData = WeatherData(highTemperature: highTemperature, lowTemperature: lowTemperature)
                completionHandler(.success(weatherData))
                
            case .failure(let error):
                completionHandler(.failure(error))
            }
        }
    }
    let reservoirDetails: [Int: (name: String, capacity: Double)] = [
        100163: ("Turquoise Lake Reservoir", 129440),
        100275: ("Twin Lakes Reservoir", 141000),
        2000: ("Green Mountain Reservoir", 154600),
        2005: ("Williams Fork Reservoir", 97000),
        1999: ("Granby Lake", 539758)
    ]

    func getReservoirDetails(for siteID: Int, completion: @escaping (Result<ReservoirInfo, APIError>) -> Void) {
        guard let reservoir = reservoirDetails[siteID] else {
            completion(.failure(.invalidSiteID))
            return
        }

        fetchReservoirData(siteID: siteID) { result in
            switch result {
            case .success(let reservoirData):
                guard let latestStorageData = reservoirData.data.last else {
                    completion(.failure(.noData))
                    return
                }

                let percentageFilled = (latestStorageData.storage / reservoir.capacity) * 100
                let reservoirInfo = ReservoirInfo(reservoirName: reservoir.name,
                                                  reservoirData: reservoirData,
                                                  percentageFilled: min(percentageFilled, 125.0))

                completion(.success(reservoirInfo))
            case .failure(let error):
                completion(.failure(error))
            }
        }
    }
    func getFlowData(usgsSiteID: Int, completionHandler: @escaping (Result<String, Error>) -> Void) {
        let siteIDString = String(format: "%08d", usgsSiteID)
        let url = "https://waterservices.usgs.gov/nwis/iv/?sites=\(siteIDString)&parameterCd=00060&startDT=2023-03-24T23:34:28.131-06:00&endDT=2023-03-31T23:34:28.131-06:00&siteStatus=all&format=rdb"
        print("USGS URL: ", url)
        AF.request(url).validate().responseString { response in
            switch response.result {
            case .success(let value):
                let lines = value.split(separator: "\n")
                var latestDateTime: Date?
                var latestFlowData: Double?
                let dateFormatter = DateFormatter()
                dateFormatter.dateFormat = "yyyy-MM-dd HH:mm"
                dateFormatter.timeZone = TimeZone(abbreviation: "MDT")

                for line in lines {
                    if line.starts(with: "USGS") {
                        let columns = line.split(separator: "\t")
                        if columns.count > 2, let dateTime = dateFormatter.date(from: String(columns[2])) {
                            if latestDateTime == nil || dateTime > latestDateTime! {
                                latestDateTime = dateTime
                                if let flowData = Double(columns[4].trimmingCharacters(in: .whitespaces)) {
                                    latestFlowData = flowData
                                }
                            }
                        }
                    }
                }

                if let flowData = latestFlowData {
                    let formattedFlowData = String(format: "%.1f cfs", flowData)
                    completionHandler(.success(formattedFlowData))
                } else {
                    completionHandler(.failure(APIError.noData))
                }
            case .failure(let error):
                completionHandler(.failure(error))
            }
        }
    }

    func fetchReservoirData(siteID: Int, completion: @escaping (Result<ReservoirData, APIError>) -> Void) {
        let urlString = "https://www.usbr.gov/uc/water/hydrodata/reservoir_data/\(siteID)/json/17.json"

        print("urlString:", urlString)
        guard let url = URL(string: urlString) else {
            completion(.failure(.invalidURL))
            return
        }

        URLSession.shared.dataTask(with: url) { data, response, error in
            if let _ = error {
                completion(.failure(.requestFailed))
                return
            }

            guard let data = data else {
                completion(.failure(.noData))
                return
            }

            do {
                let decodedDataResponse = try JSONDecoder().decode(ReservoirDataResponse.self, from: data)
                let storageDataArray = try decodedDataResponse.data.compactMap { element -> StorageData? in
                    guard let dateString = element[0] as? String else {
                        throw APIError.decodingError
                    }

                    let dateFormatter = DateFormatter()
                    dateFormatter.dateFormat = "yyyy-MM-dd"
                    guard let date = dateFormatter.date(from: dateString) else {
                        throw APIError.decodingError
                    }

                    if let storage = element[1] as? Double {
                        return StorageData(date: date, storage: storage)
                    } else {
                        return nil
                    }
                }
                completion(.success(ReservoirData(data: storageDataArray)))
            } catch {
                print("Error decoding JSON:", error)
                completion(.failure(.decodingError))
            }
        }.resume()
    }

    func getSnowpackDataFromCSV(stationID: String, completion: @escaping (Result<SnowpackData, Error>) -> Void) {
        print("Fetching snowpack data for station ID: \(stationID)")
        let url = "https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customSingleStationReport/daily/\(stationID):CO:SNTL%7Cid=%22%22%7Cname/0,0/WTEQ::value,WTEQ::pctOfAverage_1991"

        AF.request(url).responseData { response in
            switch response.result {
            case .success(let data):
                do {
                    let stream = InputStream(data: data)
                    let reader = try CSVReader(stream: stream, hasHeaderRow: true)
                    let dateFormatter = DateFormatter()
                    dateFormatter.dateFormat = "yyyy-MM-dd"
                    
                    var latestDataRow: [String]?
                    while let row = reader.next() {
                        if row[0].starts(with: "2") {
                            latestDataRow = row
                        }
                    }
                    
                    guard let dataRow = latestDataRow else {
                        completion(.failure(APIError.noData))
                        return
                    }
                    
                    guard let date = dateFormatter.date(from: dataRow[0]) else {
                        completion(.failure(APIError.invalidSiteID))
                        return
                    }

                    guard let sweValue = Double(dataRow[1]) else {
                        completion(.failure(APIError.noData))
                        return
                    }

                    guard let swePctOfAverage = Double(dataRow[2]) else {
                        completion(.failure(APIError.noData))
                        return
                    }
                    
                    let stationName = "Station \(stationID)"
                    let snowDepth = 0.0 // No snow depth provided in CSV data
                    let snowpackData = SnowpackData(stationName: stationName, reportDate: date, snowWaterEquivalent: sweValue, snowDepth: snowDepth, percentOfAverage: swePctOfAverage)
                    completion(.success(snowpackData))
                } catch {
                    completion(.failure(error))
                }
            case .failure(let error):
                completion(.failure(error))
            }
        }
    }
    
    func extractSWEValue(from row: [String: String], stationID: String) -> (String, Double)? {
        let pattern = "(\(stationID)) Snow Water Equivalent \\(in\\) Start of Day Values"
        
        for (key, value) in row {
            if key.range(of: pattern, options: .regularExpression) != nil {
                if let doubleValue = Double(value) {
                    return (key, doubleValue)
                } else {
                    return nil
                }
            }
        }
        
        return nil
    }
}
