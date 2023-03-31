//
//  APIManager.swift
//  WW-app
//
//  Created by Tyler Martin on 3/29/23.
//

import Foundation
import Alamofire
import SwiftyJSON

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
        100163: ("Turquoise Lake Reservoir", 129_440),
        100275: ("Twin Lakes Reservoir", 141_000),
        2000: ("Green Mountain Reservoir", 154_600),
        2005: ("Williams Fork Reservoir", 97_000),
        1999: ("Granby Lake", 539_758)
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

        func getSnowpackData(stationID: String, completion: @escaping (Result<SnowpackData, Error>) -> Void) {
            let url = "\(nrcsBaseURL)\(stationID)&dateTime=latest"
            
            AF.request(url).responseData { response in
                switch response.result {
                case .success(let data):
                    print("Raw JSON Response: \(String(data: data, encoding: .utf8) ?? "")")
                    do {
                        print("Raw JSON Response: \(String(data: data, encoding: .utf8) ?? "")")
                        let nrcsResponse = try JSONDecoder().decode(NRCSResponse.self, from: data)
                        
                        guard let elementData = nrcsResponse.report.data.first,
                              let snowWaterEquivalentString = elementData.data["SNWD"],
                              let snowDepthString = elementData.data["SNOW"],
                              let snowWaterEquivalent = Double(snowWaterEquivalentString),
                              let snowDepth = Double(snowDepthString) else {
                            completion(.failure(NSError(domain: "", code: -1, userInfo: nil)))
                            return
                        }
                        
                        let stationName = elementData.metadata.station.name
                        let snowpackData = SnowpackData(stationName: stationName, snowWaterEquivalent: snowWaterEquivalent, snowDepth: snowDepth)
                        completion(.success(snowpackData))
                    } catch {
                        completion(.failure(error))
                    }
                case .failure(let error):
                    completion(.failure(error))
                }
            }
        }
    }
