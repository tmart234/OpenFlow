//
//  River.swift
//  WW-app
//
//  Created by Tyler Martin on 3/29/23.
//

import Foundation

struct DWRResponse: Codable {
    var resultList: [RiverData]?
    
    enum CodingKeys: String, CodingKey {
        case resultList = "ResultList"
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        resultList = try container.decodeIfPresent([RiverData].self, forKey: .resultList)
    }
}

struct DWRFlowResponse: Codable {
    let stationNum: Int?
    let abbrev: String?
    let usgsSiteId: String?
    let measType: String?
    let measDate: String?
    let value: Double?
    let flagA: String?
    let flagB: String?
    let flagC: String?
    let flagD: String?
    let dataSource: String?
    let modified: String?
    let measUnit: String?
}

class RiverDataModel: ObservableObject {
    @Published var rivers: [RiverData] = []
    @Published var riverCoordinates: [String: Coordinates] = [:]
    var favoriteRivers: [RiverData] = []
    private var isUSGSFetchComplete = false
    private var isDWRFetchComplete = false
    
    init() {
        if let fetchedFavorites = LocalStorage.getFavoriteRivers() {
            self.favoriteRivers = fetchedFavorites
            for index in rivers.indices {
                if favoriteRivers.contains(where: { $0.siteNumber == rivers[index].siteNumber }) {
                    rivers[index].isFavorite = true
                }
            }
        }
        if !isDWRFetchComplete {
            fetchDWRRivers()
        }
        
        if !isUSGSFetchComplete {
            fetchAndParseData()
        }
    }
    
    func fetchDWRRivers() {
        guard !isDWRFetchComplete else {
            return
        }
        
        guard let url = URL(string: "https://dwr.state.co.us/rest/get/api/v2/surfacewater/surfacewaterstations") else {
            print("Invalid URL")
            return
        }
        
        URLSession.shared.dataTask(with: url) { data, response, error in
            if let error = error {
                print("Error fetching DWR rivers: \(error)")
                return
            }
            
            if let data = data {
                let decoder = JSONDecoder()
                decoder.dateDecodingStrategy = .iso8601
                
                do {
                    if let json = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any],
                       let resultList = json["ResultList"] as? [[String: Any]] {
                        var updatedData: [RiverData] = []
                        
                        for riverDict in resultList {
                            let siteNumber = riverDict["usgsSiteId"] as? String ?? ""
                            let stationName = riverDict["stationName"] as? String ?? ""
                            let latitude = riverDict["latitude"] as? Double
                            let longitude = riverDict["longitude"] as? Double
                            let measDate = riverDict["measDate"] as? String
                            let source = riverDict["dataSource"] as? String ?? ""
                            if (source.lowercased().contains("dwr")) {
                                let riverData = RiverData(
                                    agency: "DWR",
                                    siteNumber: siteNumber,
                                    stationName: stationName,
                                    timeSeriesID: "",
                                    parameterCode: "",
                                    resultDate: "",
                                    resultTimezone: "",
                                    resultCode: "",
                                    resultModifiedDate: "",
                                    snotelStationID: "",
                                    reservoirSiteIDs: [],
                                    lastFetchedDate: Date(),
                                    latitude: latitude,
                                    longitude: longitude,
                                    flowRateValue: 0.0
                                )
                                
                                updatedData.append(riverData)
                            }
                        }
                        
                        DispatchQueue.main.async {
                            self.rivers.append(contentsOf: updatedData)
                            self.isDWRFetchComplete = true
                            print("Fetched \(updatedData.count) DWR rivers")
                        }
                    } else {
                        print("Invalid DWR API response format")
                    }
                } catch {
                    print("Error decoding DWR JSON: \(error)")
                }
            }
        }.resume()
    }
    
    func fetchDWRFlow(for river: RiverData, completion: @escaping () -> Void) {
        guard let encodedSiteNumber = river.siteNumber.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed),
              let url = URL(string: "https://dwr.state.co.us/rest/get/api/v2/surfacewater/surfacewatertsday?usgsSiteId=\(encodedSiteNumber)&format=json") else {
            print("Invalid URL")
            completion()
            return
        }
        print(url)
        
        URLSession.shared.dataTask(with: url) { data, response, error in
            if let error = error {
                print("Error: \(error.localizedDescription)")
                return
            }
            
            guard let httpResponse = response as? HTTPURLResponse, (200...299).contains(httpResponse.statusCode) else {
                if let data = data, let errorString = String(data: data, encoding: .utf8) {
                    print("API Error: \(errorString)")
                } else {
                    print("Unknown API Error")
                }
                return
            }
            
            if let data = data {
                let decoder = JSONDecoder()
                decoder.dateDecodingStrategy = .iso8601
                
                do {
                    if let json = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any],
                       let resultList = json["ResultList"] as? [[String: Any]],
                       let latestFlowData = resultList.last {
                        if let value = latestFlowData["value"] as? Double,
                           let measDate = latestFlowData["measDate"] as? String {
                            DispatchQueue.main.async {
                                if let index = self.rivers.firstIndex(where: { $0.id == river.id }) {
                                    self.rivers[index].flowRateValue = value
                                    self.rivers[index].resultDate = measDate
                                }
                            }
                        }
                    } else {
                        print("Invalid DWR API response format")
                    }
                } catch {
                    print("Error decoding DWR JSON: \(error)")
                }
            }
            completion()
        }.resume()
    }
    func fetchMLStationIDs(completion: @escaping ([String]) -> Void) {
        guard let url = URL(string: "https://raw.githubusercontent.com/tmart234/OpenFlowColorado/main/.github/site_ids.txt") else { return }
        
        URLSession.shared.dataTask(with: url) { data, _, _ in
            guard let data = data, let content = String(data: data, encoding: .utf8) else {
                completion([])
                return
            }
            
            let stationIDs = content.components(separatedBy: .newlines).filter { !$0.isEmpty }
            DispatchQueue.main.async {
                completion(stationIDs)
            }
        }.resume()
    }
    
    func updateCoordinates(_ coordinates: [[String: String]]) {
        DispatchQueue.main.async {
            for coordinateDict in coordinates {
                if let siteNumber = coordinateDict["number"],
                   let latitude = Double(coordinateDict["lat"] ?? ""),
                   let longitude = Double(coordinateDict["long"] ?? "") {
                    
                    // Update coordinates for rivers
                    if let index = self.rivers.firstIndex(where: { $0.siteNumber.hasSuffix(siteNumber) }) {
                        self.rivers[index].latitude = latitude
                        self.rivers[index].longitude = longitude
                    }
                }
            }
        }
    }
    
    func fetchAndParseData() {
        guard !isUSGSFetchComplete else {
            return
        }
        if let dataURL = URL(string: "https://waterdata.usgs.gov/co/nwis/current?index_pmcode_STATION_NM=1&index_pmcode_DATETIME=2&index_pmcode_00060=3&group_key=NONE&sitefile_output_format=html_table&column_name=agency_cd&column_name=site_no&column_name=station_nm&sort_key_2=site_no&html_table_group_key=NONE&format=rdb&rdb_compression=value&list_of_search_criteria=realtime_parameter_selection") {
            let task = URLSession.shared.dataTask(with: dataURL) { (data: Data?, response: URLResponse?, error: Error?) in
                if let data = data {
                    if let dataString = String(data: data, encoding: .utf8) {
                        self.parseUSGSData(dataString)
                        
                        CoordinatesFetcher.fetchUSGSCoordinates { result in
                            switch result {
                            case .success(let coordinates):
                                DispatchQueue.main.async {
                                    self.updateCoordinates(coordinates)
                                    self.isUSGSFetchComplete = true
                                }
                            case .failure(let error):
                                print("Error fetching coordinates: \(error)")
                            }
                        }
                    }
                }
            }
            task.resume()
        }
    }
    
    func parseUSGSData(_ data: String) {
        let lines = data.components(separatedBy: .newlines)
        
        var linesToSkip = 2
        var parsedRivers: [RiverData] = []
        for line in lines {
            if !line.isEmpty && !line.hasPrefix("#") {
                if linesToSkip > 0 {
                    linesToSkip -= 1
                    continue
                }
                
                let values = line.components(separatedBy: "\t")
                if values.count >= 9 {
                    let siteNumber = "USGS \(values[1].trimmingCharacters(in: .whitespaces))"
                    let stationName = values[2]
                    let flowReading = values[7]
                    let dateOfReading = values[5]
                    let flowRate = Double(flowReading) ?? 0.0

                    let river = RiverData(
                        agency: "USGS",
                        siteNumber: siteNumber,
                        stationName: stationName,
                        timeSeriesID: "",
                        parameterCode: "",
                        resultDate: dateOfReading,
                        resultTimezone: "",
                        resultCode: "",
                        resultModifiedDate: "",
                        snotelStationID: "",
                        reservoirSiteIDs: [],
                        lastFetchedDate: Date(),
                        latitude: nil,
                        longitude: nil,
                        flowRateValue: flowRate
                    )
                    parsedRivers.append(river)
                }
            }
        }
        
        DispatchQueue.main.async {
            self.rivers = parsedRivers
        }
    }

    func updateFavoriteRivers() {
        favoriteRivers = rivers.filter { $0.isFavorite }
        LocalStorage.saveFavoriteRivers(favoriteRivers)
    }
    
    func loadFavoriteRivers() {
        if let savedFavorites = LocalStorage.getFavoriteRivers() {
            self.rivers = savedFavorites
        }
    }
    
    func toggleFavorite(at index: Int) {
        rivers[index].isFavorite.toggle()
        updateFavoriteRivers()
    }
}
    
class LocalStorage {
    private static let favoriteRiversKey = "favoriteRivers"
    
    // Save favorite rivers
    static func saveFavoriteRivers(_ favoriteRivers: [RiverData]) {
        if let encodedData = try? JSONEncoder().encode(favoriteRivers) {
            UserDefaults.standard.set(encodedData, forKey: favoriteRiversKey)
        }
    }
    
    // Retrieve favorite rivers
    static func getFavoriteRivers() -> [RiverData]? {
        if let data = UserDefaults.standard.data(forKey: favoriteRiversKey) {
            return try? JSONDecoder().decode([RiverData].self, from: data)
        }
        return nil
    }
}
