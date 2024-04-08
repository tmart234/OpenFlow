//
//  River.swift
//  WW-app
//
//  Created by Tyler Martin on 3/29/23.
//

import Foundation

struct DWRResponse: Codable {
    let resultList: [RiverData]?
}

class RiverDataModel: ObservableObject {
    @Published var rivers: [RiverData] = []
    @Published var riverCoordinates: [String: Coordinates] = [:]
    var favoriteRivers: [RiverData] = []
    
    init() {
        if let fetchedFavorites = LocalStorage.getFavoriteRivers() {
            self.favoriteRivers = fetchedFavorites
            for index in rivers.indices {
                if favoriteRivers.contains(where: { $0.siteNumber == rivers[index].siteNumber }) {
                    rivers[index].isFavorite = true
                }
            }
        }
        fetchAndParseData()
    }
    
    func fetchDWRRivers() {
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
                decoder.dateDecodingStrategy = .iso8601 // Set the date decoding strategy

                if let dwrResponse = try? decoder.decode(DWRResponse.self, from: data) {
                    if let decodedData = dwrResponse.resultList {
                        DispatchQueue.main.async {
                            self.rivers.append(contentsOf: decodedData)
                            print("Fetched \(decodedData.count) DWR rivers")
                        }
                    } else {
                        print("Error decoding DWR JSON: 'resultList' is nil or not found")
                    }
                } else {
                    print("Error decoding DWR JSON: Invalid data")
                }
            }
        }.resume()
    }
    
    func fetchDWRFlow(for river: RiverData) {
        guard let url = URL(string: "https://dwr.state.co.us/rest/get/api/v2/surfacewater/surfacewatertsday?abbrev=\(river.siteNumber)") else {
            print("Invalid URL")
            return
        }
        
        URLSession.shared.dataTask(with: url) { data, response, error in
            if let data = data {
                do {
                    let decodedData = try JSONDecoder().decode([RiverData].self, from: data)
                    if let flowData = decodedData.first {
                        DispatchQueue.main.async {
                            if let index = self.rivers.firstIndex(where: { $0.id == river.id }) {
                                self.rivers[index].value = flowData.value
                                self.rivers[index].measDate = flowData.measDate
                            }
                        }
                    }
                } catch {
                    print("Error decoding JSON: \(error)")
                }
            }
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
        for coordinateDict in coordinates {
            if let siteNumber = coordinateDict["number"],
               let latitude = Double(coordinateDict["lat"] ?? ""),
               let longitude = Double(coordinateDict["long"] ?? "") {
                
                // Update coordinates for rivers
                if let index = rivers.firstIndex(where: { $0.siteNumber == "USGS \(siteNumber)" }) {
                    var updatedRiver = rivers[index]
                    updatedRiver.latitude = latitude
                    updatedRiver.longitude = longitude
                    DispatchQueue.main.async {
                        self.rivers[index] = updatedRiver
                    }
                }
            }
        }
    }
    
    func fetchAndParseData() {
        if let dataURL = URL(string: "https://waterdata.usgs.gov/co/nwis/current?index_pmcode_STATION_NM=1&index_pmcode_DATETIME=2&index_pmcode_00060=3&group_key=NONE&sitefile_output_format=html_table&column_name=agency_cd&column_name=site_no&column_name=station_nm&sort_key_2=site_no&html_table_group_key=NONE&format=rdb&rdb_compression=value&list_of_search_criteria=realtime_parameter_selection") {
            let task = URLSession.shared.dataTask(with: dataURL) { (data: Data?, response: URLResponse?, error: Error?) in
                if let data = data {
                    if let dataString = String(data: data, encoding: .utf8) {
                        self.parseUSGSData(dataString)
                        
                        CoordinatesFetcher.fetchUSGSCoordinates { result in
                            switch result {
                            case .success(let coordinates):
                                self.updateCoordinates(coordinates)
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
                    let flowRate = Int(flowReading) ?? 0
                    
                    let river = RiverData(
                        agency: "USGS",
                        siteNumber: siteNumber,
                        stationName: stationName,
                        timeSeriesID: "",
                        parameterCode: "",
                        resultDate: dateOfReading,
                        resultTimezone: "",
                        resultValue: flowReading,
                        resultCode: "",
                        resultModifiedDate: "",
                        snotelStationID: "",
                        reservoirSiteIDs: [],
                        lastFetchedDate: Date(),
                        latitude: nil,
                        longitude: nil,
                        flowRate: flowRate
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
