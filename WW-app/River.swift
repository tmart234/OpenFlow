//
//  River.swift
//  WW-app
//
//  Created by Tyler Martin on 3/29/23.
//

import Foundation

struct USGSRiverData: Codable, Identifiable {
    var id = UUID()
    let agency: String
    let siteNumber: String
    let stationName: String
    let timeSeriesID: String
    let parameterCode: String
    let resultDate: String
    let resultTimezone: String
    let resultValue: String
    let resultCode: String
    let resultModifiedDate: String
    let snotelStationID: String
    let reservoirSiteIDs: [Int]
    let lastFetchedDate: Date
    var isFavorite: Bool
    var latitude: Double?
    var longitude: Double?
    var flowRate: Int
    // CustomStringConvertible conformance
    var description: String {
        return "Site Number: \(siteNumber), Station Name: \(stationName), Flow Rate: \(flowRate)"
    }

}

struct DWRRiverData: Codable, Identifiable {
    var id = UUID()
    let stationNum: Int
    let abbrev: String
    let stationName: String
    let latitude: Double?
    let longitude: Double?
    var value: Double?
    var measDate: Date?
    var isFavorite: Bool = false
}

class RiverDataModel: ObservableObject {
    @Published var rivers: [USGSRiverData] = []
    @Published var dwrRivers: [DWRRiverData] = []
    var favoriteRivers: [USGSRiverData] = []
    
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
                do {
                    let decodedData = try JSONDecoder().decode([DWRRiverData].self, from: data)
                    DispatchQueue.main.async {
                        self.dwrRivers = decodedData
                        print("Fetched \(decodedData.count) DWR rivers")
                    }
                } catch {
                    print("Error decoding JSON: \(error)")
                }
            }
        }.resume()
    }
    func fetchDWRFlow(for river: DWRRiverData) {
        guard let url = URL(string: "https://dwr.state.co.us/rest/get/api/v2/surfacewater/surfacewatertsday?abbrev=\(river.abbrev)") else {
            print("Invalid URL")
            return
        }
        
        URLSession.shared.dataTask(with: url) { data, response, error in
            if let data = data {
                do {
                    let decodedData = try JSONDecoder().decode([DWRRiverData].self, from: data)
                    if let flowData = decodedData.first {
                        DispatchQueue.main.async {
                            if let index = self.dwrRivers.firstIndex(where: { $0.id == river.id }) {
                                self.dwrRivers[index].value = flowData.value
                                self.dwrRivers[index].measDate = flowData.measDate
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
    func fetchAndParseData() {
        if let dataURL = URL(string: "https://waterdata.usgs.gov/co/nwis/current?index_pmcode_STATION_NM=1&index_pmcode_DATETIME=2&index_pmcode_00060=3&group_key=NONE&sitefile_output_format=html_table&column_name=agency_cd&column_name=site_no&column_name=station_nm&sort_key_2=site_no&html_table_group_key=NONE&format=rdb&rdb_compression=value&list_of_search_criteria=realtime_parameter_selection") {
            let task = URLSession.shared.dataTask(with: dataURL) { data, response, error in
                if let data = data {
                    if let dataString = String(data: data, encoding: .utf8) {
                        self.parseData(dataString)
                        
                        self.fetchCoordinatesForAllStations { result in
                            switch result {
                            case .success(let stationCoordinates):
                                DispatchQueue.main.async {
                                    for (siteNumber, coordinates) in stationCoordinates {
                                        if let index = self.rivers.firstIndex(where: { $0.siteNumber == siteNumber }) {
                                            self.rivers[index].latitude = coordinates.latitude
                                            self.rivers[index].longitude = coordinates.longitude
                                            print("Updated coordinates for station: \(siteNumber), latitude: \(coordinates.latitude), longitude: \(coordinates.longitude)")
                                        }
                                    }
                                }
                            case .failure(let error):
                                print("Error fetching coordinates for stations: \(error)")
                            }
                        }
                    }
                }
            }
            print("USGS data URL: ", dataURL)
            task.resume()
        }
    }
    func parseData(_ data: String) {
        let lines = data.components(separatedBy: .newlines)
        
        var linesToSkip = 2  // Counter to track lines to skip
        var parsedRivers: [USGSRiverData] = [] // Temporary storage for the parsed rivers
        for line in lines {
            if !line.isEmpty && !line.hasPrefix("#") {
                if linesToSkip > 0 {
                    linesToSkip -= 1
                    continue
                }
                
                let values = line.components(separatedBy: "\t")
                if values.count >= 9 {
                    let siteNumber = values[1]
                    let stationName = values[2]
                    let flowReading = values[7]
                    let dateOfReading = values[5]
                    
                    let river = USGSRiverData(
                        agency: "",
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
                        isFavorite: false,
                        latitude: nil,
                        longitude: nil,
                        flowRate: 0
                    )
                    parsedRivers.append(river)
                }
            }
        }
        DispatchQueue.main.async {
            self.rivers = parsedRivers
        }
    }

    func fetchCoordinatesForAllStations(completion: @escaping (Result<[String: (latitude: Double, longitude: Double)], Error>) -> Void) {
        let urlString = "https://waterdata.usgs.gov/nwis/inventory?state_cd=co&group_key=NONE&format=sitefile_output&sitefile_output_format=rdb&column_name=agency_cd&column_name=site_no&column_name=station_nm&column_name=site_tp_cd&column_name=dec_lat_va&column_name=dec_long_va&list_of_search_criteria=state_cd"
        
        guard let url = URL(string: urlString) else {
            completion(.failure(NSError(domain: "", code: 400, userInfo: [NSLocalizedDescriptionKey: "Invalid URL"])))
            return
        }
        
        URLSession.shared.dataTask(with: url) { data, response, error in
            if let error = error {
                completion(.failure(error))
                return
            }
            
            guard let data = data, let dataString = String(data: data, encoding: .utf8) else {
                completion(.failure(NSError(domain: "", code: 500, userInfo: [NSLocalizedDescriptionKey: "No data"])))
                return
            }
            
            let lines = dataString.components(separatedBy: .newlines)
            var stationCoordinates: [String: (latitude: Double, longitude: Double)] = [:]
            
            // Skip the header lines
            var linesToSkip = 0
            for line in lines {
                if line.hasPrefix("#") {
                    linesToSkip += 1
                } else {
                    break
                }
            }
            
            for line in lines.dropFirst(linesToSkip) {
                let values = line.components(separatedBy: .whitespaces)
                if values.count >= 6,
                   let latitude = Double(values[4]),
                   let longitude = Double(values[5]) {
                    let siteNumber = values[1]
                    stationCoordinates[siteNumber] = (latitude, longitude)
                    print("Fetched coordinates for station: \(siteNumber), latitude: \(latitude), longitude: \(longitude)")
                }
            }
            
            completion(.success(stationCoordinates))
        }.resume()
        
        
        func updateFavoriteRivers() {
            favoriteRivers = rivers.filter { $0.isFavorite }
            LocalStorage.saveFavoriteRivers(favoriteRivers)
        }
        
        func loadFavoriteRivers() {
            if let savedFavorites = LocalStorage.getFavoriteRivers() {
                rivers = savedFavorites
            }
        }
        func toggleFavorite(at index: Int) {
            rivers[index].isFavorite.toggle()
            updateFavoriteRivers()
        }
        
        func toggleFavorite(at index: Int, isDWR: Bool) {
            dwrRivers[index].isFavorite.toggle()
        }
        
    }
}
    
// saves:
// 1) favorite USGS rivers and
// 2) feteched river data in RiverDeatilView
class LocalStorage {
    private static let riverDataKey = "riverData"
    private static let favoriteRiversKey = "favoriteRivers"
    // Save the river data
    static func saveRiverData(_ riverData: USGSRiverData) {
        if let encodedData = try? JSONEncoder().encode(riverData) {
            UserDefaults.standard.set(encodedData, forKey: riverDataKey)
        }
    }
    // Retrieve the river data
    static func getRiverData() -> USGSRiverData? {
        if let data = UserDefaults.standard.data(forKey: riverDataKey) {
            return try? JSONDecoder().decode(USGSRiverData.self, from: data)
        }
        return nil
    }
    
    // Save favorite rivers
    static func saveFavoriteRivers(_ favoriteRivers: [USGSRiverData]) {
        if let encodedData = try? JSONEncoder().encode(favoriteRivers) {
            UserDefaults.standard.set(encodedData, forKey: favoriteRiversKey)
        }
    }
    
    // Retrieve favorite rivers
    static func getFavoriteRivers() -> [USGSRiverData]? {
        if let data = UserDefaults.standard.data(forKey: favoriteRiversKey) {
            return try? JSONDecoder().decode([USGSRiverData].self, from: data)
        }
        return nil
    }
}
