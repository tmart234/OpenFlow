//
//  River.swift
//  WW-app
//
//  Created by Tyler Martin on 3/29/23.
//

import Foundation

struct USGSRiverData: Identifiable, Codable {
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
    let latitude: Double
    let longitude: Double
    var flowRate: Int
    // CustomStringConvertible conformance
    var description: String {
        return "Site Number: \(siteNumber), Station Name: \(stationName), Flow Rate: \(flowRate)"
    }

}

class RiverDataModel: ObservableObject {
    @Published var rivers: [USGSRiverData] = []
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
                    }
                }
            }
            print("USGS data URL: ", dataURL)
            task.resume()
        }
    }

    
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
                if values.count >= 10 {
                    let isFavorite = favoriteRivers.contains(where: { $0.siteNumber == values[1] })
                    let cleanedStationName = values[2].replacingOccurrences(of: ", CO", with: "").replacingOccurrences(of: ".", with: "")
                    
                    let river = USGSRiverData(
                        agency: values[0],
                        siteNumber: values[1],
                        stationName: cleanedStationName,
                        timeSeriesID: values[3],
                        parameterCode: values[4],
                        resultDate: values[5],
                        resultTimezone: values[6],
                        resultValue: values[7],
                        resultCode: values[8],
                        resultModifiedDate: values[9],
                        snotelStationID: "",
                        reservoirSiteIDs: [],
                        lastFetchedDate: Date(),
                        isFavorite: isFavorite,
                        latitude: 0.0, // Placeholder value
                        longitude: 0.0,  // Placeholder value
                        flowRate: 0
                    )
                    parsedRivers.append(river)
                }
            }
        }

        // Update the rivers property on the main thread
        DispatchQueue.main.async {
            self.rivers = parsedRivers
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


