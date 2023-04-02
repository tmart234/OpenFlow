//
//  River.swift
//  WW-app
//
//  Created by Tyler Martin on 3/29/23.
//

import Foundation

struct RiverData: Identifiable, Codable {
    let id: Int
    let name: String
    let location: String
    let snotelStationID: String
    let usgsSiteID: Int
    let reservoirSiteIDs: [Int]
    let lastFetchedDate: Date
}

class LocalStorage {
    private static let riverDataKey = "riverData"

    static func saveRiverData(_ riverData: RiverData) {
        if let encodedData = try? JSONEncoder().encode(riverData) {
            UserDefaults.standard.set(encodedData, forKey: riverDataKey)
        }
    }

    static func getRiverData() -> RiverData? {
        if let data = UserDefaults.standard.data(forKey: riverDataKey) {
            return try? JSONDecoder().decode(RiverData.self, from: data)
        }
        return nil
    }
}


