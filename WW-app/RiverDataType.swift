//
//  RiverDataType.swift
//  WW-app
//
//  Created by Tyler Martin on 3/30/24.
//

import Foundation

struct RiverData: Codable, Identifiable {
    var id = UUID()
    // USGS, DWR, etc
    var agency: String
    let siteNumber: String
    let stationName: String
    let stationNum: Int
    let timeSeriesID: String
    let parameterCode: String
    var resultDate: String
    let resultTimezone: String
    let resultCode: String
    let resultModifiedDate: String
    let snotelStationID: String
    let reservoirSiteIDs: [Int]
    let lastFetchedDate: Date
    var isFavorite: Bool = false
    var latitude: Double?
    var longitude: Double?
    var flowRateValue: Double?
    var measDate: Date?
    
    // CustomStringConvertible conformance
    var description: String {
        return "Site Number: \(siteNumber), Station Name: \(stationName), Flow Rate: \(String(describing: flowRateValue) )"
    }
}
