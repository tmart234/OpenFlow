//
//  RiverDataType.swift
//  WW-app
//
//  Created by Tyler Martin on 3/30/24.
//

import Foundation

enum RiverDataType: Identifiable {
    case usgs(USGSRiverData)
    case dwr(DWRRiverData)
    
    var stationName: String {
        switch self {
        case .usgs(let riverData):
            return riverData.stationName
        case .dwr(let riverData):
            return riverData.stationName
        }
    }
    
    var id: UUID {
        switch self {
        case .usgs(let riverData):
            return riverData.id
        case .dwr(let riverData):
            return riverData.id
        }
    }
    
    var siteNumber: String {
        switch self {
        case .usgs(let riverData):
            return riverData.siteNumber
        case .dwr(let riverData):
            return String(riverData.stationNum)
        }
    }
    
    var latitude: Double? {
        switch self {
        case .usgs(let riverData):
            return riverData.latitude
        case .dwr(let riverData):
            return riverData.latitude
        }
    }
    
    var longitude: Double? {
        switch self {
        case .usgs(let riverData):
            return riverData.longitude
        case .dwr(let riverData):
            return riverData.longitude
        }
    }
    
    var snotelStationID: String {
        switch self {
        case .usgs(let riverData):
            return riverData.snotelStationID
        case .dwr:
            return ""
        }
    }
    
    var lastFetchedDate: Date {
        switch self {
        case .usgs(let riverData):
            return riverData.lastFetchedDate
        case .dwr:
            return Date()
        }
    }
    
    var reservoirSiteIDs: [Int] {
        switch self {
        case .usgs(let riverData):
            return riverData.reservoirSiteIDs
        case .dwr:
            return []
        }
    }
    
    var isFavorite: Bool {
        get {
            switch self {
            case .usgs(let riverData):
                return riverData.isFavorite
            case .dwr(let riverData):
                return riverData.isFavorite
            }
        }
        set {
            switch self {
            case .usgs(var riverData):
                riverData.isFavorite = newValue
            case .dwr(var riverData):
                riverData.isFavorite = newValue
            }
        }
    }
}
