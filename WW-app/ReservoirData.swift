//
//  ReservoirData.swift
//  WW-app
//
//  Created by Tyler Martin on 3/30/23.
//

import Foundation
import AnyCodable

struct ReservoirDataResponse: Decodable {
    let data: [[Any]]
    
    enum CodingKeys: String, CodingKey {
        case data = "data"
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        data = try container.decodeIfPresent([[AnyCodable]].self, forKey: .data)?.map { $0.map { $0.value } } ?? []
    }
}

struct ReservoirData {
    var data: [StorageData]
}


struct ReservoirInfo: Identifiable {
    let id = UUID()
    let reservoirName: String
    let reservoirData: ReservoirData
    let percentageFilled: Double
    
    static let reservoirDetails: [Int: (name: String, capacity: Double)] = [
        100163: ("Turquoise Lake Reservoir", 129440),
        100275: ("Twin Lakes Reservoir", 141000),
        2000: ("Green Mountain Reservoir", 154600),
        2005: ("Williams Fork Reservoir", 097000),
        1999: ("Granby Lake", 539758)
    ]
}

