//
//  ReservoirManager.swift
//  WW-app
//
//  Created by Tyler Martin on 3/30/23.
//

import Foundation
import AnyCodable

class ReservoirManager {
    static let shared = ReservoirManager()
    
    private init() {}
    
    func calculatePercentageFilled(current: Double, reservoirID: Int) -> Double {
        if let reservoir = ReservoirInfo.reservoirDetails[reservoirID] {
            return (current / reservoir.capacity) * 100
        } else {
            return 0.0
        }
    }
    
    func fetchReservoirData(siteIDs: [Int], completion: @escaping (Result<[ReservoirInfo], Error>) -> Void) {
        var reservoirData: [ReservoirInfo] = []
        let dispatchGroup = DispatchGroup()
        
        for siteID in siteIDs {
            dispatchGroup.enter()
            APIManager.shared.getReservoirDetails(for: siteID) { result in
                switch result {
                case .success(let info):
                    // Find the most recent storage value
                    var currentStorage: Double = 0.0
                    var mostRecentDate: Date?
                    for storageData in info.reservoirData.data {
                        if mostRecentDate == nil || storageData.date > mostRecentDate! {
                            mostRecentDate = storageData.date
                            currentStorage = storageData.storage
                        }
                    }
                    
                    let percentageFilled = self.calculatePercentageFilled(current: currentStorage, reservoirID: siteID)
                    
                    let reservoirInfo = ReservoirInfo(reservoirName: info.reservoirName, reservoirData: info.reservoirData, percentageFilled: percentageFilled)
                    reservoirData.append(reservoirInfo)
                case .failure(let error):
                    print("Error fetching reservoir details:", error)
                }
                dispatchGroup.leave()
            }
        }
        
        dispatchGroup.notify(queue: .main) {
            completion(.success(reservoirData))
        }
    }
}

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

