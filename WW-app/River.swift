//
//  River.swift
//  WW-app
//
//  Created by Tyler Martin on 3/29/23.
//

import Foundation

struct River: Identifiable {
    let id: Int
    let name: String
    let location: String
    let snotelStationID: String
    let siteIDs: [Int]
}
