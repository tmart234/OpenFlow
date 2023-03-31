//
//  NRCSResponse.swift
//  WW-app
//
//  Created by Tyler Martin on 3/29/23.
//

import Foundation

struct NRCSResponse: Codable {
    let report: Report
}

struct Report: Codable {
    let data: [ElementData]
}

struct ElementData: Codable {
    let metadata: Metadata
    let data: [String: String]
}

struct Metadata: Codable {
    let station: Station
}

struct Station: Codable {
    let name: String
}
