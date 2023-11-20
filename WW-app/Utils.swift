//
//  Utils.swift
//  WW-app
//
//  Created by Tyler Martin on 11/19/23.
//

import Foundation

class Utility {
    static func splitStationName(_ stationName: String) -> (String, String, String) {
        let splitKeywords = [" NEAR ", " AT ", " ABOVE ", " ABV ", " BELOW ", " BLW ", " NR ", " AB ", " BL "]
        var parts = [String]()
        
        // Helper function to split the string by the first occurrence of any keyword and remove it.
        func splitByKeyword(_ string: String) -> [String] {
            for keyword in splitKeywords {
                if let range = string.range(of: keyword) {
                    let partBeforeKeyword = String(string[..<range.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
                    let partAfterKeyword = String(string[range.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
                    return [partBeforeKeyword, partAfterKeyword]
                }
            }
            return [string]
        }
        
        // First split to determine the parts.
        parts = splitByKeyword(stationName)
        
        // If the first part contains any keyword, split it again.
        if parts.count > 1 && splitKeywords.contains(where: parts[0].contains) {
            parts = splitByKeyword(parts[0]) + [parts[1]]
        }
        
        // Assign parts to variables, filling in empty strings if there are less than 3 parts.
        let part1 = parts.count > 0 ? parts[0] : ""
        let part2 = parts.count > 1 ? parts[1] : ""
        let part3 = parts.count > 2 ? parts[2] : ""
        
        return (part1, part2, part3)
    }
}
