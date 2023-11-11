//  RiverListView.swift
//  WW-app
//
//  Created by Tyler Martin on 3/29/23.
//

import SwiftUI

struct RiverListView: View {
    @EnvironmentObject var riverDataModel: RiverDataModel
    @State private var searchTerm: String = ""

    // This computed property will filter the rivers based on the search term.
    var filteredRivers: [USGSRiverData] {
        if searchTerm.isEmpty {
            return riverDataModel.rivers
        } else {
            return riverDataModel.rivers.filter { $0.stationName.lowercased().contains(searchTerm.lowercased()) }
        }
    }

    var body: some View {
        NavigationView {
            VStack {
                // Search bar
                TextField("Search by station name...", text: $searchTerm)
                    .padding(10)
                    .background(Color(.systemGray6))
                    .cornerRadius(8)
                    .padding(.horizontal)

                List {
                    ForEach(filteredRivers.indices, id: \.self) { index in
                        let splitName = splitStationName(filteredRivers[index].stationName)
                        NavigationLink(destination: RiverDetailView(river: filteredRivers[index])) {
                            HStack {
                                VStack(alignment: .leading) {
                                    Text(splitName.0) // Part 1 of the title
                                        .font(.headline)
                                    // Conditionally display part 2 and part 3
                                    if !splitName.2.isEmpty {
                                        Text(splitName.1) // Part 2 of the title
                                            .font(.subheadline)
                                        Text(splitName.2) // Part 3 of the title, if it exists
                                            .font(.subheadline)
                                    } else {
                                        // If part 3 is empty, display part 2 in its place
                                        Text(splitName.1)
                                            .font(.subheadline)
                                    }
                                }
                                Spacer()
                            }
                        }
                        .swipeActions {
                            Button(action: {
                                riverDataModel.toggleFavorite(at: index) // Ensure this works with filtered rivers
                            }) {
                                Label(filteredRivers[index].isFavorite ? "Unfavorite" : "Favorite", systemImage: filteredRivers[index].isFavorite ? "star.slash.fill" : "star.fill")
                            }
                            .tint(filteredRivers[index].isFavorite ? .gray : .yellow)
                        }
                    }
                }
            }
            .navigationBarTitle("Rivers")
        }
    }
    // Helper function to split the station name.
    func splitStationName(_ stationName: String) -> (String, String, String) {
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

