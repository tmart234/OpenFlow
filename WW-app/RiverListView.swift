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
        var part1 = ""
        var part2 = ""
        var part3 = ""

        // Define a mutable copy of the station name to work with.
        var mutableStationName = stationName.uppercased() // Convert to uppercase for case-insensitive comparison

        // Attempt to split the station name by each keyword.
        for keyword in splitKeywords {
            let components = mutableStationName.components(separatedBy: keyword.uppercased())
            if components.count > 1 {
                part1 = components.first!.trimmingCharacters(in: .whitespacesAndNewlines)
                
                // If any of the remaining components contain another keyword, split again.
                let remainingComponents = components.dropFirst().joined(separator: " ")
                for secondKeyword in splitKeywords {
                    let secondComponents = remainingComponents.components(separatedBy: secondKeyword.uppercased())
                    if secondComponents.count > 1 {
                        part2 = secondComponents.first!.trimmingCharacters(in: .whitespacesAndNewlines)
                        part3 = secondComponents.dropFirst().joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)
                        break
                    }
                }
                
                // If no further keywords are found in the remaining components, assign to part2.
                if part2.isEmpty {
                    part2 = remainingComponents.trimmingCharacters(in: .whitespacesAndNewlines)
                }
                break
            }
        }

        // If no keywords are found, assign the entire name to part1.
        if part1.isEmpty {
            part1 = stationName.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        
        return (part1, part2, part3)
    }
}

