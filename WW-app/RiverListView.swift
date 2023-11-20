//  RiverListView.swift
//  WW-app
//
//  Created by Tyler Martin on 3/29/23.
//

import SwiftUI
import Foundation

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
                        let splitName = Utility.splitStationName(filteredRivers[index].stationName) // Updated to use Utility class
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
}

