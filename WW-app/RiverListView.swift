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
                        NavigationLink(destination: RiverDetailView(river: filteredRivers[index])) {
                            HStack {
                                VStack(alignment: .leading) {
                                    Text(filteredRivers[index].stationName)
                                        .font(.headline)
                                    Text("\(filteredRivers[index].siteNumber)")
                                        .font(.subheadline)
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




