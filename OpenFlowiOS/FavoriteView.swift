// FavoriteView.swift
// WW-app
//
// Created by Tyler Martin on 10/29/23.
//

import Foundation
import SwiftUI

struct FavoriteView: View {
    @EnvironmentObject var riverDataModel: RiverDataModel
    
    var favoriteRivers: [RiverData] {
        return riverDataModel.rivers.filter { $0.isFavorite }
    }
    
    var body: some View {
        NavigationView {
            List {
                ForEach(favoriteRivers) { river in
                    let splitName = Utility.splitStationName(river.stationName)
                    NavigationLink(destination: RiverDetailView(river: river, isMLRiver: false)) {
                        HStack {
                            VStack(alignment: .leading) {
                                Text(splitName.0)
                                    .font(.headline)
                                if !splitName.2.isEmpty {
                                    Text(splitName.1)
                                        .font(.subheadline)
                                    Text(splitName.2)
                                        .font(.subheadline)
                                } else {
                                    Text(splitName.1)
                                        .font(.subheadline)
                                }
                            }
                            Spacer()
                        }
                    }
                    .swipeActions(edge: .trailing, allowsFullSwipe: true) {
                        Button(action: {
                            if let index = riverDataModel.rivers.firstIndex(where: { $0.id == river.id }) {
                                riverDataModel.toggleFavorite(at: index)
                            }
                        }) {
                            Label("Unfavorite", systemImage: "star.slash.fill")
                        }
                        .tint(.gray)
                    }
                }
            }
            .navigationBarTitle("Favorites")
            .onAppear(perform: riverDataModel.loadFavoriteRivers)
        }
    }
}
