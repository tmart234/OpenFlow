//
//  FavoriteView.swift
//  WW-app
//
//  Created by Tyler Martin on 10/29/23.
//

import Foundation
import SwiftUI

struct FavoriteView: View {
    @EnvironmentObject var riverDataModel: RiverDataModel
    
    var favoriteRivers: [USGSRiverData] {
        return riverDataModel.rivers.filter { $0.isFavorite }
    }

    var body: some View {
        NavigationView {
            List {
                ForEach(favoriteRivers) { river in
                    NavigationLink(destination: RiverDetailView(river: river)) {
                        HStack {
                            VStack(alignment: .leading) {
                                Text(river.stationName)
                                    .font(.headline)
                                Text("\(river.siteNumber)")
                                    .font(.subheadline)
                            }
                            Spacer()
                        }
                    }
                }
            }
            .navigationBarTitle("Favorites")
            .onAppear(perform: riverDataModel.loadFavoriteRivers)
        }
    }
}
