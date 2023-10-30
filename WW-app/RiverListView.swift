//  RiverListView.swift
//  WW-app
//
//  Created by Tyler Martin on 3/29/23.
//

import SwiftUI

struct RiverListView: View {
    @EnvironmentObject var riverDataModel: RiverDataModel

    var body: some View {
        NavigationView {
            List {
                ForEach(riverDataModel.rivers.indices, id: \.self) { index in
                    NavigationLink(destination: RiverDetailView(river: riverDataModel.rivers[index])) {
                        HStack {
                            VStack(alignment: .leading) {
                                Text(riverDataModel.rivers[index].stationName)
                                    .font(.headline)
                                Text("\(riverDataModel.rivers[index].siteNumber)")
                                    .font(.subheadline)
                            }
                            Spacer()
                        }
                    }
                    .swipeActions {
                        Button(action: {
                            riverDataModel.toggleFavorite(at: index)
                        }) {
                            Label(riverDataModel.rivers[index].isFavorite ? "Unfavorite" : "Favorite", systemImage: riverDataModel.rivers[index].isFavorite ? "star.slash.fill" : "star.fill")
                        }
                        .tint(riverDataModel.rivers[index].isFavorite ? .gray : .yellow)
                    }
                }
            }
            .navigationBarTitle("Rivers")
        }
    }
}





