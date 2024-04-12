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
    @State private var showMapView = false
    // enables DWR agency for RiverDataType
    @State private var showDWRRivers = false
    
    var filteredRivers: [RiverData] {
        let rivers = riverDataModel.rivers.filter { showDWRRivers ? $0.agency == "DWR" : $0.agency == "USGS" }
        if searchTerm.isEmpty {
            return rivers
        } else {
            return rivers.filter { $0.stationName.lowercased().contains(searchTerm.lowercased()) }
        }
    }
    
    var body: some View {
        NavigationView {
            VStack {
                if showMapView {
                    MapView(rivers: filteredRivers)
                        .environmentObject(riverDataModel)
                } else {
                    // Search bar
                    TextField("Search by station name...", text: $searchTerm)
                        .padding(10)
                        .background(Color(.systemGray6))
                        .cornerRadius(8)
                        .padding(.horizontal)
                    // DWR rivers toggle
                    Toggle("Show DWR Rivers", isOn: $showDWRRivers)
                       .padding(.horizontal)
                    
                    List {
                        ForEach(filteredRivers) { river in
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
                          .swipeActions {
                              Button(action: {
                                  if let index = riverDataModel.rivers.firstIndex(where: { $0.id == river.id }) {
                                      riverDataModel.toggleFavorite(at: index)
                                  }
                              }) {
                                  Label(river.isFavorite ? "Unfavorite" : "Favorite", systemImage: river.isFavorite ? "star.slash.fill" : "star.fill")
                              }
                              .tint(river.isFavorite ? .gray : .yellow)
                          }
                        }
                    }
                }
            }
            .navigationBarTitle("Rivers")
            .navigationBarItems(trailing:
                Button(action: {
                    showMapView.toggle()
                }) {
                    Image(systemName: showMapView ? "list.bullet" : "map")
                }
            )
        }
    }
}
