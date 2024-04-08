//  RiverListView.swift
//  WW-app
//
//  Created by Tyler Martin on 3/29/23.
//

import SwiftUI
import Foundation
import MapKit

struct RiverListView: View {
    @StateObject var riverDataModel = RiverDataModel()
    @State private var searchTerm: String = ""
    @State private var showMapView = false
    // enables DWR agency for RiverDataType
    @State private var showDWRRivers = false
    @State private var riverCoordinates: [String: Coordinates] = [:]
    
    var filteredRivers: [RiverData] {
        if searchTerm.isEmpty {
            return riverDataModel.rivers
        } else {
            return riverDataModel.rivers.filter { $0.stationName.lowercased().contains(searchTerm.lowercased()) }
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
                    
                    List {
                        ForEach(filteredRivers) { river in
                          let splitName = Utility.splitStationName(river.stationName)
                          NavigationLink(destination: RiverDetailView(river: river, isMLRiver: false, coordinates: riverDataModel.riverCoordinates[river.siteNumber])) {
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
                            .onAppear {
                                CoordinatesFetcher.fetchUSGSCoordinates { result in
                                    switch result {
                                    case .success(let coordinates):
                                        riverDataModel.updateCoordinates(coordinates)
                                    case .failure(let error):
                                        print("Error fetching coordinates: \(error)")
                                    }
                                }
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

struct MapView: View {
    let rivers: [RiverData]
    @State private var region = MKCoordinateRegion(center: CLLocationCoordinate2D(latitude: 39.0, longitude: -105.5), span: MKCoordinateSpan(latitudeDelta: 5.0, longitudeDelta: 5.0))
    @EnvironmentObject var riverDataModel: RiverDataModel
    
    var body: some View {
        Map(coordinateRegion: $region, annotationItems: rivers.filter { river in
            let hasCoordinates = river.latitude != nil && river.longitude != nil
            if !hasCoordinates {
                print("River missing coordinates: \(river.stationName)")
            }
            return hasCoordinates
        }) { river in
            MapAnnotation(coordinate: CLLocationCoordinate2D(latitude: river.latitude!, longitude: river.longitude!)) {
                RiverAnnotationView(river: river, coordinates: riverDataModel.riverCoordinates[river.siteNumber])
            }
        }
        .edgesIgnoringSafeArea(.all)
    }
}

struct RiverAnnotationView: View {
    let river: RiverData
    let coordinates: Coordinates?
    
    var body: some View {
        NavigationLink(destination: RiverDetailView(river: river, isMLRiver: false, coordinates: coordinates)) {
            Image(systemName: "mappin.circle.fill")
                .foregroundColor(.blue)
                .frame(width: 44, height: 44)
                .background(Color.white)
                .clipShape(Circle())
        }
    }
}
