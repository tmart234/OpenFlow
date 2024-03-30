//  RiverListView.swift
//  WW-app
//
//  Created by Tyler Martin on 3/29/23.
//

import SwiftUI
import Foundation
import MapKit

struct RiverListView: View {
    @ObservedObject var riverDataModel = RiverDataModel()
    @State private var searchTerm: String = ""
    @State private var showMapView = false
    @State private var showDWRRivers = false
    
    var filteredRivers: [USGSRiverData] {
        if searchTerm.isEmpty {
            return riverDataModel.rivers
        } else {
            return riverDataModel.rivers.filter { $0.stationName.lowercased().contains(searchTerm.lowercased()) }
        }
    }
    
    var filteredDWRRivers: [DWRRiverData] { // New computed property
        if searchTerm.isEmpty {
            return riverDataModel.dwrRivers
        } else {
            return riverDataModel.dwrRivers.filter { $0.stationName.lowercased().contains(searchTerm.lowercased()) }
        }
    }
    
    var body: some View {
        NavigationView {
            VStack {
                if showMapView {
                    MapView(rivers: filteredRivers.map { RiverDataType.usgs($0) } + filteredDWRRivers.map { RiverDataType.dwr($0) })
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
                        ForEach(showDWRRivers ? filteredDWRRivers.indices : filteredRivers.indices, id: \.self) { index in
                            let river: RiverDataType = showDWRRivers ? .dwr(filteredDWRRivers[index]) : .usgs(filteredRivers[index])
                            let splitName = Utility.splitStationName(river.stationName)
                            NavigationLink(destination: RiverDetailView(river: river, isMLRiver: false)) {
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
                                    if case .usgs(let usgsRiver) = river {
                                        let index = riverDataModel.rivers.firstIndex(where: { $0.id == usgsRiver.id })
                                        if let index = index {
                                            riverDataModel.toggleFavorite(at: index)
                                        }
                                    } else if case .dwr(let dwrRiver) = river {
                                        let index = riverDataModel.dwrRivers.firstIndex(where: { $0.id == dwrRiver.id })
                                        if let index = index {
                                            riverDataModel.toggleFavorite(at: index, isDWR: true)
                                        }
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

struct MapView: View {
    let rivers: [RiverDataType]
    @State private var region = MKCoordinateRegion(center: CLLocationCoordinate2D(latitude: 39.0, longitude: -105.5), span: MKCoordinateSpan(latitudeDelta: 5.0, longitudeDelta: 5.0))
    
    var body: some View {
        Map(coordinateRegion: $region, annotationItems: rivers.filter { river in
            let hasCoordinates = river.latitude != nil && river.longitude != nil
            if hasCoordinates {
                print("River with coordinates: \(river.stationName), latitude: \(river.latitude!), longitude: \(river.longitude!)")
            } else {
                print("River missing coordinates: \(river.stationName)")
            }
            return hasCoordinates
        }) { river in
            MapAnnotation(coordinate: CLLocationCoordinate2D(latitude: river.latitude!, longitude: river.longitude!)) {
                RiverAnnotationView(river: river)
            }
        }
        .edgesIgnoringSafeArea(.all)
    }
}

struct RiverAnnotationView: View {
    let river: RiverDataType
    
    var body: some View {
        NavigationLink(destination: RiverDetailView(river: river, isMLRiver: false)) {
            Image(systemName: "mappin.circle.fill")
                .foregroundColor(.blue)
                .frame(width: 44, height: 44)
                .background(Color.white)
                .clipShape(Circle())
        }
    }
}
