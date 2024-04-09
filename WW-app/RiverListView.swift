//  RiverListView.swift
//  WW-app
//
//  Created by Tyler Martin on 3/29/23.
//

import SwiftUI
import Foundation
import MapKit

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
                          .simultaneousGesture(TapGesture().onEnded {
                                if river.agency == "DWR" {
                                    riverDataModel.fetchDWRFlow(for: river)
                                }
                            })
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

struct MapView: View {
    let rivers: [RiverData]
    @State private var region = MKCoordinateRegion(center: CLLocationCoordinate2D(latitude: 39.0, longitude: -105.5), span: MKCoordinateSpan(latitudeDelta: 5.0, longitudeDelta: 5.0))
    @EnvironmentObject var riverDataModel: RiverDataModel
    @State private var showPNGImage = false
    
    var body: some View {
        ZStack {
            if showPNGImage {
                ScrollView([.horizontal, .vertical], showsIndicators: false) {
                    ZoomableView {
                        AsyncImage(url: URL(string: "https://github.com/tmart234/OpenFlowColorado/blob/main/.github/assets/colorado_swe.png?raw=true")) { image in
                            image
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                        } placeholder: {
                            ProgressView()
                        }
                    }
                }
                .edgesIgnoringSafeArea(.all)
            } else {
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
            
            VStack {
                HStack {
                    Spacer()
                    Button(action: {
                        showPNGImage.toggle()
                    }) {
                        Image(systemName: showPNGImage ? "map" : "photo")
                            .font(.title)
                            .foregroundColor(.blue)
                            .padding()
                            .background(Color.white)
                            .clipShape(Circle())
                            .shadow(radius: 5)
                    }
                }
                Spacer()
            }
            .padding()
        }
    }
}

struct ZoomableView<Content: View>: View {
    private var content: Content
    @State private var scale: CGFloat = 20.0 // Change the initial scale to a larger value
    
    init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }
    
    var body: some View {
        GeometryReader { geometry in
            content
                .scaleEffect(scale)
                .gesture(
                    MagnificationGesture()
                        .onChanged { value in
                            scale = value.magnitude
                        }
                )
                .frame(width: geometry.size.width, height: geometry.size.height)
        }
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
