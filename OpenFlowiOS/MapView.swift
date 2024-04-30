//
//  MapView.swift
//  WW-app
//
//  Created by Tyler Martin on 4/9/24.
//

import Foundation
import SwiftUI
import MapKit

struct MapView: View {
    let rivers: [RiverData]
    @State private var region = MKCoordinateRegion(center: CLLocationCoordinate2D(latitude: 39.0, longitude: -105.5), span: MKCoordinateSpan(latitudeDelta: 5.0, longitudeDelta: 5.0))
    @EnvironmentObject var riverDataModel: RiverDataModel
    @State private var showPNGImage = false
    @State private var scale: CGFloat = 1.0
    @State private var offset: CGSize = .zero
    
    var body: some View {
        ZStack {
            if showPNGImage {
                GeometryReader { geometry in
                    ScrollView([.horizontal, .vertical], showsIndicators: false) {
                        AsyncImage(url: URL(string: "https://github.com/tmart234/OpenFlowColorado/blob/main/.github/assets/colorado_swe.png?raw=true")) { image in
                            image
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .scaleEffect(scale)
                                .offset(offset)
                                .gesture(
                                    SimultaneousGesture(
                                        MagnificationGesture()
                                            .onChanged { value in
                                                DispatchQueue.main.async {
                                                    self.scale = value
                                                }
                                            }
                                            .onEnded { _ in
                                                DispatchQueue.main.async {
                                                    self.scale = max(1.0, min(self.scale, 5.0))
                                                }
                                            },
                                        DragGesture()
                                            .onChanged { value in
                                                DispatchQueue.main.async {
                                                    self.offset = CGSize(width: value.translation.width + self.offset.width, height: value.translation.height + self.offset.height)
                                                }
                                            }
                                            .onEnded { _ in
                                                DispatchQueue.main.async {
                                                    let maxX = (self.scale - 1) * geometry.size.width / 2
                                                    let maxY = (self.scale - 1) * geometry.size.height / 2
                                                    let clampedX = max(-maxX, min(maxX, self.offset.width))
                                                    let clampedY = max(-maxY, min(maxY, self.offset.height))
                                                    self.offset = CGSize(width: clampedX, height: clampedY)
                                                }
                                            }
                                    )
                                )
                                .frame(width: geometry.size.width, height: geometry.size.height)
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
                        RiverAnnotationView(river: river)
                    }
                }
                .edgesIgnoringSafeArea(.all)
            }
            
            VStack {
                HStack {
                    Spacer()
                    Button(action: {
                        withAnimation {
                            showPNGImage.toggle()
                            if !showPNGImage {
                                scale = 1.0
                                offset = .zero
                            }
                        }
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

struct RiverAnnotationView: View {
    let river: RiverData
    
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
