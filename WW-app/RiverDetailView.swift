//
//  RiverDetailView.swift
//  WW-app
//
//  Created by Tyler Martin on 3/29/23.
//

import SwiftUI
import Foundation
import Amplify
import CoreML
import Zip


struct RiverDetailView: View {
    let river: RiverData
    let isMLRiver: Bool
    let coordinates: Coordinates?
    @EnvironmentObject var sharedModelData: SharedModelData
    @State private var reservoirData: [ReservoirInfo] = []
    @State private var selectedDate = Date()
    @State private var snowpackData: SnowpackData?
    @State private var errorMessage: String?
    @State private var highTemperature: String = ""
    @State private var lowTemperature: String = ""
    @State private var _showAlert = false
    @State private var alertTitle: String = ""
    @State private var alertMessage: String = ""
    @State private var flowData: String = ""
    @State private var latitude: Double? = nil
    @State private var longitude: Double? = nil
    @State private var mlModel: MLModel?

    func fetchSnowpackData() {
        print("Calling fetchSnowpackData() for station ID: \(river.snotelStationID)")
        APIManager.shared.getSnowpackDataFromCSV(stationID: river.snotelStationID) { result in
            DispatchQueue.main.async {
                switch result {
                case .success(let snowpackData):
                    self.snowpackData = snowpackData
                case .failure(let error):
                    _showAlert = true
                    alertTitle = "Error fetching snowpack data"
                    alertMessage = error.localizedDescription
                }
            }
        }
    }
    func fetchWeatherData() {
        guard let lat = latitude, let lon = longitude else {
            print("Coordinates are not available.")
            return
        }

        let apiKey = "74e6dd2bb94e19344d2ce618bdb33147" // Replace with your actual OpenWeatherMap API key

        APIManager().getWeatherData(latitude: lat, longitude: lon, apiKey: apiKey) { result in
            switch result {
            case .success(let weatherData):
                DispatchQueue.main.async {
                    self.highTemperature = String(format: "%.1f°F", weatherData.currentHighTemperature)
                    self.lowTemperature = String(format: "%.1f°F", weatherData.currentLowTemperature)
                }
            case .failure(let error):
                DispatchQueue.main.async {
                    print("Error fetching weather data: \(error)")
                }
            }
        }
    }

    private func fetchFlowData(for siteID: String) {
        APIManager.shared.getFlowData(usgsSiteID: siteID) { result in
            switch result {
            case .success(let flowData):
                DispatchQueue.main.async {
                    self.flowData = flowData
                }
            case .failure(let error):
                print("Error fetching flow data:", error)
            }
        }
    }

    var body: some View {
        VStack {
            if isMLRiver {
                FlowGraphView(river: river)
            }
            if let snowpackData = snowpackData {
                Text("Nearest SWE: \(snowpackData.snowWaterEquivalent, specifier: "%.1f")in (or \(snowpackData.percentOfAverage, specifier: "%.1f")% of avg)")
            } else {
                Text("Fetching SWE data...")
            }
            Text("Current Flow: \(flowData)")
                .font(.title2)
                .padding(.top)
            Text("Daily High Temperature: \(highTemperature)")
                .font(.title2)
                .padding(.top)

            Text("Daily Low Temperature: \(lowTemperature)")
                .font(.title2)
                .padding(.top)
            
            Spacer()
                .frame(height: 20)

            if reservoirData.isEmpty {
                ProgressView()
            } else {
                VStack {
                    Text("Reservoir Data:")
                    ForEach(reservoirData) { info in
                        VStack(alignment: .leading) {
                            Text(info.reservoirName)
                                .font(.headline)
                            
                            Text("Percentage filled: \(info.percentageFilled, specifier: "%.1f")%")
                        }
                    }
                }
            }

            if let errorMessage = errorMessage {
                Text(errorMessage)
                    .foregroundColor(.red)
                    .padding(.vertical)
            }
            Text("Last updated: \(formattedDate(date: river.lastFetchedDate))")
                .font(.footnote)
                .foregroundColor(.gray)


        } // This is the corrected closing brace for the VStack
        .padding(.horizontal)
        .navigationTitle(river.stationName)
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            if isMLRiver {
                if sharedModelData.compiledModel != nil && sharedModelData.isModelLoaded {
                    print("Model loaded successfully")
                } else {
                    print("Model not loaded")
                }
            }
            ReservoirManager.shared.fetchReservoirData(siteIDs: river.reservoirSiteIDs) { result in
                 switch result {
                 case .success(let reservoirData):
                     self.reservoirData = reservoirData
                 case .failure(let error):
                     print("Error fetching reservoir data:", error)
                 }
             }

            fetchWeatherData()
            fetchSnowpackData()
            fetchFlowData(for: self.river.siteNumber)
            }
        }
    }

    func formattedDate(date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }
