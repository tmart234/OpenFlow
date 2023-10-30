//
//  RiverDetailView.swift
//  WW-app
//
//  Created by Tyler Martin on 3/29/23.
//

import SwiftUI
import Foundation
import Amplify



struct RiverDetailView: View {
    let river: USGSRiverData

    @State private var selectedDate = Date()
    @State private var snowpackData: SnowpackData?
    @State private var errorMessage: String?
    @State private var highTemperature: String = ""
    @State private var lowTemperature: String = ""
    @State private var reservoirData: [ReservoirInfo] = []
    @State private var _showAlert = false
    @State private var alertTitle: String = ""
    @State private var alertMessage: String = ""
    @State private var flowData: String = ""
    @EnvironmentObject private var backend: Backend

    func calculatePercentageFilled(current: Double, reservoirID: Int) -> Double {
        if let reservoir = ReservoirInfo.reservoirDetails[reservoirID] {
            return (current / reservoir.capacity) * 100
        } else {
            return 0.0
        }
    }

    private func latestReservoirStorageData(reservoirData: ReservoirData) -> StorageData? {
        return reservoirData.data.sorted { $0.date > $1.date }.first
    }
    
    private func fetchReservoirData(siteIDs: [Int]) {
        for siteID in river.reservoirSiteIDs {
            APIManager.shared.getReservoirDetails(for: siteID) { result in
                switch result {
                case .success(let info):
                    DispatchQueue.main.async {
                        // Find the most recent storage value
                        var currentStorage: Double = 0.0
                        var mostRecentDate: Date? = nil
                        for storageData in info.reservoirData.data {
                            if mostRecentDate == nil || storageData.date > mostRecentDate! {
                                mostRecentDate = storageData.date
                                currentStorage = storageData.storage
                            }
                        }
                        
                        let percentageFilled = calculatePercentageFilled(current: currentStorage, reservoirID: siteID)

                        let reservoirInfo = ReservoirInfo(reservoirName: info.reservoirName, reservoirData: info.reservoirData, percentageFilled: percentageFilled)
                        self.reservoirData.append(reservoirInfo)
                    }
                case .failure(let error):
                    print("Error fetching reservoir details:", error)
                }
            }
        }
    }

    private func fetchFlowData() {
        if let siteID = Int(river.siteNumber) {
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
        } else {
            print("Invalid site number:", river.siteNumber)
        }
    }


    var body: some View {
        VStack {
            
            Text("Flow: \(flowData)")
                .font(.title2)
                .padding(.top)
            
            if let snowpackData = snowpackData {
                Text("Nearest SWE: \(snowpackData.snowWaterEquivalent, specifier: "%.1f")in (or \(snowpackData.percentOfAverage, specifier: "%.1f")% of avg)")
            } else {
                Text("Fetching SWE data...")
            }
            Text("High Temperature: \(highTemperature)")
                .font(.title2)
                .padding(.top)

            Text("Low Temperature: \(lowTemperature)")
                .font(.title2)
                .padding(.top)
            
            Spacer()
                .frame(height: 20)

            if reservoirData.isEmpty {
                ProgressView()
            } else {
                VStack {
                    Text("Reservoir Data:")
                    ForEach(reservoirData, id: \.reservoirName) { info in
                        VStack(alignment: .leading) {
                            Text(info.reservoirName)
                                .font(.headline)
                            
                            if let latestStorageData = info.reservoirData.data.sorted(by: { $0.date > $1.date }).first {
                                Text("Percentage filled: \(info.percentageFilled, specifier: "%.1f")%")
                            }
                        }
                    }
                }
            }
            if backend.isSignedIn {
                DatePicker("Select Date", selection: $selectedDate, displayedComponents: .date)
                    .padding(.horizontal)

                Button("Get Flow Prediction") {
                }
                .padding(.vertical)
            } else {
                Text("Please log in to access advanced features")
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
            fetchWeatherData()
            fetchSnowpackData()
            fetchReservoirData(siteIDs: river.reservoirSiteIDs)
            fetchFlowData()
        }
    }
    func formattedDate(date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }
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
            let apiKey = "74e6dd2bb94e19344d2ce618bdb33147" // Replace with your actual OpenWeatherMap API key
            let coordinates: (latitude: Double, longitude: Double)

        if river.stationName == "Arkansas River by the Numbers" {
                coordinates = (latitude: 38.5577, longitude: -106.2031)
        } else if river.stationName == "Upper Colorado River" {
                coordinates = (latitude: 39.9827, longitude: -106.5384)
            } else {
                return
            }

            APIManager().getWeatherData(latitude: coordinates.latitude, longitude: coordinates.longitude, apiKey: apiKey) { result in
                switch result {
                case .success(let weatherData):
                    DispatchQueue.main.async {
                        self.highTemperature = String(format: "%.1f°F", weatherData.highTemperature)
                        self.lowTemperature = String(format: "%.1f°F", weatherData.lowTemperature)
                    }
                case .failure(let error):
                    DispatchQueue.main.async {
                        print("Error fetching weather data: \(error)")
                    }
                }
            }
        }
}
