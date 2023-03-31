//
//  RiverDetailView.swift
//  WW-app
//
//  Created by Tyler Martin on 3/29/23.
//

import SwiftUI
import Foundation

struct RiverDetailView: View {
    let river: River

    @State private var selectedDate = Date()
    @State private var snowpackData: SnowpackData?
    @State private var errorMessage: String?
    @State private var highTemperature: String = ""
    @State private var lowTemperature: String = ""
    @State private var reservoirData: [ReservoirInfo] = []

    private func latestReservoirStorageData(reservoirData: ReservoirData) -> StorageData? {
        return reservoirData.data.sorted { $0.date > $1.date }.first
    }
    private func fetchReservoirData(siteIDs: [Int]) {
        for siteID in river.siteIDs {
            APIManager.shared.getReservoirDetails(for: siteID) { result in
                switch result {
                case .success(let info):
                    DispatchQueue.main.async {
                        self.reservoirData.append(info)
                    }
                case .failure(let error):
                    print("Error fetching reservoir details:", error)
                }
            }
        }
    }


    var body: some View {
        VStack {
            Text(river.name)
                .font(.largeTitle)
                .padding(.top)
            
            Text(river.location)
                .font(.title2)
                .padding(.bottom)
            
            Text("High Temperature: \(highTemperature)")
                .font(.title2)
                .padding(.top)

            Text("Low Temperature: \(lowTemperature)")
                .font(.title2)
                .padding(.top)
            
            if reservoirData.isEmpty {
                ProgressView()
            } else {
                VStack {
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
            DatePicker("Select Date", selection: $selectedDate, displayedComponents: .date)
                .padding(.horizontal)
            
            Button("Get Flow Prediction") {
                fetchSnowpackData()
            }
            .padding(.vertical)
            
            if let snowpackData = snowpackData {
                VStack {
                    Text("Snow Water Equivalent: \(snowpackData.snowWaterEquivalent) inches")
                    Text("Snow Depth: \(snowpackData.snowDepth) inches")
                }
                .padding(.vertical)
            } else {
                ProgressView()
            }
            if let errorMessage = errorMessage {
                Text(errorMessage)
                    .foregroundColor(.red)
                    .padding(.vertical)
            }
            
            Spacer()
        }
        .padding(.horizontal)
        .navigationTitle(river.name)
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            fetchWeatherData()
            fetchReservoirData(siteIDs: river.siteIDs)
        }
    }
    func fetchSnowpackData() {
        APIManager.shared.getSnowpackData(stationID: river.snotelStationID) { result in
            DispatchQueue.main.async {
                switch result {
                case .success(let data):
                    self.snowpackData = data
                    self.errorMessage = nil
                case .failure(let error):
                    print("Error fetching snowpack data: \(error.localizedDescription)")
                    self.errorMessage = "Failed to fetch snowpack data. Error: \(error.localizedDescription)"
                }
            }
        }
    }
    func fetchWeatherData() {
            let apiKey = "74e6dd2bb94e19344d2ce618bdb33147" // Replace with your actual OpenWeatherMap API key
            let coordinates: (latitude: Double, longitude: Double)

            if river.name == "Arkansas River by the Numbers" {
                coordinates = (latitude: 38.5577, longitude: -106.2031)
            } else if river.name == "Upper Colorado River" {
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


struct RiverDetailView_Previews: PreviewProvider {
    static var previews: some View {
        RiverDetailView(river: River(id: 1, name: "Upper Colorado River", location: "Colorado", snotelStationID: "303", siteIDs: [1999, 2000, 2005]))
    }
}
