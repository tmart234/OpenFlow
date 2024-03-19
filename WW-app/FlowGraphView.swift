//
//  FlowGraphView.swift
//  WW-app
//
//  Created by Tyler Martin on 3/18/24.
//
// ML Prediction Graph used in RiverDetailView

import Foundation
import SwiftUI
import CoreML
import Charts

struct FlowGraphView: View {
    let river: USGSRiverData
    @EnvironmentObject var sharedModelData: SharedModelData
    @State private var flowData: [(date: Date, flow: Double)] = []
    @State private var predictedFlowData: [(date: Date, flow: Double)] = []
    
    var body: some View {
        VStack {
            if let model = sharedModelData.compiledModel {
                RiverFlowGraphView(flowData: flowData, predictedFlowData: predictedFlowData)
                    .onAppear {
                        prepareInputDataAndPredict(with: model)
                    }
            } else {
                Text("Model not loaded")
            }
        }
    }
    
      private func prepareInputDataAndPredict(with model: MLModel) {
          // Prepare the input data for prediction
          guard let inputFeatures = prepareInputData() else {
              print("Failed to prepare input data")
              return
          }
          
          // Make predictions using the loaded model
          guard let output = try? model.prediction(from: inputFeatures) else {
              print("Failed to make predictions")
              return
          }
          
          // Process the model output and update the predicted flow data
          // Assuming the model output is a dictionary with "predictedFlow" key
          guard let predictedFlow = output.featureValue(for: "predictedFlow") else {
              print("Failed to get predicted flow data")
              return
          }
          
          // Convert MLMultiArray to [(date: Date, flow: Double)]
          if let predictedFlowArray = predictedFlow.multiArrayValue?.doubleArrayFromMLMultiArray() {
              let dateFormatter = DateFormatter()
              dateFormatter.dateFormat = "yyyy-MM-dd"
              let startDate = dateFormatter.date(from: "2024-03-18") ?? Date()
              
              let predictedFlowData = predictedFlowArray.enumerated().map { (index, flow) in
                  let date = Calendar.current.date(byAdding: .day, value: index, to: startDate) ?? Date()
                  return (date: date, flow: flow)
              }
              
              DispatchQueue.main.async {
                  self.predictedFlowData = predictedFlowData
              }
          } else {
              print("Failed to convert predicted flow data to [(date: Date, flow: Double)]")
          }
      }
      
    private func prepareInputData() -> MLFeatureProvider? {
        // Prepare the input data for the model based on the training script
        
        // Get the USGS station ID
        let stationID = river.siteNumber
        
        // Get 14 days of future temperature predictions (max and min)
        guard let futureTempData = getFutureTemperatureData(forDays: 14) else {
            print("Failed to get future temperature data")
            return nil
        }
        
        // Get 60 days of historical flow data (max and min)
        guard let historicalFlowData = getHistoricalFlowData(forDays: 60) else {
            print("Failed to get historical flow data")
            return nil
        }
        
        // Get the normalized date as a fraction
        let normalizedDate = getNormalizedDate()
        
        // Create the MLMultiArray for future temperature data
        guard let futureTempMultiArray = try? MLMultiArray(shape: [14, 2], dataType: .double) else {
            print("Failed to create MLMultiArray for future temperature data")
            return nil
        }
        
        // Create the MLMultiArray for historical flow data
        guard let historicalFlowMultiArray = try? MLMultiArray(shape: [60, 2], dataType: .double) else {
            print("Failed to create MLMultiArray for historical flow data")
            return nil
        }
        
        // Set the values for future temperature data
        for (rowIndex, tempData) in futureTempData.enumerated() {
            for (colIndex, value) in tempData.enumerated() {
                futureTempMultiArray[[rowIndex, colIndex] as [NSNumber]] = value
            }
        }

        // Set the values for historical flow data
        for (rowIndex, flowData) in historicalFlowData.enumerated() {
            for (colIndex, value) in flowData.enumerated() {
                historicalFlowMultiArray[[rowIndex, colIndex] as [NSNumber]] = value
            }
        }
        
        // Create the input features dictionary
        let inputFeatures: [String: Any] = [
            "future_temp_data": futureTempMultiArray,
            "historical_flow_data": historicalFlowMultiArray,
            "station_id": stationIDMultiArray,
            "date_normalized": normalizedDate
        ]
        
        // Create an MLDictionaryFeatureProvider with the input features
        return try? MLDictionaryFeatureProvider(dictionary: inputFeatures)
    }
      
    private func getFutureTemperatureData(forDays days: Int) -> [[Double]]? {
        var futureTempData: [[Double]]?
        let semaphore = DispatchSemaphore(value: 0)
        
        CoordinatesFetcher.getCoordinates(siteNumber: river.siteNumber) { result in
            switch result {
            case .success(let coordinates):
                let lat = coordinates.latitude
                let lon = coordinates.longitude
                
                let apiKey = "74e6dd2bb94e19344d2ce618bdb33147" // Replace with your actual OpenWeatherMap API key
                
                APIManager().getWeatherData(latitude: lat, longitude: lon, apiKey: apiKey) { result in
                    switch result {
                    case .success(let weatherData):
                        futureTempData = weatherData.futureTempData
                    case .failure(let error):
                        print("Error fetching weather data: \(error)")
                    }
                    semaphore.signal()
                }
            case .failure(let error):
                print("Error fetching coordinates: \(error)")
                semaphore.signal()
            }
        }
        
        _ = semaphore.wait(timeout: .distantFuture)
        return futureTempData
    }
      
      private func getHistoricalFlowData(forDays days: Int) -> [[Double]]? {
          // Implement the logic to fetch historical flow data (max and min) for the specified number of days
          // Return the data as an array of [Double] arrays, where each inner array represents a day's flow data
          // Example: [[maxFlow1, minFlow1], [maxFlow2, minFlow2], ...]
          // Return nil if the data is not available
          return nil // Placeholder, replace with your actual implementation
      }
      
    private func getNormalizedDate() -> Double {
        let currentDate = Date()
        let normalizedDate = normalizeDate(currentDate)
        return normalizedDate
    }
  }

extension MLMultiArray {
    func doubleArrayFromMLMultiArray() -> [Double]? {
        guard let pointer = try? UnsafeBufferPointer<Double>(self) else {
            return nil
        }
        return Array(pointer)
    }
}

struct RiverFlowGraphView: View {
    let flowData: [(date: Date, flow: Double)]
    let predictedFlowData: [(date: Date, flow: Double)]
    
    var body: some View {
        // Use a charting library like SwiftUICharts or create a custom graph view
        // to display the flow data and predicted flow data
        // For simplicity, we'll just display the data as text for now
        VStack {
            Text("Flow Data: \(flowData.description)")
            Text("Predicted Flow Data: \(predictedFlowData.description)")
        }
    }
}

private func normalizeDate(_ date: Date) -> Double {
    let calendar = Calendar.current
    let dayOfYear = Double(calendar.ordinality(of: .day, in: .year, for: date) ?? 1)
    let isLeapYear = calendar.range(of: .day, in: .year, for: date)?.count == 366
    let yearFraction = (dayOfYear - 1) / (isLeapYear ? 366.0 : 365.0)
    return yearFraction
}

private func normalizeStationID(_ stationID: String) -> [Double] {
    // Create a dictionary to map station IDs to one-hot encoded vectors
    let stationIDMap = [
        "station_1": [1.0, 0.0, 0.0],
        "station_2": [0.0, 1.0, 0.0],
        "station_3": [0.0, 0.0, 1.0]
        // Add more station IDs as needed
    ]
    
    return stationIDMap[stationID] ?? Array(repeating: 0.0, count: stationIDMap.count)
}
