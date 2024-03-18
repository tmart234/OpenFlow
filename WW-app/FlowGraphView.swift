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
                        makePredictions(with: model)
                    }
            } else {
                Text("Model not loaded")
            }
        }
    }
    
    private func makePredictions(with model: MLModel) {
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
        // Prepare the input data for the model
        // This will depend on the specific requirements of your model
        // Create and return an MLFeatureProvider with the appropriate input features
        // Example:
        // let inputFeatures = try? MLDictionaryFeatureProvider(dictionary: ["input": MLMultiArray(...)])
        // return inputFeatures
        return nil // Placeholder, replace with your actual implementation
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
        Chart {
            ForEach(flowData, id: \.date) { dataPoint in
                LineMark(
                    x: .value("Date", dataPoint.date),
                    y: .value("Flow", dataPoint.flow)
                )
                .foregroundStyle(Color.blue)
            }
            
            ForEach(predictedFlowData, id: \.date) { dataPoint in
                LineMark(
                    x: .value("Date", dataPoint.date),
                    y: .value("Predicted Flow", dataPoint.flow)
                )
                .foregroundStyle(Color.red)
            }
        }
        .chartXAxis {
            AxisMarks(values: .automatic) { _ in
                AxisGridLine()
                AxisTick()
                AxisValueLabel(format: .dateTime.day().month())
            }
        }
        .chartYAxis {
            AxisMarks(position: .trailing)
        }
        .chartYScale(domain: .automatic)
        .chartXScale(domain: .automatic)
        .chartPlotStyle { plotContent in
            plotContent
                .frame(height: 300)
        }
    }
}
