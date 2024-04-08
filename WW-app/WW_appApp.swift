//
//  WW_appApp.swift
//  WW-app
//
//  Created by Tyler Martin on 3/29/23.
//

import SwiftUI
import Amplify
import AWSPluginsCore
import AWSCognitoAuthPlugin
import AWSAPIPlugin
import Zip
import CoreML

@main
struct WW_appApp: App {
    @StateObject private var sharedModelData = SharedModelData()
    @State private var isModelLoaded = false
    // Initialize your model once here
    let riverDataModel = RiverDataModel()
    
    init() {
        // Configure Amplify here if needed
    }

    var body: some Scene {
        WindowGroup {
            TabView {
                MLListView()
                    .environmentObject(riverDataModel)
                    .environmentObject(sharedModelData) // Use sharedModelData directly
                    .tabItem {
                        Label("Forecast", systemImage: "waveform.path.ecg")
                    }
                RiverListView()
                    .environmentObject(riverDataModel)
                    .tabItem {
                        Label("Rivers", systemImage: "waveform.path.ecg")
                    }
                FavoriteView()
                    .environmentObject(riverDataModel)
                    .tabItem {
                        Label("Favorites", systemImage: "star.fill")
                    }
                ProfileView() // Add the ProfileView as a new tab
                      .tabItem {
                          Label("Profile", systemImage: "person.crop.circle")
                      }
            }
            .onAppear {
                fetchLatestMLModel()
            }
        }
    }
    
    private func fetchLatestMLModel() {
        let releasesUrl = "https://api.github.com/repos/tmart234/OpenFlowColorado/releases/latest"

        guard let url = URL(string: releasesUrl) else {
            print("Error: Invalid releases URL")
            return
        }

        let task = URLSession.shared.dataTask(with: url) { data, response, error in
            if let error = error {
                print("Error fetching releases: \(error.localizedDescription)")
                return
            }

            guard let httpResponse = response as? HTTPURLResponse, (200...299).contains(httpResponse.statusCode) else {
                print("Error: Invalid response from server")
                return
            }

            guard let data = data else {
                print("Error: No data received from server")
                return
            }

            do {
                let decodedResponse = try JSONDecoder().decode(GitHubRelease.self, from: data)
                if let asset = decodedResponse.assets.first(where: { $0.name.hasPrefix("lstm_model_") && $0.name.hasSuffix(".mlpackage.zip") }) {
                    DispatchQueue.main.async {
                        self.downloadMLModel(from: asset.browserDownloadUrl)
                    }
                } else {
                    print("Error: No suitable ML model file found in the release")
                }
            } catch {
                print("Error decoding response: \(error.localizedDescription)")
            }
        }
        task.resume()
    }
    
    private func unzipAndLoadModel(at url: URL) {
        let fileManager = FileManager.default
        let documentsDirectory = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
        let destinationUrl = documentsDirectory.appendingPathComponent("unzippedModel")
        
        // Delete the destination directory if it already exists
        if fileManager.fileExists(atPath: destinationUrl.path) {
            do {
                try fileManager.removeItem(at: destinationUrl)
            } catch {
                print("Error deleting existing destination directory: \(error.localizedDescription)")
                return
            }
        }
        
        // Unzip the outer mlpackage file
        do {
            try Zip.unzipFile(url, destination: destinationUrl, overwrite: true, password: nil, fileOutputHandler: { unzippedFile in
                print("Unzipped file: \(unzippedFile)")
            })
            print("Unzipped outer mlpackage successfully")
        } catch {
            print("Error unzipping the outer mlpackage file: \(error.localizedDescription)")
            return
        }
        
        // Find the inner mlpackage file in the "model" directory
        let modelDirectoryPath = destinationUrl.appendingPathComponent("model")
        let innerMlpackagePath = modelDirectoryPath.appendingPathComponent("lstm_model_0.1.6.mlpackage")
        
        // Construct the path to the .mlmodel file within the inner mlpackage
        let modelFilePath = innerMlpackagePath.appendingPathComponent("Data/com.apple.CoreML/model.mlmodel")
        print("Model file path: \(modelFilePath)")
        
        // Check if the .mlmodel file exists
        if !fileManager.fileExists(atPath: modelFilePath.path) {
            print("Error: .mlmodel file not found at the specified path")
            print("Model file path: \(modelFilePath)")
            return
        }
        do {
            DispatchQueue.global().async {
                do {
                    let compiledModelURL = try MLModel.compileModel(at: modelFilePath)
                    let model = try MLModel(contentsOf: compiledModelURL)
                    DispatchQueue.main.async {
                        sharedModelData.compiledModel = model
                        sharedModelData.isModelLoaded = true
                    }
                } catch {
                    print("Error compiling or loading the model: \(error.localizedDescription)")
                }
            }
        }
    }
    
    private func downloadMLModel(from urlString: String) {
        guard let url = URL(string: urlString) else {
            print("Invalid URL")
            return
        }

        let task = URLSession.shared.downloadTask(with: url) { tempLocalUrl, response, error in
            if let error = error {
                print("Download failed: \(error.localizedDescription)")
                return
            }

            guard let tempLocalUrl = tempLocalUrl else {
                print("No temporary local URL for downloaded ML model")
                return
            }

            let fileManager = FileManager.default
            do {
                let documentsDirectory = try fileManager.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
                let destinationUrl = documentsDirectory.appendingPathComponent(url.lastPathComponent)

                // Check if file exists and remove it
                if fileManager.fileExists(atPath: destinationUrl.path) {
                    try fileManager.removeItem(at: destinationUrl)
                }

                try fileManager.moveItem(at: tempLocalUrl, to: destinationUrl)
                print("Moved ML model to: \(destinationUrl)")
                DispatchQueue.main.async {
                    self.unzipAndLoadModel(at: destinationUrl)
                }
            } catch {
                print("File move or unzip failed: \(error)")
            }
        }
        task.resume()
    }
}

class SharedModelData: ObservableObject {
    @Published var compiledModel: MLModel?
    @Published var isModelLoaded: Bool = false
}

struct GitHubRelease: Codable {
    var assets: [GitHubAsset]
}

struct GitHubAsset: Codable {
    var browserDownloadUrl: String
    var name: String

    enum CodingKeys: String, CodingKey {
        case browserDownloadUrl = "browser_download_url"
        case name
    }
}
