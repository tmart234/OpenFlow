//
//  MLListView.swift
//  WW-app
//
//  Created by Tyler Martin on 11/19/23.
//

import SwiftUI
import Combine
import Zip

struct MLListView: View {
    @EnvironmentObject var riverDataModel: RiverDataModel
    @State private var searchTerm: String = ""
    @State private var stationIDs: [String] = []

    var body: some View {
        NavigationView {
            VStack {
                TextField("Search by station name...", text: $searchTerm)
                    .padding(10)
                    .background(Color(.systemGray6))
                    .cornerRadius(8)
                    .padding(.horizontal)

                List(filteredRivers.indices, id: \.self) { index in
                    let splitName = Utility.splitStationName(filteredRivers[index].stationName)
                    NavigationLink(destination: RiverDetailView(river: filteredRivers[index])) {
                        HStack {
                            VStack(alignment: .leading) {
                                Text(splitName.0)
                                    .font(.headline)
                                Text(splitName.1)
                                    .font(.subheadline)
                                if !splitName.2.isEmpty {
                                    Text(splitName.2)
                                        .font(.subheadline)
                                }
                            }
                            Spacer()
                        }
                    }
                }
            }
            .navigationBarTitle("ML Rivers")
            .onAppear {
                fetchLatestMLModel()
                riverDataModel.fetchMLStationIDs { fetchedIDs in
                    self.stationIDs = fetchedIDs
                }
            }
        }
    }

    var filteredRivers: [USGSRiverData] {
        riverDataModel.rivers.filter { river in
            stationIDs.contains(river.siteNumber) &&
            (searchTerm.isEmpty || river.stationName.lowercased().contains(searchTerm.lowercased()))
        }
    }

    private func fetchLatestMLModel() {
        let releasesUrl = "https://api.github.com/repos/tmart234/OpenFlowColorado/releases/latest"

        guard let url = URL(string: releasesUrl) else {
            print("Invalid URL")
            return
        }

        let task = URLSession.shared.dataTask(with: url) { data, response, error in
            if let data = data {
                do {
                    let decodedResponse = try JSONDecoder().decode(GitHubRelease.self, from: data)
                    if let asset = decodedResponse.assets.first(where: { $0.name.hasSuffix(".mlpackage.zip") }) {
                        self.downloadMLModel(from: asset.browserDownloadUrl)
                    }
                } catch {
                    print("Decoding failed: \(error)")
                    print("Response: \(String(data: data, encoding: .utf8) ?? "Invalid data")")
                }
            } else {
                print("Fetch failed: \(error?.localizedDescription ?? "Unknown error")")
            }
        }
        task.resume()
    }
    private func unzipModel(at url: URL) {
        let fileManager = FileManager.default
        do {
            let documentsDirectory = try fileManager.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
            let destinationUrl = documentsDirectory

            try Zip.unzipFile(url, destination: destinationUrl, overwrite: true, password: nil) { (progress) -> () in
                print("Unzip Progress: \(progress)")
            }

            print("Unzipped ML model to: \(destinationUrl.path)")

            // List contents of the directory to verify
            let contents = try fileManager.contentsOfDirectory(atPath: destinationUrl.path)
            print("Contents of Documents directory: \(contents)")
        } catch {
            print("An error occurred while unzipping the ML model: \(error)")
        }
    }

    private func downloadMLModel(from urlString: String) {
        guard let url = URL(string: urlString) else {
            print("Invalid URL")
            return
        }

        let task = URLSession.shared.downloadTask(with: url) { tempLocalUrl, response, error in
            if let tempLocalUrl = tempLocalUrl {
                let fileManager = FileManager.default
                do {
                    let documentsDirectory = try fileManager.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
                    let permanentUrl = documentsDirectory.appendingPathComponent(url.lastPathComponent)

                    // Check if file exists and remove it
                    if fileManager.fileExists(atPath: permanentUrl.path) {
                        try fileManager.removeItem(at: permanentUrl)
                    }

                    try fileManager.moveItem(at: tempLocalUrl, to: permanentUrl)
                    print("Moved ML model to: \(permanentUrl)")
                    self.unzipModel(at: permanentUrl)
                } catch {
                    print("File move or unzip failed: \(error)")
                }
            } else {
                print("Download failed: \(error?.localizedDescription ?? "Unknown error")")
            }
        }
        task.resume()
    }
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
