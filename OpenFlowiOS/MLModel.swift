func fetchLatestModel() {
    let urlString = "https://api.github.com/repos/tmart234/OpenFlowColorado/releases/latest"
    guard let url = URL(string: urlString) else { return }

    let task = URLSession.shared.dataTask(with: url) { data, response, error in
        if let error = error {
            print("Error fetching the latest release: \(error)")
            return
        }

        guard let data = data else { return }
        do {
            let release = try JSONDecoder().decode(GitHubRelease.self, from: data)
            if let asset = release.assets.first(where: { $0.name.contains(".mlmodel") }) {
                print("Latest model URL: \(asset.browser_download_url)")
                
                // Now download the model using the URL
                self.downloadModel(from: asset.browser_download_url)
            }
        } catch {
            print("Error decoding the release data: \(error)")
        }
    }

    task.resume()
}

func downloadModel(from urlString: String) {
    guard let url = URL(string: urlString) else { return }

    let downloadTask = URLSession.shared.downloadTask(with: url) { localURL, response, error in
        if let error = error {
            print("Error downloading the model: \(error)")
            return
        }

        guard let localURL = localURL else { return }
        // You can move the file from the localURL to a permanent location in your app's sandboxed file system
        // Remember to handle this on the main thread if you're updating the UI
    }

    downloadTask.resume()
}

struct GitHubRelease: Decodable {
    let assets: [Asset]
}

struct Asset: Decodable {
    let name: String
    let browser_download_url: String
}