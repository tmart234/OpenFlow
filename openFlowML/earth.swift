import Foundation

struct SoilMoistureData: Codable {
    let date: String
    let surfaceSoilMoisture: Double
    let rootZoneSoilMoisture: Double
}

class EarthEngineAppCaller {
    // Replace this with your actual Earth Engine app URL
    private let appUrl = "https://your-ee-app-url.com/soil_moisture"
    
    func getSoilMoistureData(lat: Double, lon: Double, startDate: String, endDate: String) async throws -> [SoilMoistureData] {
        var urlComponents = URLComponents(string: appUrl)!
        urlComponents.queryItems = [
            URLQueryItem(name: "lat", value: String(lat)),
            URLQueryItem(name: "lon", value: String(lon)),
            URLQueryItem(name: "start_date", value: startDate),
            URLQueryItem(name: "end_date", value: endDate)
        ]
        
        guard let url = urlComponents.url else {
            throw URLError(.badURL)
        }
        
        let (data, _) = try await URLSession.shared.data(from: url)
        
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode([SoilMoistureData].self, from: data)
    }
}

// Usage example
Task {
    do {
        let caller = EarthEngineAppCaller()
        let soilMoistureData = try await caller.getSoilMoistureData(
            lat: 39.5501,
            lon: -105.3111,
            startDate: "2022-06-01",
            endDate: "2022-08-31"
        )
        for data in soilMoistureData {
            print("Date: \(data.date), Surface: \(data.surfaceSoilMoisture), Root Zone: \(data.rootZoneSoilMoisture)")
        }
    } catch {
        print("Error: \(error)")
    }
}