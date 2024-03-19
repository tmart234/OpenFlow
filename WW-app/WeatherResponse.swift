//
//  WeatherResponse.swift
//  WW-app
//
//  Created by Tyler Martin on 3/30/23.
//

import Foundation

struct MainWeather: Decodable {
    let temp_max: Double
    let temp_min: Double
}

struct WeatherData {
    let highTemperature: Double
    let lowTemperature: Double
    let futureTempData: [[Double]]
}

struct WeatherResponse: Decodable {
    let main: MainData
    let forecast: ForecastData
    
    struct MainData: Decodable {
        let temp_max: Double
        let temp_min: Double
    }
    
    struct ForecastData: Decodable {
        let list: [ForecastItem]
    }
    
    struct ForecastItem: Decodable {
        let main: MainData
        let dt: Int
    }
}
