//
//  WeatherResponse.swift
//  WW-app
//
//  Created by Tyler Martin on 3/30/23.
//
import Foundation

struct CurrentWeatherResponse: Decodable {
    let main: MainWeather
}

struct ForecastResponse: Decodable {
    let list: [ForecastItem]
    
    struct ForecastItem: Decodable {
        let main: MainWeather
    }
}

struct MainWeather: Decodable {
    let temp_max: Double
    let temp_min: Double
}

struct WeatherData {
    let currentHighTemperature: Double
    let currentLowTemperature: Double
    let futureTempData: [[Double]]
}
