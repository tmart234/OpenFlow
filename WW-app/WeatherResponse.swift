//
//  WeatherResponse.swift
//  WW-app
//
//  Created by Tyler Martin on 3/30/23.
//

import Foundation

struct WeatherResponse: Decodable {
    let main: MainWeather
}

struct MainWeather: Decodable {
    let temp_max: Double
    let temp_min: Double
}

struct WeatherData {
    let highTemperature: Double
    let lowTemperature: Double
}
