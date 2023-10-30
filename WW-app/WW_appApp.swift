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

@main
struct WW_appApp: App {
    var body: some Scene {
        WindowGroup {
            TabView {
                RiverListView()
                    .environmentObject(RiverDataModel())
                    .tabItem {
                        Label("Rivers", systemImage: "waveform.path.ecg")
                    }
                FavoriteView()
                    .environmentObject(RiverDataModel())
                    .tabItem {
                        Label("Favorites", systemImage: "star.fill")
                    }
            }
        }
    }
}
