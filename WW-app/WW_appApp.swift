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
    // Initialize your model once here
    let riverDataModel = RiverDataModel()
    
    init() {
        // Configure Amplify here if needed
    }

    var body: some Scene {
        WindowGroup {
            TabView {
                RiverListView()
                    .environmentObject(riverDataModel) // Pass the same instance to both views
                    .tabItem {
                        Label("Rivers", systemImage: "waveform.path.ecg")
                    }
                FavoriteView()
                    .environmentObject(riverDataModel) // Pass the same instance to both views
                    .tabItem {
                        Label("Favorites", systemImage: "star.fill")
                    }
                ProfileView() // Add the ProfileView as a new tab
                      .tabItem {
                          Label("Profile", systemImage: "person.crop.circle")
                      }
            }
        }
    }
}
