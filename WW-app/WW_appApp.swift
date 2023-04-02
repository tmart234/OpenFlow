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
    init() {
        // initialize Amplify
        _ = Backend.initialize()
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(Backend.shared)
        }
    }
}
