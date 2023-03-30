//
//  WW_appApp.swift
//  WW-app
//
//  Created by Tyler Martin on 3/29/23.
//

import SwiftUI

@main
struct WW_appApp: App {
    let persistenceController = PersistenceController.shared

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(\.managedObjectContext, persistenceController.container.viewContext)
        }
    }
}
