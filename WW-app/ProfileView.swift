//
//  ProfileView.swift
//  WW-app
//
//  Created by Tyler Martin on 11/11/23.
//

import Foundation
import SwiftUI

// ProfileView is responsible for user profile settings and authentication.
struct ProfileView: View {
    @AppStorage("isDarkMode") private var isDarkMode: Bool = false // Store the theme preference
    @State private var isSignedIn: Bool = false // Store the sign-in status

    var body: some View {
        VStack {
            Toggle(isOn: $isDarkMode) {
                Text("Dark Mode")
            }
            .padding()
            .onChange(of: isDarkMode) { newValue in
                updateAppearance(basedOn: newValue)
            }
            // Conditional display based on sign-in state
            if isSignedIn {
                SignOutButton {
                    // Action to handle sign out
                    isSignedIn = false
                    // Implement the sign out functionality here
                }
            } else {
                SignInButton {
                    // Action to handle sign in
                    isSignedIn = true
                    // Implement the sign in functionality here
                }
            }
        }
        .padding()
        // Title for the navigation bar
        .navigationTitle("Profile")
        // Automatically switch to the selected color scheme
        .preferredColorScheme(isDarkMode ? .dark : .light)
    }
    private func updateAppearance(basedOn preference: Bool) {
        // Safely unwrap the window scene
        guard let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene else { return }
        
        // Apply the preference to all windows in the scene
        windowScene.windows.forEach { window in
            window.overrideUserInterfaceStyle = preference ? .dark : .light
        }
    }
}
