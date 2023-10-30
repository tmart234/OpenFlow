//
//  RiverListView.swift
//  WW-app
//
//  Created by Tyler Martin on 3/29/23.
//

import SwiftUI
import Amplify

struct RiverListView: View {
    let rivers: [RiverData]
    @EnvironmentObject private var backend: Backend
    
    var body: some View {
        VStack {
            List(rivers) { riverData in
                NavigationLink(destination: RiverDetailView(river: riverData).environmentObject(backend)) {
                    VStack(alignment: .leading) {
                        Text(riverData.name)
                            .font(.headline)
                        Text(riverData.location)
                            .font(.subheadline)
                    }
                }
            }
            .navigationTitle("Rivers")
            if backend.isSignedIn {
                SignOutButton(action: { Task { await Backend.shared.signOut() } })
            } else {
                SignInButton(action: { Task { await Backend.shared.signIn() } })
            }
        }
        .onAppear {
            Task {
                do {
                    let session = try await Amplify.Auth.fetchAuthSession()
                    backend.isSignedIn = session.isSignedIn
                } catch {
                    print("Fetch auth session failed with error - \(error)")
                }
            }
        }
    }
}
    


