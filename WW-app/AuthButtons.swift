//
//  AuthButtons.swift
//  WW-app
//
//  Created by Tyler Martin on 10/29/23.
//

import SwiftUI

struct SignInButton: View {
    var action: () -> Void
    
    var body: some View {
        Button(
            action: action,
            label: {
                HStack {
                    Image(systemName: "person.fill")
                        .scaleEffect(1.5)
                        .padding()
                    Text("Sign In")
                        .font(.largeTitle)
                }
                .padding()
                .foregroundColor(.white)
                .background(Color.green)
                .cornerRadius(30)
            }
        )
    }
}

struct SignOutButton: View {
    var action: () -> Void
    
    var body: some View {
        Button(
            action: action,
            label: { Text("Sign Out") }
        )
    }
}
