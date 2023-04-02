//
//  Backend.swift
//  WW-app
//
//  Created by Tyler Martin on 4/1/23.
//
import UIKit
import Amplify
import AWSCognitoAuthPlugin
import AWSAPIPlugin
import Combine
import SwiftUI

class UserData: ObservableObject {
    static let shared = UserData()
    
    @Published var isSignedIn: Bool = false

    private init() { }
}

class Backend: ObservableObject {
    static let shared = Backend()
    static func initialize() -> Backend {
        return .shared
    }
    @Published var isSignedIn: Bool = false
    // signin with Cognito web user interface
    public func signIn() async {
        do {
            let signInResult = try await Amplify.Auth.signInWithWebUI(presentationAnchor: UIApplication.shared.windows.first!)
            if signInResult.isSignedIn {
                print("Sign in succeeded")
            }
        } catch let error as AuthError {
            print("Sign in failed \(error)")
        } catch {
            print("Unexpected error: \(error)")
        }
    }
    
    // signout
    public func signOut() async {
        let result = await Amplify.Auth.signOut()
        guard let signOutResult = result as? AWSCognitoSignOutResult
    else {
            print("Signout failed")
            return
        }

        switch signOutResult {
        case .complete:
        print("Successfully signed out")

        case let .partial(revokeTokenError, globalSignOutError, hostedUIError):
            if let hostedUIError = hostedUIError {
                print("HostedUI error: \(String(describing: hostedUIError))")
            }

            if let globalSignOutError = globalSignOutError {
                print("GlobalSignOut error: \(String(describing: globalSignOutError))")
            }

            if let revokeTokenError = revokeTokenError {
                print("Revoke token error: \(String(describing: revokeTokenError))")
            }
        case .failed(let error):
            // Sign Out failed with an exception, leaving the user signed in.
            print("SignOut failed with \(error)")
        }
    }
    
    func updateUserData(withSignInStatus status : Bool) async {
        await MainActor.run {
            let userData : UserData = .shared
            userData.isSignedIn = status
        }
    }
    
    private init() {
        // in private init() function
        // listen to auth events.
        // see https://github.com/aws-amplify/amplify-swift/blob/main/Amplify/Categories/Auth/Models/AuthEventName.swift
        // Add necessary plugins before configuring Amplify
        do {
           try Amplify.add(plugin: AWSCognitoAuthPlugin())
//           try Amplify.add(plugin: AWSAPIPlugin(modelRegistration: AmplifyModels()))
           try Amplify.configure()
           print("Initialized Amplify")
        } catch {
           print("Could not initialize Amplify: \(error)")
        }

        
        _ = Amplify.Hub.listen(to: .auth) { (payload) in

            switch payload.eventName {

            case HubPayload.EventName.Auth.signedIn:
                print("==HUB== User signed In, update UI")
                Task {
                    await self.updateUserData(withSignInStatus: true)
                }

            case HubPayload.EventName.Auth.signedOut:
                print("==HUB== User signed Out, update UI")
                Task {
                    await self.updateUserData(withSignInStatus: false)
                }

            case HubPayload.EventName.Auth.sessionExpired:
                print("==HUB== Session expired, show sign in UI")
                Task {
                    await self.updateUserData(withSignInStatus: false)
                }

            default:
                //print("==HUB== \(payload)")
                break
            }
        }
        
        // let's check if user is signedIn or not
        Task {
            do {
                let session = try await Amplify.Auth.fetchAuthSession()
                
                // let's update UserData and the UI
                await self.updateUserData(withSignInStatus: session.isSignedIn)
            } catch {
                print("Fetch auth session failed with error - \(error)")
            }
        }
    }
}
