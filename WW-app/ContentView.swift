//
//  ContentView.swift
//  WW-app
//
//  Created by Tyler Martin on 3/29/23.
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        NavigationView {
            RiverListView(rivers: [
                RiverData(id: 1, name: "Upper Colorado River", location: "Colorado", snotelStationID: "1120", usgsSiteID: 9058000, reservoirSiteIDs: [1999, 2000, 2005],lastFetchedDate: Date()),
                RiverData(id: 2, name: "Arkansas River by the Numbers", location: "Colorado", snotelStationID: "369", usgsSiteID: 7087050, reservoirSiteIDs: [100163, 100275],lastFetchedDate: Date())
            ])
        }
    }
}


struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

