//
//  RiverListView.swift
//  WW-app
//
//  Created by Tyler Martin on 3/29/23.
//

import SwiftUI

struct RiverListView: View {
    let rivers: [River]
    
    var body: some View {
        List(rivers) { river in
            NavigationLink(destination: RiverDetailView(river: river)) {
                VStack(alignment: .leading) {
                    Text(river.name)
                        .font(.headline)
                    Text(river.location)
                        .font(.subheadline)
                }
            }
        }
        .navigationTitle("Rivers")
    }
}

struct RiverListView_Previews: PreviewProvider {
    static var previews: some View {
        RiverListView(rivers: [
            River(id: 1, name: "Upper Colorado River", location: "Colorado", snotelStationID: "1120", usgsSiteID: 09058000, reservoirSiteIDs: [1999, 2000, 2005]),
            River(id: 2, name: "Arkansas River by the Numbers", location: "Colorado", snotelStationID: "369", usgsSiteID: 07087050, reservoirSiteIDs: [100163, 100275])
        ])
    }
}


