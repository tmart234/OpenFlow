import SwiftUI
import Combine
import CoreML

struct MLListView: View {
    @EnvironmentObject var sharedModelData: SharedModelData
    @EnvironmentObject var riverDataModel: RiverDataModel
    @State private var searchTerm: String = ""
    @State private var stationIDs: [String] = []
    
    var body: some View {
        VStack {
            if !sharedModelData.isModelLoaded {
                Text("Loading ML model...")
                    .font(.headline)
                    .padding()
            }
            NavigationView {
                VStack {
                    TextField("Search by station name...", text: $searchTerm)
                        .padding(10)
                        .background(Color(.systemGray6))
                        .cornerRadius(8)
                        .padding(.horizontal)
                    
                    List(filteredRivers.indices, id: \.self) { index in
                        let splitName = Utility.splitStationName(filteredRivers[index].stationName)
                        NavigationLink(destination: RiverDetailView(river: filteredRivers[index], isMLRiver: true, coordinates: riverDataModel.riverCoordinates[filteredRivers[index].siteNumber])
                            .environmentObject(sharedModelData)) {
                                HStack {
                                    VStack(alignment: .leading) {
                                        Text(splitName.0)
                                            .font(.headline)
                                        Text(splitName.1)
                                            .font(.subheadline)
                                        if !splitName.2.isEmpty {
                                            Text(splitName.2)
                                                .font(.subheadline)
                                        }
                                    }
                                    Spacer()
                                }
                            }
                    }
                }
                .navigationBarTitle("ML Rivers")
                .onAppear {
                    riverDataModel.fetchDWRRivers()
                    riverDataModel.fetchMLStationIDs { fetchedIDs in
                        self.stationIDs = fetchedIDs
                    }
                }
            }
        }
    }
    
    var filteredRivers: [RiverData] {
        riverDataModel.rivers.filter { river in
            stationIDs.contains(river.siteNumber) &&
            (searchTerm.isEmpty || river.stationName.lowercased().contains(searchTerm.lowercased()))
        }
    }
}
