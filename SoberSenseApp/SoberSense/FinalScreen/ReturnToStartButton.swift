//
//  ReturnToStartButton.swift
//  SoberSense
//
//  Created by nick on 12/11/2023.
//

import SwiftUI

struct ReturnToStartButton: View {
    @Binding var dataHasBeenShared: Bool
    @Binding var isNavigationActive: Bool
    @Binding var showAlert: Bool

    var body: some View {
        HStack {
            Button("Return to start") {
                if dataHasBeenShared {
                    isNavigationActive = true
                } else {
                    showAlert = true
                }
            }
            .padding()
            .alert(isPresented: $showAlert) {
                Alert(
                    title: Text("Warning"),
                    message: Text("Have you sent your recorded data?"),
                    primaryButton: .default(Text("Yes")) {
                        dataHasBeenShared = true
                    },
                    secondaryButton: .cancel(Text("No"))
                )
            }
        }
    }
}
