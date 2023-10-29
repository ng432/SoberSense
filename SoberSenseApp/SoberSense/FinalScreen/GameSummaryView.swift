//
//  GameAttemptView.swift
//  SoberSense
//
//  Created by nick on 26/10/2023.
//

import SwiftUI

struct GameSummaryView: View {
    let gameAttempt: TestTrack
    let decimalFormatter: NumberFormatter = {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.minimumFractionDigits = 1
        formatter.maximumFractionDigits = 1
        return formatter
    }()

    var body: some View {
        
        VStack {
            
            HStack {
                Text("Units Drunk:")
                Spacer()
                Text(formatUnitsDrunk(gameAttempt.unitsDrunk))
            }

            HStack {
                Text("Gender:")
                Spacer()
                Text(gameAttempt.gender)
            }

            HStack {
                Text("Weight (kg):")
                Spacer()
                Text(String(gameAttempt.weight))
            }

            HStack {
                Text("Time Since First Drink:")
                Spacer()
                Text("\(gameAttempt.timeHours) hours \(gameAttempt.timeMinutes) minutes")
            }
        }
        .font(.system(size: 14))
        .foregroundColor(.gray)
        .multilineTextAlignment(.center)
        
    }

    private func formatUnitsDrunk(_ units: Float?) -> String {
        guard let units = units else { return "0.0" }
        return decimalFormatter.string(from: NSNumber(value: units)) ?? "0.0"
    }
}
