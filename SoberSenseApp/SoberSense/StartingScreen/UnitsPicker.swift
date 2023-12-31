//
//  IncrementalValuePicker.swift
//  SoberSense
//
//  Created by nick on 29/10/2023.
//

import SwiftUI

struct UnitsPicker: View {
    @Binding var selectedValue: Double

    var body: some View {
        VStack {

            Stepper(
                value: $selectedValue,
                in: 0.0...30.0,
                step: 0.1
            ) {
                HStack {
                    Text("Units drunk")
                    Spacer()
                    Text("\(selectedValue, specifier: "%.1f")")
                        .foregroundColor(Color.blue)
                }
            }
            
            
        }
    }
}
