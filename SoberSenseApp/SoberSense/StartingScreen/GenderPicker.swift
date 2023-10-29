//
//  GenderPicker.swift
//  SoberSense
//
//  Created by nick on 29/10/2023.
//

import SwiftUI

struct GenderPicker: View {
    @Binding var selectedGender: String

    var body: some View {
        HStack {
            Picker("Select Gender", selection: $selectedGender) {
                Text("Male").tag("Male")
                Text("Female").tag("Female")
            }
            .pickerStyle(SegmentedPickerStyle())
        }
        .frame(width: 200)
        .padding()
    }
}

