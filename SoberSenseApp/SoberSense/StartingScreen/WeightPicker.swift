//
//  WeightPicker.swift
//  SoberSense
//
//  Created by nick on 29/10/2023.
//

import SwiftUI

struct WeightPicker: View {
    @Binding var weight: Int
    @Binding var isWeightInputValid: Bool

    var body: some View {
        VStack{
            HStack() {
                Text("Approximate weight:")
                    .frame(width: 150)
                    .multilineTextAlignment(.center)
    
                
                let weightValues = [0] + Array(30...150)
                
                Spacer()
                
                Picker("Number", selection: $weight) {
                    ForEach(weightValues, id: \.self) { number in
                        Text("\(number) kg")
                    }
                }
                .pickerStyle(DefaultPickerStyle())
            }
            
            Text(isWeightInputValid ? "" : "Please enter a valid weight")
                .foregroundColor(.red)
                .font(.system(size: 14))
            
        }
        
    }
}
