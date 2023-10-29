//
//  GenderPicker.swift
//  SoberSense
//
//  Created by nick on 29/10/2023.
//

import SwiftUI

struct TimePicker: View {
    @Binding var selectedHour: Int
    @Binding var selectedMinute: Int

    var body: some View {
        HStack {
            Text("Time since the first drink")
                .frame(maxWidth: 150)
                .multilineTextAlignment(.center)
            
            Spacer()
            
            VStack (spacing: 0.1){
                Picker("Number", selection: $selectedHour) {
                    ForEach(0 ..< 11, id: \.self) { number in
                        Text("\(number) hours")
                    }
                }
                .pickerStyle(DefaultPickerStyle())
                
                Picker("Number", selection: $selectedMinute) {
                    ForEach(0 ..< 6, id: \.self) { number in
                        Text("\(number * 10) minutes")
                    }
                }
                .pickerStyle(DefaultPickerStyle())
            }
            
            Spacer()
        
        }
    }
}
