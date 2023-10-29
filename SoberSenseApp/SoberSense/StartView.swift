//
//  StartView.swift
//  SoberSense
//
//  Created by nick on 22/09/2023.
//

import SwiftUI

struct StartView: View {
    
    @State private var isExplanationViewVisible = true
    
    @State private var selectedHour = 0
    @State private var selectedMinute = 0
    
    @State private var weight = 0
    @State private var isWeightInputValid = false
    
    @State private var unitsDrunk: Double = 0.0
    @State private var isUnitsInputValid = false
    @FocusState private var isEditingUnits: Bool
    
    @State private var gameAttempt = TestTrack()
    
    @State private var selectedGender: String = "Male" // Initial selection
    
    @State private var showDetails = false
    
    var body: some View {
        NavigationView{
    
            VStack{
                
                ExplanationView()
                    .frame(maxWidth: 350)
                    .padding(8)
                
                UnitsPicker(selectedValue: $unitsDrunk)
                    .padding(.horizontal, 80)
                
                AlcoholUnitTable()
                    .padding()
                
                
                // Recording sex
                HStack {
                    Picker("Select Gender", selection: $selectedGender) {
                        Text("Male").tag("Male")
                        Text("Female").tag("Female")
                    }
                    .pickerStyle(SegmentedPickerStyle())
                }
                .frame(width: 200)
                .padding()
                
                
                // Recording weight
                HStack(spacing: 2){
                    Text("Approximate weight:")
                    
                    let weightValues = [0] + Array(30...150)
                    
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
                
    
                VStack {
                    
                    HStack {
                        
                        Text("Time since first drink:")
                        
                        VStack (spacing: 0){
                            
                            Picker("Number", selection: $selectedHour) {
                                ForEach(0 ..< 11, id: \.self) { number in
                                    Text("\(number) hours")
                                }
                            }
                            .pickerStyle(DefaultPickerStyle())
                            
                            
                            Picker("Number", selection: $selectedMinute) {
                                ForEach(0 ..< 6, id: \.self) { number in
                                    Text("\(number*10) minutes")
                                }
                            }
                            .pickerStyle(DefaultPickerStyle())
                            
                            
                            
                        }
                        
                    }
                   
                    
                    Text("If you have drunk alcohol, please wait at least half an hour from your first drink to record data.")
                                               .font(.system(size: 14)) // Adjust the size as needed
                                               .foregroundColor(.gray)
                                               .multilineTextAlignment(.center)
                    
                    
                }
            
                
                NavigationLink("Next", destination: AnimationView(gameAttempt: $gameAttempt))
                    .padding(20)
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
    
                
            }
            
        }
        .navigationBarBackButtonHidden(true)

        .onChange(of: unitsDrunk) { newInput in
                gameAttempt.unitsDrunk = Float(newInput)
        }
        
        .onChange(of: selectedGender) { newGender in
            gameAttempt.gender = newGender
            }
        
        .onChange(of: weight) { newWeight in
            if newWeight == 0 {
                    isWeightInputValid = false
                } else {
                    isWeightInputValid = true
                    gameAttempt.weight = newWeight
                }
            }
        
        .onChange(of: selectedHour) { newHour in
            gameAttempt.timeHours = newHour
            }
        
        .onChange(of: selectedMinute) { newMinute in
            gameAttempt.timeMinutes = newMinute
            }
        
        .onAppear {
            // Reinitialize gameAttempt instance with a new UUID once the current game has finished, and this view appears again
            gameAttempt = TestTrack()
            gameAttempt.gender = "Male"
        }
    }
}

struct StartView_previews: PreviewProvider
{
    static var previews: some View {
        StartView()
    }
}
