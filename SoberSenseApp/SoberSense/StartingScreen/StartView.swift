//
//  StartView.swift
//  SoberSense
//
//  Created by nick on 22/09/2023.
//

import SwiftUI

struct StartView: View {
    
    @State private var isExplanationViewVisible = true
    @State private var isShowingUnitTable = false
    
    @State private var selectedHour = 0
    @State private var selectedMinute = 0
    
    @State private var weight = 0
    @State private var isWeightInputValid = false
    
    @State private var unitsDrunk: Double = 0.0
    
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
                
                

                
                HStack {
                    Text("Units of drinks")
                    Image(systemName: "questionmark.circle.fill")
                }
                .foregroundColor(.blue)
                .onTapGesture {
                    withAnimation {
                        isShowingUnitTable.toggle()
                    }
                }
                            
                if isShowingUnitTable {
                    AlcoholUnitTable()
                        .padding()
                }
                

                WeightPicker(weight: $weight, isWeightInputValid: $isWeightInputValid)
                    .frame(width: 350)
        
        
                TimePicker(selectedHour: $selectedHour, selectedMinute: $selectedMinute)
                    .frame(width: 350)
                
                
                GenderPicker(selectedGender: $selectedGender)
                
                NavigationLink("Next", destination: AnimationView(gameAttempt: $gameAttempt))
                    .padding(10)
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                    .disabled(!isWeightInputValid)
                
                
    
                Text("If you have drunk alcohol, please wait at least half an hour from your first drink to record data.")
                       .font(.system(size: 14)) // Adjust the size as needed
                       .foregroundColor(.gray)
                       .multilineTextAlignment(.center)
                       .frame(width: 300)
                       .padding()
                
                
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
