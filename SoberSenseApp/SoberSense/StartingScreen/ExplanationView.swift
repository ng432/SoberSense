//
//  ExplanationView.swift
//  SoberSense
//
//  Created by nick on 26/10/2023.
//

import SwiftUI


struct ExplanationView: View {
    var body: some View {
        VStack{
            
            Text("This app is designed to collect touch data during a game, in order to train a model which can detect the user's blood alcohol concentration (BAC).")
            .multilineTextAlignment(.center)
            .font(.subheadline)
            

            Text("Please enter your details, including an estimate of many units you have drunk, if any.")
            .multilineTextAlignment(.center)
            .font(.subheadline)
            .bold()
            
            Rectangle()
                .fill(Color.blue.opacity(0.5))
                .frame(height: 2)

        }
            
    }
}


