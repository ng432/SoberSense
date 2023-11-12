//
//  GameOverView.swift
//  SoberSense
//
//  Created by nick on 22/09/2023.
//

import Foundation
import SwiftUI
import MessageUI

struct GameOverView: View {
    
    @State private var isNavigationActive = false
    @State private var isMailComposeViewShowing = false
    @State private var canSendMail = MFMailComposeViewController.canSendMail()
    @State private var testTrackData: Data?
    @State private var isFileWritingComplete = false
    
    @State private var showAlert = false
    @State private var dataHasBeenShared = false
    
    @Binding var gameAttempt: TestTrack
    
    init(gameAttempt: Binding<TestTrack>) {
        self._gameAttempt = gameAttempt

    }
    
    var body: some View {
        NavigationView{
            
            if isFileWritingComplete {
                
                VStack{
                    
                    VStack {
                        
                        if gameAttempt.touchData.isEmpty
                        {
                            
                            Text("No touch data was recorded. \n Please try again.")
                                .multilineTextAlignment(.center)
                                .onAppear{
                                    // necessary so return back can function
                                    dataHasBeenShared = true
                                }
                            
                        }
                        else if canSendMail {
                            
                            Text("Your estimated BAC is \(gameAttempt.BAC) %")
                            Text("The drink drive limit in the UK is 0.08 %")
                                .padding()
                            
                            GameSummaryView(gameAttempt: gameAttempt)
                                .frame(maxWidth: 300)
                                .padding()
                            
                            Text("Please check the above data is correct. ")
                                .frame(maxWidth: 300)
                            
                            Button("Click here to email the recorded data.") {
                                self.isMailComposeViewShowing.toggle()
                                dataHasBeenShared = true
                            }
                            .padding()
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                            
                            
                        }
                        else {
                            
                            Text("Your estimated BAC is \(gameAttempt.BAC) %")
                            Text("The drink drive limit in the UK is 0.08 %")
                                .padding()
                            
                            GameSummaryView(gameAttempt: gameAttempt)
                                .frame(maxWidth: 300)
                                .padding()
                            
                            Text("Email is not configured on this device.")
                                .onAppear{
                                    // necessary so return back can function
                                    dataHasBeenShared = true
                                }
                            
                        }
                    }
                    .sheet(isPresented: $isMailComposeViewShowing) {
                        MailComposeView(isShowing: self.$isMailComposeViewShowing, attachmentData: testTrackData, gameAttempt: gameAttempt)
                    }
                
                    ReturnToStartButton(dataHasBeenShared: $dataHasBeenShared,
                                        isNavigationActive: $isNavigationActive,
                                        showAlert: $showAlert)
                    
                    NavigationLink("", destination: StartView(), isActive: $isNavigationActive)
                                       .opacity(0)
                    
                }

    
                } else {
                
                    ProgressView("Writing Data...")
                    
                }
                        
        }
        .navigationBarBackButtonHidden(true)
        
        .onAppear {
            
            gameAttempt.BAC = calculateBAC(gender: gameAttempt.gender, hours: gameAttempt.timeHours, minutes: gameAttempt.timeMinutes, units: gameAttempt.unitsDrunk ?? 0, weight: gameAttempt.weight)
                    
                    //print(testTrackData)
                    if testTrackData == nil {
                        
                        do {
                            
                            let encoder = JSONEncoder()
                            testTrackData = try encoder.encode($gameAttempt.wrappedValue)
                            isFileWritingComplete = true
                            print("Encoding finished")
                            
                            
                        } catch {
                            print("Error encoding TestTrack: \(error)")
                        }
                    }
            }
    }
}



struct GameOverView_previews: PreviewProvider
{
    static var previews: some View {
        
        let gameAttempt = Binding.constant(TestTrack())
        return GameOverView(gameAttempt: gameAttempt)
        
    }
}

