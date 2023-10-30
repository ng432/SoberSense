//
//  AnimationView.swift
//  SoberSense
//
//  Created by nick on 22/09/2023.
//

import SwiftUI
import Combine


struct AnimationOverView: View {
    
    @Binding var gameAttempt: TestTrack
    
    // To store animation data and time 
    @State private var pathRecord: [coordinateDimScale] = []
    
    // To store touchData during animation
    @State private var touchData: [singleTouch] = []
    
    // To track progress through coordinate path
    @State private var currentIndex: Int = 0
    @State private var timer: Publishers.Autoconnect<Timer.TimerPublisher> = Timer.publish(every: 2, on: .main, in: .common).autoconnect()
    
    @State private var touchPoint: CGPoint?
    
    @State private var gameIsOver = false
    
    @State private var animationHasStarted = false

    let screenSize = UIScreen.main.bounds.size
    let circleAnimatedRadius: CGFloat = 50
    let lineThickness: CGFloat = 2
    
    // Variables to define coordinate path creation
    let duration: Int = 20
    let jumpTotal: Int = 25
    let minDistance: Float = 0.1
    let extremeCoord: Float = 0.4
    let jumpFreq: Double
    
    // Defining variables for Bezier curve of animation
    let controlPoints: [Double] = [0.42, 0.0, 0.58, 1.0]
    let animationDuration: TimeInterval = 0.40
    
    // List of coordinates defining random path
    let coordinates: [coordinateDimScale]
    
    init(gameAttempt: Binding<TestTrack>) {
        
        self._gameAttempt = gameAttempt
        
        let parameters: ([coordinateDimScale], Float) = RandomPath(
            duration: duration, jumpTotal: jumpTotal, minDistance: minDistance, extremeCoord: extremeCoord)
        self.coordinates = parameters.0
        self.jumpFreq = Double(parameters.1)
        
        }
    

    var body: some View {
        
        NavigationStack{
                
                ZStack{
                    
                    // Running through animation path
                    if currentIndex < coordinates.count && animationHasStarted {
                        
                        AnimatingCircleView(
                                    touchPoint: touchPoint,
                                    lineThickness: lineThickness,
                                    screenSize: screenSize,
                                    currentIndex: currentIndex,
                                    coordinates: coordinates,
                                    circleAnimatedRadius: circleAnimatedRadius
                                )
                       
                        RecordingTouchView(touchData: $touchData)
                        
                        
                    }
                    else if !animationHasStarted {
                        
                        InitialAnimationView(lineThickness: lineThickness, screenSize: screenSize, circleAnimatedRadius: circleAnimatedRadius, animationHasStarted: $animationHasStarted)
                            
    
                    }
                    // Runs once the random path has finished
                    else {

                        // Navigates to GameOverView, saving touchData and randomPath
                        NavigationLink(" ", destination: GameOverView(gameAttempt: $gameAttempt), isActive: $gameIsOver)
                            .onAppear {
                                gameIsOver = true
                                
                                // Saving relevant data for export
                                gameAttempt.touchData = touchData
                                gameAttempt.randomPath = pathRecord
                                gameAttempt.controlPoints = controlPoints
                                gameAttempt.animationDuration = animationDuration
                                gameAttempt.duration = duration
                                gameAttempt.jumpTotal = jumpTotal
                                gameAttempt.minDistance = minDistance
                                gameAttempt.extremeCoord = extremeCoord
                                
                            }
                    }
                    
                }
                // Closure to run code on signal from timer
                .onReceive(timer) { _ in
        
                    if animationHasStarted {
                    
                    withAnimation (.timingCurve(controlPoints[0], controlPoints[1], controlPoints[2], controlPoints[3], duration: animationDuration))
                        {
                            // Recording start time of movement to next coordinate
                            
                            if currentIndex < coordinates.count{
                                
                                var currentCoordinate = coordinates[currentIndex]
                                currentCoordinate.time = Double(CACurrentMediaTime())
                                pathRecord.append(currentCoordinate)
                
                            }
                            
                            // Move to the next coordinate in the list
                            currentIndex = (currentIndex + 1)
                        }
                    }
                }
                .gesture(DragGesture(coordinateSpace: .local)
                    .onChanged { value in
                        if animationHasStarted {
                            touchPoint = value.location
                        }
                    }
                    .onEnded { _ in
                        touchPoint = nil
                    }
                )
        
        }
        .navigationBarBackButtonHidden(true)
        .padding(.horizontal, 5)
        .padding(.vertical, 2)
        
        .onAppear {
                    
            // Xcode didn't like initalising timer and gameAttempt together within 'init' (throwing bug)
            // So instead have initalised timer to defualt freq, then on appear reset timer to correct freq
        
            self.timer.upstream.connect().cancel()
            self.timer = Timer.publish(every: self.jumpFreq, on: .main, in: .common).autoconnect()
            
        }
            

    }
        
        
}

struct AnimationView_previews: PreviewProvider
{
    static var previews: some View {
        let gameAttempt = Binding.constant(TestTrack())
        return AnimationOverView(gameAttempt: gameAttempt)
    }
}
