//
//  AnimationView.swift
//  SoberSense
//
//  Created by nick on 22/09/2023.
//

import SwiftUI
import Combine

// For recording touch Data
struct singleTouch: Codable, Equatable{
    let xLocation: Float
    let yLocation: Float
    let time: Double
}

struct AnimationOverView: View {
    
    @Binding var gameAttempt: TestTrack
    
    // To store animation data and timing
    @State private var pathRecord: [coordinateDimScale] = []
    
    // To store touchData during animation
    @State private var touchData: [singleTouch] = []
    
    // To track progress through coordinate path
    @State private var currentIndex: Int = 0
    @State private var timer: Publishers.Autoconnect<Timer.TimerPublisher> = Timer.publish(every: 2, on: .main, in: .common).autoconnect()
    
    @State private var touchPoint: CGPoint?
    @State private var gameIsOver = false
    @State private var animationHasStarted = false
    @State private var recordTouches = false

    let screenSize = UIScreen.main.bounds.size
    let circleAnimatedRadius: CGFloat = 50
    let lineThickness: CGFloat = 2
    
    // Variables to define coordinate path creation
    let minDistance: Float = 0.1
    let extremeCoord: Float = 0.4
    
    // Attempting to define by path
    let jumpIntervals: [Double]
    let minInterval: Double = 0.6
    let maxInterval: Double = 1.1
    let duration: Int = 20
    
    // Defining variables for Bezier curve of animation
    let controlPoints: [Double] = [0.42, 0.0, 0.58, 1.0]
    let animationDuration: TimeInterval = 0.4
    
    
    // List of coordinates defining random path
    let coordinates: [coordinateDimScale]
    @State private var currentJumpIndex = 0
    
    init(gameAttempt: Binding<TestTrack>) {
        
        self._gameAttempt = gameAttempt
        
        self.jumpIntervals = RandomTimeIntervals(duration: duration, minInterval: minInterval, maxInterval: maxInterval)
        
        let parameters: ([coordinateDimScale], Float) = RandomPath(
            duration: duration, jumpTotal: jumpIntervals.count - 1, minDistance: minDistance, extremeCoord: extremeCoord)
        self.coordinates = parameters.0
        
        }
    
    var body: some View {
        
        NavigationStack{
            
            ZStack{
                // Initial start screen
                if !animationHasStarted {
                    
                    InitialAnimationView(
                        lineThickness: lineThickness,
                        screenSize: screenSize,
                        circleAnimatedRadius: circleAnimatedRadius,
                        animationHasStarted: $animationHasStarted)
                        
                }
                // Running through animation path
                else if currentIndex < coordinates.count && animationHasStarted {
                    
                    // Display circle on touch, and record touch data
                    if let touchPoint = touchPoint {
                            Circle()
                                .frame(width: 60, height: 60)
                                .foregroundColor(Color.red)
                                .opacity(0.5)
                                .position(touchPoint)
                                .onChange(of: touchPoint)
                                    {newValue in
                                        let touchPoint = singleTouch(
                                            xLocation: Float(newValue.x),
                                            yLocation: Float(newValue.y),
                                            time: Double(CACurrentMediaTime())
                                            )
                                    touchData.append(touchPoint)
                                    }
                    }

                    AnimatingCircleView(
                                lineThickness: lineThickness,
                                screenSize: screenSize,
                                currentIndex: currentIndex,
                                coordinates: coordinates,
                                circleAnimatedRadius: circleAnimatedRadius
                            )
                    .onAppear{
                        performNextJump()
                    }
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
                            gameAttempt.minDistance = minDistance
                            gameAttempt.extremeCoord = extremeCoord
                            gameAttempt.jumpIntervals = jumpIntervals
                            
                        }
                }
                
            }
        
            // necessary to recognise touch for recording
            .simultaneousGesture(DragGesture(coordinateSpace: .local)
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
    }
    
    // performs and times animation depending on defined intervals
    private func performNextJump() {
            if currentJumpIndex < jumpIntervals.count {
                DispatchQueue.main.asyncAfter(deadline: .now() + jumpIntervals[currentJumpIndex]) {
                    
                    withAnimation (.timingCurve(controlPoints[0], controlPoints[1], controlPoints[2], controlPoints[3], duration: animationDuration)) {
                        
                        // records starting point for an animation, and time of start
                        if currentIndex < coordinates.count {
                            var currentCoordinate = coordinates[currentIndex]
                            currentCoordinate.time = Double(CACurrentMediaTime())
                            pathRecord.append(currentCoordinate)
                            
                        }
                        currentIndex = (currentIndex + 1)
                    }
                    currentJumpIndex += 1
                    performNextJump() // Schedule the next jump
                }
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
