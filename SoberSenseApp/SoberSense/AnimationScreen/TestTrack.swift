//
//  TestTrack.swift
//  SoberSense
//
//  Created by nick on 22/09/2023.
//

import Foundation
import SwiftUI


struct screenDim: Codable {
    let width: Float
    let height: Float
}


struct TestTrack: Identifiable, Codable {
    let id: UUID
    
    var unitsDrunk: Float?
    var weight: Int
    var gender: String
    
    // time since first drink
    var timeHours: Int
    var timeMinutes: Int
    
    var BAC: Float
    
    var touchData: [singleTouch]
    
    var randomPath: [coordinateDimScale]
    var controlPoints: [Double]
    var animationDuration: TimeInterval
    
    var screenSize: screenDim
    var duration: Int
    var jumpTotal: Int
    var minDistance: Float
    var extremeCoord: Float
    

    init() {
        
        self.id = UUID()
        // -1 to show any erroneous recording
        self.unitsDrunk = 0
        self.weight = 0
        // default as what picker starts on
        self.gender = "Male"
        
        self.BAC = 0
    
        self.timeHours = 0
        self.timeMinutes = 0
        
        self.touchData = []
        self.controlPoints = []
        self.animationDuration = 0
        self.randomPath = []
        self.screenSize = screenDim(width: Float(UIScreen.main.bounds.size.width), height: Float(UIScreen.main.bounds.size.height))
        self.duration = 0
        self.jumpTotal = 0
        self.minDistance = 0
        self.extremeCoord = 0
    }
}


