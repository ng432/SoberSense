//
//  GeneratePath.swift
//  SoberSense
//
//  Created by nick on 22/09/2023.
//

import SwiftUI

struct coordinateDimScale: Codable {
    let xDimScale: Float
    let yDimScale: Float
    // To record the start time of animation AWAY from this coordinate, and to next
    var time: Double?
}

// Function to generate random coordinate value at minDistance away from current coordinate
// Coordinate generted in range -extremeCoord...extremeCoord
func RandomScale(SingleDim: Float, minDistance: Float, extremeCoord: Float) -> Float
{
    
    // First need to check if coordinate is closer to edge than minDistance
    // If it is, need to generate only from one range
    if (SingleDim - minDistance) < -extremeCoord {
        
        return Float.random(in: (SingleDim + minDistance)...extremeCoord)
        
    }
    else if (Float(SingleDim) + minDistance) > extremeCoord {
        
        return Float.random(in: -extremeCoord...(SingleDim - minDistance))
        
    }
    // Below is to account for selecting for 2 different ranges around current coordinate
    else {
        
        let totalPossibleWidth = 2 * (extremeCoord - minDistance)
        
        // gives probability measure to select lowRange proportional to its width
        let lowRangeProb = ((SingleDim - minDistance) + extremeCoord) / totalPossibleWidth
        
        let rangeSelector = Float.random(in: 0...1)
        
        // generate from lowRange
        if rangeSelector < lowRangeProb {
            
            return Float.random(in: -extremeCoord...(SingleDim - minDistance))
            
        }
        else {
            
            return Float.random(in: (SingleDim + minDistance)...extremeCoord)
        }
    }
}

    

// Function generates array of coordinates for path to follow
// Duration is length of time of animation
// jumpTotal is number of jumps generated
// minDistance (<0.5) provides minimum distance from current coordinate to next, in terms of total dimension of screen
func RandomPath(duration: Int, jumpTotal: Int, minDistance: Float, extremeCoord: Float) -> ([coordinateDimScale], Float)
{
    // Initialise path at centre of screen
    var path: [coordinateDimScale] = [coordinateDimScale(xDimScale: 0, yDimScale: 0, time: nil)]
    let jumpFreq: Float = Float(duration) / Float(jumpTotal)
    
    for i in 0...(jumpTotal-1) {
        
        path.append(coordinateDimScale(
            xDimScale: RandomScale(SingleDim: path[i].xDimScale, minDistance: minDistance, extremeCoord: extremeCoord),
            yDimScale: RandomScale(SingleDim: path[i].yDimScale, minDistance: minDistance, extremeCoord: extremeCoord),
            time: nil))
        
    }
    return (path, jumpFreq)
}


