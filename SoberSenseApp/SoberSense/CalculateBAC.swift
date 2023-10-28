//
//  CalculateBAC.swift
//  SoberSense
//
//  Created by nick on 26/10/2023.
//

// Calculating using the Widmark formula
func calculateBAC(gender: String, hours: Int, minutes: Int, units: Float, weight: Int) -> Float {
    
    let rValue: Float
    var bacTimeAdjusted: Float = 0.0
    if gender == "Male" {
        rValue = 0.68
    } else if gender == "Female" {
        rValue = 0.55
    } else {
        return -1.0 // Invalid gender
    }
    
    if weight > 0 {
        let gramsOfAlcohol = units * 8.0 // 1 unit (UK) = 8 grams of alcohol
        let gramsOfWeight = Float(weight) * 1000 // Kg to grams
        let totalTime = (Float(hours) + Float(minutes)) / 60
        
        let bacUnadjusted = (gramsOfAlcohol / (gramsOfWeight * rValue)) * 100
        bacTimeAdjusted = bacUnadjusted - (totalTime * 0.015)
    }
    
    return max(bacTimeAdjusted, 0.0)
}
