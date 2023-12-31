//
//  AnimatingCircleView.swift
//  SoberSense
//
//  Created by nick on 30/10/2023.
//

import SwiftUI

struct AnimatingCircleView: View {
    var lineThickness: CGFloat
    var screenSize: CGSize
    var currentIndex: Int
    var coordinates: [coordinateDimScale]
    var circleAnimatedRadius: CGFloat

    var body: some View {
        ZStack {

            // Vertical Line
            Rectangle()
                .frame(width: lineThickness, height: screenSize.height)
                .foregroundColor(Color.red)
                .offset(x: CGFloat(coordinates[currentIndex].xDimScale) * screenSize.width, y: 0)
                .edgesIgnoringSafeArea(.all)

            // Horizontal Line
            Rectangle()
                .frame(width: screenSize.width, height: lineThickness)
                .foregroundColor(Color.red)
                .offset(x: 0, y: CGFloat(coordinates[currentIndex].yDimScale) * screenSize.height)
                .edgesIgnoringSafeArea(.all)

            
            Circle()
                .frame(width: circleAnimatedRadius, height: circleAnimatedRadius)
                .foregroundColor(Color.blue)
                .offset(
                    x: CGFloat(coordinates[currentIndex].xDimScale) * screenSize.width,
                    y: CGFloat(coordinates[currentIndex].yDimScale) * screenSize.height
                )
            
        }
    }
}
