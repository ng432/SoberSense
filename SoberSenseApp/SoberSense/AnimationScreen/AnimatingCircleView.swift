//
//  AnimatingCircleView.swift
//  SoberSense
//
//  Created by nick on 30/10/2023.
//

import SwiftUI

struct AnimatingCircleView: View {
    var touchPoint: CGPoint?
    var lineThickness: CGFloat
    var screenSize: CGSize
    var currentIndex: Int
    var coordinates: [coordinateDimScale]
    var circleAnimatedRadius: CGFloat

    var body: some View {
        ZStack {
            if let touchPoint = touchPoint {
                Circle()
                    .frame(width: 60, height: 60)
                    .foregroundColor(Color.red)
                    .opacity(0.5)
                    .position(touchPoint)
            }

            // Vertical Line
            Rectangle()
                .frame(width: lineThickness, height: screenSize.height)
                .foregroundColor(Color.red)
                .offset(x: CGFloat(coordinates[currentIndex].xDimScale) * screenSize.width, y: 0)

            // Horizontal Line
            Rectangle()
                .frame(width: screenSize.width, height: lineThickness)
                .foregroundColor(Color.red)
                .offset(x: 0, y: CGFloat(coordinates[currentIndex].yDimScale) * screenSize.height)

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
