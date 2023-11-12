//
//  StartAnimationView.swift
//  SoberSense
//
//  Created by nick on 26/10/2023.
//

import SwiftUI

struct InitialAnimationView: View {
    let lineThickness: CGFloat
    let screenSize: CGSize
    let circleAnimatedRadius: CGFloat
    @Binding var animationHasStarted: Bool

    var body: some View {
        
        NavigationStack {
            
            ZStack {
                
                Rectangle()
                    .frame(width: lineThickness, height: screenSize.height)
                    .foregroundColor(Color.red)
                    .edgesIgnoringSafeArea(.all)
                
                Rectangle()
                    .frame(width: screenSize.width, height: lineThickness)
                    .foregroundColor(Color.red)
                    .edgesIgnoringSafeArea(.all)
                
                Circle()
                    .frame(width: circleAnimatedRadius, height: circleAnimatedRadius)
                    .foregroundColor(Color.blue)
                    .onLongPressGesture(minimumDuration: 0.1){
                        animationHasStarted = true
                    }
                
                NavigationLink("Go back", destination: StartView())
                    .padding(10)
                    .background(Color.red)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                    .offset(x: 0, y: -250)
                
                
                Text("Hold your phone with your non-dominant hand.")
                    .multilineTextAlignment(.center)
                    .padding(5)
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                    .frame(maxWidth: 250)
                    .offset(x: 0, y: -180)
                
                
                Text("Try to keep the index finger of your dominant hand on the circle as it moves.")
                    .multilineTextAlignment(.center)
                    .padding(5)
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                    .frame(maxWidth: 230)
                    .offset(x: 0, y: -90)
                
                Text("Once the circle starts moving, don't take your finger off the screen.")
                    .multilineTextAlignment(.center)
                    .padding(6)
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                    .frame(maxWidth: 220)
                    .offset(x: 0, y: 90)
                
                Text("Hold your finger on the circle to start.")
                    .multilineTextAlignment(.center)
                    .padding(6)
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                    .frame(maxWidth: 150)
                    .offset(x: 0, y: 180)
                    .onTapGesture {
                        animationHasStarted = true
                    }
                
            }
        }
    }
    
}

struct InitialAnimationView_previews: PreviewProvider
{
    static var previews: some View {
        @State var animationHasStarted = false
        let screenSize = UIScreen.main.bounds.size
        let circleAnimatedRadius: CGFloat = 50
        let lineThickness: CGFloat = 2
        return InitialAnimationView(lineThickness: lineThickness, screenSize: screenSize, circleAnimatedRadius: circleAnimatedRadius, animationHasStarted: $animationHasStarted)
    }
}
