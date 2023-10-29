//
//  RecordingTouchView.swift
//  SoberSense
//
//  Created by nick on 05/10/2023.
//
//

// Using UKKit to create view which can store touch data

import SwiftUI
import UIKit

struct singleTouch: Codable {
    let xLocation: Float
    let yLocation: Float
    let time: Double
}

// Using UIViewRepresentable to integrate UIView into SwiftUI
struct RecordingTouchView: UIViewRepresentable {
    
    // This will receive touchData binding from parent view
    let touchData: Binding<[singleTouch]>
    
    func makeUIView(context: Context) -> UIView {
        
        let uiView = RecordingTouchUIView(touchData: touchData)
        
        return uiView
    }

    func updateUIView(_ uiView: UIView, context: Context) {
    }
}


class RecordingTouchUIView: UIView {
    
    // This allows the underlying touchData to be modified
    @Binding var touchData: [singleTouch]
        
    init(touchData: Binding<[singleTouch]>) {
        self._touchData = touchData
        super.init(frame: .zero)
    }
    
    required init?(coder: NSCoder) {
            fatalError("init(coder:) has not been implemented")
    }
    
    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        super.touchesMoved(touches, with: event)
        
        if let touch = touches.first {
            
            
            // touch.location(in: nil) gives coordinates across whole screen

            let touchPoint = singleTouch(
                xLocation: Float(touch.location(in: nil).x),
                yLocation: Float(touch.location(in: nil).y),
                // can use timestamp of touch (touch.timestamp), but difficult to compare to animation timings
                // therefore use CACurrentMediaTime
                time: Double(CACurrentMediaTime())
            )
            touchData.append(touchPoint)
        
        }
    }
}
