//
//  AlcoholUnitTable.swift
//  SoberSense
//
//  Created by nick on 26/10/2023.
//

import SwiftUI

struct AlcoholUnitTable: View {
    var body: some View {
        VStack {
            Group {
                HStack {
                    Text("Shot")
                    Spacer()
                    Text(" 1 unit")
                }

                HStack {
                    Text("Pint of lager, beer, or cider")
                    Spacer()
                    Text(" 2.8 units")
                }
                
                HStack {
                    Text("Can of lager, beer, or cider (440ml)")
                    Spacer()
                    Text(" 2.4 units")
                }

                HStack {
                    Text("Bottle of lager, beer, or cider (330ml)")
                    Spacer()
                    Text(" 1.7 units")
                }

                HStack {
                    Text("Small/large glass of wine (125/250ml)")
                    Spacer()
                    Text(" 1.5 / 3 units")
                }
            }
            .font(.system(size: 14))
            .foregroundColor(.gray)
            .multilineTextAlignment(.center)
        }
    }
}
