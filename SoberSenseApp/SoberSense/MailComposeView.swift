//
//  ShareView.swift
//  SoberSense
//
//  Created by nick on 16/10/2023.
//

import SwiftUI
import MessageUI

struct MailComposeView: UIViewControllerRepresentable {
    
    @Binding var isShowing: Bool
    
    let attachmentData: Data? // Data to be attached to the email
    let gameAttempt: TestTrack


    func makeUIViewController(context: Context) -> MFMailComposeViewController {
        let mailComposeViewController = MFMailComposeViewController()
        mailComposeViewController.setToRecipients(["ng432@cam.ac.uk"])
        
        let subject = "Anonymous user ID: " + gameAttempt.id.uuidString
        mailComposeViewController.setSubject(subject)
        
        mailComposeViewController.setMessageBody("This email contains touch data required to train a model.", isHTML: false)
        mailComposeViewController.mailComposeDelegate = context.coordinator

        if let attachmentData = attachmentData {
            
            let fileName = gameAttempt.id.uuidString + ".json"
            mailComposeViewController.addAttachmentData(attachmentData, mimeType: "application/octet-stream", fileName: fileName)
        }

        return mailComposeViewController
    }

    func updateUIViewController(_ uiViewController: MFMailComposeViewController, context: Context) {
        // You can implement any necessary updates here
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject, MFMailComposeViewControllerDelegate {
        var parent: MailComposeView

        init(_ parent: MailComposeView) {
            self.parent = parent
        }

        func mailComposeController(_ controller: MFMailComposeViewController, didFinishWith result: MFMailComposeResult, error: Error?) {
            parent.isShowing = false
        }
    }
}

