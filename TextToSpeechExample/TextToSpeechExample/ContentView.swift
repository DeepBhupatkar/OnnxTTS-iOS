//
//  ContentView.swift
//  TextToSpeechExample
//
//  Created by Deep Bhupatkar on 31/01/25.
//

import SwiftUI
import AVFoundation

// Audio manager class to handle playback
class AudioManager: NSObject, AVAudioPlayerDelegate {
    static let shared = AudioManager()
    var audioPlayer: AVAudioPlayer?
    
    func playAudio(data: Data) {
        do {
            // Configure audio session
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.playback, mode: .spokenAudio)
            try audioSession.setActive(true)
            
            // Create and play audio
            audioPlayer = try AVAudioPlayer(data: data)
            audioPlayer?.delegate = self
            audioPlayer?.volume = 1.0  // Normal volume
            audioPlayer?.enableRate = true
            audioPlayer?.rate = 1.0  // Normal speed
            audioPlayer?.numberOfLoops = 0
            audioPlayer?.prepareToPlay()
            
            audioPlayer?.play()
        } catch {
            print("Error playing audio: \(error.localizedDescription)")
        }
    }
    
    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        audioPlayer = nil
    }
}

struct ContentView: View {
    @State private var inputText: String = """
        Donald John Trump (born June 14, 1946) is an American politician, media personality, and businessman who has served as the 47th president of the United States since January 20, 2025. A member of the Republican Party, he previously served as the 45th president from 2017 to 2021.Born in New York City, Trump graduated with a bachelor's degree in economics from the University of Pennsylvania in 1968.
        """
    @State private var isProcessing = false
    @State private var progressText = ""
    
    func runTextToSpeech(text: String) {
        do {
            isProcessing = true
            progressText = "Preparing speech..."
            
            // Process entire text at once
            if let audioData = try ORTTextToSpeechPerformer.performText(
                text.trimmingCharacters(in: .whitespaces),
                toSpeech: "en_us",
                speed: 1.0) {
                
                progressText = "Speaking text..."
                AudioManager.shared.playAudio(data: audioData)
                
                // Wait for audio to finish
                while AudioManager.shared.audioPlayer?.isPlaying == true {
                    Thread.sleep(forTimeInterval: 0.05)
                }
            }
            
            progressText = "Finished speaking"
            isProcessing = false
        } catch {
            print("Error: \(error.localizedDescription)")
            progressText = "Error: \(error.localizedDescription)"
            isProcessing = false
        }
    }
    
    var body: some View {
        VStack {
            Text("Text to Speech").font(.title).bold()
                .frame(width: 400, height: 80)
                .border(Color.purple, width: 4)
                .background(Color.purple)
            
            TextField("Enter text to speak", text: $inputText)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .padding()
            
            Button("Speak") {
                runTextToSpeech(text: inputText)
            }
            .padding()
            .background(Color.blue)
            .foregroundColor(.white)
            .cornerRadius(10)
            .disabled(isProcessing)
            
            if !progressText.isEmpty {
                Text(progressText)
                    .foregroundColor(isProcessing ? .blue : .green)
                    .padding()
            }
        }
        .padding()
    }
}

#Preview {
    ContentView()
}
