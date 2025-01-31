//
//  ORTTextToSpeechPerformer.m
//  TextToSpeechExample
//
//  Created by Deep Bhupatkar on 31/01/25.
//

#import <Foundation/Foundation.h>
#import "ORTTextToSpeechPerformer.h"
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#include <ctype.h>


#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>


#include <onnxruntime_cxx_api.h>
#include <onnxruntime_extensions.h>

// Define WAV format constants
#define SAMPLE_RATE 22050
#define NUM_CHANNELS 1
#define BITS_PER_SAMPLE 16

// Add these constants based on config.json
const int HIDDEN_DIM = 512;
const int N_MELS = 80;
const int STYLE_DIM = 256;
const int MAX_DUR = 50;

@implementation ORTTextToSpeechPerformer

+ (nullable NSData *)performText:(NSString *)text
                      toSpeech:(NSString *)voiceFile
                         speed:(float)speed {
    NSData *outputAudio = nil;
    
    try {
        // Initialize environment
        const auto ort_log_level = ORT_LOGGING_LEVEL_WARNING;
        auto ort_env = Ort::Env(ort_log_level, "ORTTextToSpeech");
        auto session_options = Ort::SessionOptions();
        
        // Configure threading
        session_options.SetIntraOpNumThreads(2);
        session_options.SetInterOpNumThreads(1);
        session_options.SetExecutionMode(ORT_SEQUENTIAL);

        // Memory settings
        session_options.AddConfigEntry("session.memory.enable_memory_arena_shrinkage", "0");
        session_options.AddConfigEntry("session.memory.arena_extend_strategy", "kNextPowerOfTwo");
        session_options.AddConfigEntry("session.memory.initial_chunk_size_bytes", "134217728");

        // Add model configuration parameters
        std::string hidden_dim_str = std::to_string(HIDDEN_DIM);
        std::string n_mels_str = std::to_string(N_MELS);
        std::string style_dim_str = std::to_string(STYLE_DIM);
        std::string max_dur_str = std::to_string(MAX_DUR);
        
        session_options.AddConfigEntry("model.hidden_dim", hidden_dim_str.c_str());
        session_options.AddConfigEntry("model.n_mels", n_mels_str.c_str());
        session_options.AddConfigEntry("model.style_dim", style_dim_str.c_str());
        session_options.AddConfigEntry("model.max_dur", max_dur_str.c_str());

        Ort::AllocatorWithDefaultOptions ortAllocator;
        
        if (RegisterCustomOps(session_options, OrtGetApiBase()) != nullptr) {
            throw std::runtime_error("Failed to register custom operators");
        }

        // Load model
        NSString *model_path = [NSBundle.mainBundle pathForResource:@"model" ofType:@"onnx"];
        if (!model_path) {
            throw std::runtime_error("Model file not found");
        }
        
        auto sess = Ort::Session(ort_env, [model_path UTF8String], session_options);

        // Clean and process input text
        NSString *cleanedText = [[[text stringByReplacingOccurrencesOfString:@"\n" withString:@" "]
                               stringByReplacingOccurrencesOfString:@"\t" withString:@" "]
                               stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
        
        // Convert numbers to words in the entire text
        std::string processedText([cleanedText UTF8String]);
        processedText = convertNumbersToWords(processedText);
        cleanedText = [NSString stringWithUTF8String:processedText.c_str()];
        
        // Tokenize text
        std::vector<int64_t> tokens;
        tokens.reserve(512);
        tokens.push_back(0);  // Initial padding token
        
        std::string inputText([cleanedText UTF8String]);
        size_t pos = 0;
        
        auto tokenizer = [self loadTokenizer];
        
        while (pos < inputText.length() && tokens.size() < 510) {
            bool foundToken = false;
            
            // Inside the tokenization loop:
            for (size_t len = std::min(4UL, inputText.length() - pos); len > 0; len--) {
                std::string substr = inputText.substr(pos, len);
                
                auto it = tokenizer.find(substr);
                if (it != tokenizer.end()) {
                    tokens.push_back(it->second);
                    pos += len;
                    foundToken = true;
                    break;
                }
            }
            
            if (!foundToken) {
                if (std::isspace(inputText[pos])) {
                    tokens.push_back(16);  // Space token
                    pos++;
                } else {
                    pos++;  // Skip unrecognized character
                }
            }
        }
        
        // Pad tokens
        while (tokens.size() < 512) {
            tokens.push_back(0);
        }

        // Create input tensors for current sentence
        std::vector<int64_t> input_ids_dims{1, 512};
        auto input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            ortAllocator, input_ids_dims.data(), input_ids_dims.size());
        memcpy(input_ids_tensor.GetTensorMutableData<int64_t>(), tokens.data(),
               tokens.size() * sizeof(int64_t));

        // Load voice data
        NSString *voicePath = [NSBundle.mainBundle pathForResource:@"voices" ofType:@"bin"];
        if (!voicePath) {
            throw std::runtime_error("Voice file not found");
        }

        // Map voice names to style indices
        std::map<std::string, size_t> voiceMap = {
            {"en_us", 0},      // American English
            {"en_uk", 1},      // British English
            {"en_neutral", 2}  // Neutral English
        };
        
        // Get style index from voice name
        std::string voiceStr([[voiceFile lowercaseString] UTF8String]);
        size_t style_index = 0;  // Default to first style
        auto voiceIt = voiceMap.find(voiceStr);
        if (voiceIt != voiceMap.end()) {
            style_index = voiceIt->second;
        }
        NSLog(@"Selected voice style index: %zu", style_index);

        NSData *voiceData = [NSData dataWithContentsOfFile:voicePath];
        if (!voiceData) {
            throw std::runtime_error("Failed to load voice data");
        }

        // Print voice data size for debugging
        NSLog(@"Voice data size: %lu bytes", (unsigned long)voiceData.length);
        NSLog(@"Expected style vector size: %d floats", STYLE_DIM);
        NSLog(@"Number of style vectors: %lu", (unsigned long)voiceData.length / (STYLE_DIM * sizeof(float)));
        NSLog(@"Using voice: %@", voiceFile);  // Log which voice is being used

        const float* all_style_data = (const float*)voiceData.bytes;
        size_t num_styles = voiceData.length / (STYLE_DIM * sizeof(float));

        // Select style based on content but ensure we don't exceed array bounds
        if (num_styles == 0) {
            throw std::runtime_error("No style vectors found in voice file");
        }

        // Create style tensor with correct dimensions
        std::vector<int64_t> style_dims{1, STYLE_DIM};
        auto style_tensor = Ort::Value::CreateTensor<float>(
            ortAllocator, style_dims.data(), style_dims.size());
        std::vector<float> style_data(STYLE_DIM);
        
        // Ensure we're not reading past the end of the voice data
        if ((style_index + 1) * STYLE_DIM * sizeof(float) > voiceData.length) {
            throw std::runtime_error("Voice data file is too small for style vector");
        }
        
        memcpy(style_data.data(), all_style_data + (style_index * STYLE_DIM), STYLE_DIM * sizeof(float));
        memcpy(style_tensor.GetTensorMutableData<float>(), style_data.data(),
               style_data.size() * sizeof(float));

        // Create speed tensor
        float validated_speed = std::max(0.8f, std::min(speed, 1.5f));
        std::vector<int64_t> speed_dims{1};
        auto speed_tensor = Ort::Value::CreateTensor<float>(
            ortAllocator, speed_dims.data(), speed_dims.size());
        memcpy(speed_tensor.GetTensorMutableData<float>(), &validated_speed, sizeof(float));

        // Run inference
        std::array<const char*, 3> input_names = {"input_ids", "style", "speed"};
        std::array<Ort::Value, 3> input_tensors{
            std::move(input_ids_tensor),
            std::move(style_tensor),
            std::move(speed_tensor)
        };

        auto output_name = sess.GetOutputNameAllocated(0, ortAllocator);
        std::array<const char*, 1> output_names = {output_name.get()};

        auto outputs = sess.Run(Ort::RunOptions{},
                              input_names.data(),
                              input_tensors.data(),
                              input_tensors.size(),
                              output_names.data(),
                              output_names.size());

        if (outputs.empty()) {
            throw std::runtime_error("No output generated");
        }

        // Process audio output
        const auto &output_tensor = outputs.front();
        const auto info = output_tensor.GetTensorTypeAndShapeInfo();
        const float* raw_audio = output_tensor.GetTensorData<float>();
        const size_t audio_length = info.GetElementCount();

        // Audio normalization and amplification
        std::vector<float> normalized_audio(audio_length);
        float max_amplitude = 0.0f;
        
        // Find maximum amplitude
        for (size_t i = 0; i < audio_length; i++) {
            max_amplitude = std::max(max_amplitude, std::abs(raw_audio[i]));
        }
        
        // Normalize and amplify
        const float target_amplitude = 0.95f;  // Increased target amplitude
        const float amplification = max_amplitude > 0.0f ? target_amplitude / max_amplitude : 1.0f;
        
        for (size_t i = 0; i < audio_length; i++) {
            normalized_audio[i] = raw_audio[i] * amplification;
        }

        // Convert float audio to int16
        std::vector<int16_t> pcm_data(audio_length);
        for (size_t i = 0; i < audio_length; i++) {
            float sample = normalized_audio[i];
            sample = std::max(-1.0f, std::min(1.0f, sample));
            pcm_data[i] = static_cast<int16_t>(sample * 32767.0f);
        }

        // WAV header constants
        const uint32_t wav_header_size = 44;
        const uint32_t data_size = static_cast<uint32_t>(audio_length * sizeof(int16_t));
        const uint32_t file_size = wav_header_size + data_size - 8;
        
        // Create WAV header
        std::vector<uint8_t> wav_header(wav_header_size);
        
        // RIFF chunk
        wav_header[0] = 'R'; wav_header[1] = 'I'; wav_header[2] = 'F'; wav_header[3] = 'F';
        wav_header[4] = (file_size) & 0xFF;
        wav_header[5] = (file_size >> 8) & 0xFF;
        wav_header[6] = (file_size >> 16) & 0xFF;
        wav_header[7] = (file_size >> 24) & 0xFF;
        
        // WAVE chunk
        wav_header[8] = 'W'; wav_header[9] = 'A'; wav_header[10] = 'V'; wav_header[11] = 'E';
        
        // fmt subchunk
        wav_header[12] = 'f'; wav_header[13] = 'm'; wav_header[14] = 't'; wav_header[15] = ' ';
        wav_header[16] = 16; wav_header[17] = 0; wav_header[18] = 0; wav_header[19] = 0;  // Subchunk1Size
        wav_header[20] = 1; wav_header[21] = 0;  // AudioFormat (PCM)
        wav_header[22] = NUM_CHANNELS; wav_header[23] = 0;  // NumChannels
        // SampleRate
        wav_header[24] = SAMPLE_RATE & 0xFF;
        wav_header[25] = (SAMPLE_RATE >> 8) & 0xFF;
        wav_header[26] = (SAMPLE_RATE >> 16) & 0xFF;
        wav_header[27] = (SAMPLE_RATE >> 24) & 0xFF;
        // ByteRate
        uint32_t byte_rate = SAMPLE_RATE * NUM_CHANNELS * (BITS_PER_SAMPLE / 8);
        wav_header[28] = byte_rate & 0xFF;
        wav_header[29] = (byte_rate >> 8) & 0xFF;
        wav_header[30] = (byte_rate >> 16) & 0xFF;
        wav_header[31] = (byte_rate >> 24) & 0xFF;
        // BlockAlign
        wav_header[32] = NUM_CHANNELS * (BITS_PER_SAMPLE / 8);
        wav_header[33] = 0;
        // BitsPerSample
        wav_header[34] = BITS_PER_SAMPLE;
        wav_header[35] = 0;
        
        // data subchunk
        wav_header[36] = 'd'; wav_header[37] = 'a'; wav_header[38] = 't'; wav_header[39] = 'a';
        wav_header[40] = data_size & 0xFF;
        wav_header[41] = (data_size >> 8) & 0xFF;
        wav_header[42] = (data_size >> 16) & 0xFF;
        wav_header[43] = (data_size >> 24) & 0xFF;

        // Combine header and audio data
        NSMutableData *wavData = [NSMutableData dataWithBytes:wav_header.data() length:wav_header_size];
        [wavData appendBytes:pcm_data.data() length:pcm_data.size() * sizeof(int16_t)];
        NSData *currentAudioData = wavData;
        
        outputAudio = currentAudioData;
        
    } catch (const std::exception &e) {
        NSLog(@"TTS Error: %s", e.what());
        return nil;
    }
    
    return outputAudio;
}

+ (std::map<std::string, int64_t>) loadTokenizer {
    std::map<std::string, int64_t> tokenizer;
    
    NSString *tokenizerPath = [NSBundle.mainBundle pathForResource:@"tokens" ofType:@"txt"];
    if (!tokenizerPath) {
        throw std::runtime_error("Tokenizer file not found");
    }
    
    std::ifstream file([tokenizerPath UTF8String]);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open tokenizer file");
    }
    
    NSLog(@"Loading tokenizer from: %@", tokenizerPath);
    
    std::string line;
    int count = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        size_t delimiter = line.find(' ');
        if (delimiter != std::string::npos) {
            std::string token = line.substr(0, delimiter);
            // Remove quotes if present
            if (token.size() >= 2 && token.front() == '"' && token.back() == '"') {
                token = token.substr(1, token.size() - 2);
            }
            
            try {
                int64_t id = std::stoll(line.substr(delimiter + 1));
                tokenizer[token] = id;
                count++;
                if (count < 5) {
                    NSLog(@"Loaded token: '%s' -> %lld", token.c_str(), id);
                }
            } catch (const std::exception& e) {
                NSLog(@"Failed to parse token line: %s", line.c_str());
            }
        }
    }
    
    
    NSLog(@"Loaded %d tokens total", count);
    return tokenizer;
}

// Helper function to convert numbers to words
std::string convertNumbersToWords(const std::string& text) {
    std::string result;
    std::string processed_text;
    std::string current_number;
    
    // Define number word arrays
    const char* ones[] = {"", "one", "two", "three", "four",
                        "five", "six", "seven", "eight", "nine"};
    const char* teens[] = {"ten", "eleven", "twelve", "thirteen", "fourteen",
                         "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"};
    const char* tens[] = {"", "", "twenty", "thirty", "forty",
                        "fifty", "sixty", "seventy", "eighty", "ninety"};
    
    for (size_t i = 0; i < text.length(); i++) {
        if (isdigit(text[i])) {
            current_number += text[i];
            // Add space before number if it's not at the start and previous char isn't a space
            if (!result.empty() && result.back() != ' ') {
                result += ' ';
            }
        } else {
            if (!current_number.empty()) {
                // Convert number string to words
                int number = std::stoi(current_number);
                
                // Handle years (1946-2025)
                if (number >= 1900 && number <= 2025) {
                    int century = number / 100;
                    int year = number % 100;
                    
                    // Handle century part (19 or 20)
                    if (century == 19) {
                        result += "nineteen";
                    } else if (century == 20) {
                        result += "twenty";
                    }
                    
                    result += " ";
                    
                    // Handle year part
                    if (year >= 10 && year <= 19) {
                        result += teens[year - 10];
                    } else {
                        result += tens[year / 10];
                        if (year % 10 != 0) {
                            result += " ";
                            result += ones[year % 10];
                        }
                    }
                } else {
                    // Handle regular numbers
                    if (number == 0) {
                        result += "zero";
                    } else if (number < 10) {
                        result += ones[number];
                    } else if (number < 20) {
                        result += teens[number - 10];
                    } else if (number < 100) {
                        result += tens[number / 10];
                        if (number % 10 != 0) {
                            result += " ";
                            result += ones[number % 10];
                        }
                    } else {
                        // For numbers >= 100, convert recursively
                        std::string num_str = std::to_string(number);
                        result += convertNumbersToWords(num_str);
                    }
                }
                
                // Add space after the converted number
                if (i < text.length() && !isspace(text[i])) {
                    result += ' ';
                }
                
                current_number.clear();
            }
            // Only add non-space characters or single spaces
            if (!isspace(text[i]) ||
                (result.length() > 0 && !isspace(result.back()))) {
                result += text[i];
            }
        }
    }
    
    // Handle any remaining number at the end of text
    if (!current_number.empty()) {
        if (!result.empty() && result.back() != ' ') {
            result += ' ';
        }
        result += convertNumbersToWords(current_number);
    }
    
    // Clean up multiple spaces
    std::string final_result;
    bool last_was_space = true;  // Start true to trim leading spaces
    for (char c : result) {
        if (isspace(c)) {
            if (!last_was_space) {
                final_result += ' ';
                last_was_space = true;
            }
        } else {
            final_result += c;
            last_was_space = false;
        }
    }
    
    // Trim trailing space if exists
    if (!final_result.empty() && isspace(final_result.back())) {
        final_result.pop_back();
    }
    
    return final_result;
}
@end
