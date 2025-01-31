//
//  ORTTextToSpeechPerformer.h
//  TextToSpeechExample
//
//  Created by Deep Bhupatkar on 31/01/25.
//

#ifndef ORTTextToSpeechPerformer_h
#define ORTTextToSpeechPerformer_h

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface ORTTextToSpeechPerformer : NSObject

+ (nullable NSData *)performText:(NSString *)text
                      toSpeech:(NSString *)voiceFile
                         speed:(float)speed;

@end

NS_ASSUME_NONNULL_END

#endif
