/**
 * Standalone AudioFlux test to debug DLL integration
 */

#include <iostream>
#include <vector>
#include <cmath>

#ifdef USE_AUDIOFLUX
#include "../src/audio/AudioFluxBeatDetector.h"
#endif

// Simple test tone generator
std::vector<float> generateTestTone(int sampleRate, double duration, double freq) {
    size_t numSamples = static_cast<size_t>(sampleRate * duration);
    std::vector<float> samples(numSamples);

    for (size_t i = 0; i < numSamples; ++i) {
        double t = static_cast<double>(i) / sampleRate;
        // Generate a simple sine wave with some clicks at beat positions
        samples[i] = 0.3f * std::sin(2.0 * 3.14159265 * freq * t);

        // Add clicks every 0.5 seconds (120 BPM)
        double beatTime = std::fmod(t, 0.5);
        if (beatTime < 0.01) {
            samples[i] += 0.7f;
        }
    }

    return samples;
}

int main() {
    std::cout << "=== AudioFlux Integration Test ===" << std::endl;

#ifdef USE_AUDIOFLUX
    std::cout << "AudioFlux support: ENABLED" << std::endl;

    // Test 1: Check if AudioFlux is available
    std::cout << "\nTest 1: Checking AudioFlux availability..." << std::endl;
    try {
        bool available = AudioFluxBeatDetector::isAvailable();
        std::cout << "  AudioFlux available: " << (available ? "YES" : "NO") << std::endl;

        if (!available) {
            std::cerr << "  ERROR: AudioFlux not available!" << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "  EXCEPTION in isAvailable(): " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "  UNKNOWN EXCEPTION in isAvailable()!" << std::endl;
        return 1;
    }

    // Test 2: Create detector and analyze synthetic audio
    std::cout << "\nTest 2: Creating AudioFluxBeatDetector..." << std::endl;
    try {
        AudioFluxBeatDetector detector;
        std::cout << "  Detector created successfully" << std::endl;

        // Generate 5 seconds of test audio at 44100 Hz
        std::cout << "\nTest 3: Generating test audio (5 sec, 120 BPM clicks)..." << std::endl;
        auto testAudio = generateTestTone(44100, 5.0, 440.0);
        std::cout << "  Generated " << testAudio.size() << " samples" << std::endl;

        // Run detection
        std::cout << "\nTest 4: Running beat detection..." << std::endl;
        auto result = detector.detect(testAudio, 44100, nullptr);

        if (!result.error.empty()) {
            std::cerr << "  ERROR: " << result.error << std::endl;
            return 1;
        }

        std::cout << "  BPM: " << result.bpm << std::endl;
        std::cout << "  Beats detected: " << result.beats.size() << std::endl;
        std::cout << "  Confidence: " << result.confidence << std::endl;

        // Print first few beat times
        std::cout << "  First beat times: ";
        for (size_t i = 0; i < std::min(size_t(5), result.beats.size()); ++i) {
            std::cout << result.beats[i] << "s ";
        }
        std::cout << std::endl;

        // Verify BPM is close to 120
        if (result.bpm > 110 && result.bpm < 130) {
            std::cout << "\n  SUCCESS: BPM is in expected range (110-130)" << std::endl;
        } else {
            std::cout << "\n  WARNING: BPM " << result.bpm << " is outside expected range" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "  EXCEPTION: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "  UNKNOWN EXCEPTION!" << std::endl;
        return 1;
    }

    std::cout << "\n=== All tests passed! ===" << std::endl;
    return 0;

#else
    std::cout << "AudioFlux support: DISABLED (USE_AUDIOFLUX not defined)" << std::endl;
    return 1;
#endif
}
