/**
 * Backend Smoke Test
 * Quick sanity check that the backend library can be instantiated
 */

#include "../audio/AudioAnalyzer.h"
#include "../video/VideoWriter.h"
#include <iostream>

int main() {
    std::cout << "=== BeatSync Backend Smoke Test ===\n";

    // Test AudioAnalyzer instantiation
    std::cout << "[1/3] Creating AudioAnalyzer... ";
    try {
        BeatSync::AudioAnalyzer analyzer;
        std::cout << "OK\n";
    } catch (const std::exception& e) {
        std::cout << "FAILED: " << e.what() << "\n";
        return 1;
    }

    // Test VideoWriter instantiation
    std::cout << "[2/3] Creating VideoWriter... ";
    try {
        BeatSync::VideoWriter writer;
        std::cout << "OK\n";
    } catch (const std::exception& e) {
        std::cout << "FAILED: " << e.what() << "\n";
        return 1;
    }

    // Test FFmpeg path resolution
    std::cout << "[3/3] Resolving FFmpeg path... ";
    try {
        BeatSync::VideoWriter writer;
        std::string ffmpegPath = writer.resolveFfmpegPath();
        if (ffmpegPath.empty()) {
            std::cout << "WARNING: FFmpeg not found (may be OK if not installed)\n";
        } else {
            std::cout << "OK: " << ffmpegPath << "\n";
        }
    } catch (const std::exception& e) {
        std::cout << "FAILED: " << e.what() << "\n";
        return 1;
    }

    std::cout << "\n=== All smoke tests passed ===\n";
    return 0;
}
