#include "audio/AudioAnalyzer.h"
#include "audio/BeatGrid.h"
#include "video/VideoProcessor.h"
#include "video/VideoWriter.h"
#include "tracing/Tracing.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cstring>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <random>

namespace fs = std::filesystem;
using namespace BeatSync;

void printUsage(const char* programName) {
    std::cout << "BeatSync Editor - Video & Audio Beat Sync Tool\n";
    std::cout << "Version 1.0.0 - Phase 2: Video Processing\n\n";
    std::cout << "Usage:\n";
    std::cout << "  " << programName << " <command> [options]\n\n";
    std::cout << "Commands:\n";
    std::cout << "  analyze    Analyze audio file and detect beats\n";
    std::cout << "  sync       Sync video with audio beats\n";
    std::cout << "  multiclip  Create beat-synced video from multiple clips\n";
    std::cout << "  split      Split video at beat timestamps\n\n";
    std::cout << "Options:\n";
    std::cout << "  -s, --sensitivity <value>   Beat detection sensitivity (0.0-1.0, default: 0.5)\n";
    std::cout << "  -d, --duration <seconds>    Clip duration for sync (default: auto)\n";
    std::cout << "  -o, --output <file>         Output file path\n";
    std::cout << "  -h, --help                  Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << programName << " analyze song.mp3\n";
    std::cout << "  " << programName << " sync video.mp4 audio.wav -o output.mp4\n";
    std::cout << "  " << programName << " multiclip clips_folder audio.wav -o output.mp4\n";
    std::cout << "  " << programName << " multiclip clips_folder audio.wav --duration 0.5\n";
    std::cout << "  " << programName << " split video.mp4 audio.wav -o \"clip_%03d.mp4\"\n";
}

int main(int argc, char* argv[]) {
    // Initialize tracing for CLI
    Tracing::ScopedInit _tracerInit("beatsync");

    // Check arguments
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string command = argv[1];

    // Help command
    if (command == "-h" || command == "--help") {
        printUsage(argv[0]);
        return 0;
    }

    // Analyze command
    if (command == "analyze") {
        if (argc < 3) {
            std::cerr << "Error: Missing audio file path\n\n";
            printUsage(argv[0]);
            return 1;
        }

        std::string audioFile = argv[2];
        double sensitivity = 0.5;

        // Parse optional arguments
        for (int i = 3; i < argc; ++i) {
            if (std::strcmp(argv[i], "-s") == 0 || std::strcmp(argv[i], "--sensitivity") == 0) {
                if (i + 1 < argc) {
                    try {
                        sensitivity = std::stod(argv[i + 1]);
                        if (sensitivity < 0.0 || sensitivity > 1.0) {
                            std::cerr << "Error: Sensitivity must be between 0.0 and 1.0\n";
                            return 1;
                        }
                        ++i;  // Skip next argument
                    } catch (...) {
                        std::cerr << "Error: Invalid sensitivity value\n";
                        return 1;
                    }
                } else {
                    std::cerr << "Error: --sensitivity requires a value\n";
                    return 1;
                }
            }
        }

        // Perform analysis
        std::cout << "========================================\n";
        std::cout << "BeatSync Audio Analyzer\n";
        std::cout << "========================================\n\n";
        std::cout << "Analyzing: " << audioFile << "\n";
        std::cout << "Sensitivity: " << sensitivity << "\n\n";

        AudioAnalyzer analyzer;
        analyzer.setSensitivity(sensitivity);

        BeatGrid beatGrid = analyzer.analyze(audioFile);

        if (beatGrid.isEmpty()) {
            std::cerr << "\nError: " << analyzer.getLastError() << "\n";
            return 1;
        }

        // Print results
        std::cout << "\n========================================\n";
        std::cout << "Analysis Results\n";
        std::cout << "========================================\n\n";
        std::cout << beatGrid.toString() << "\n";

        // Print first 20 beats with timestamps
        if (beatGrid.getNumBeats() > 0) {
            std::cout << "Beat Timestamps (first 20):\n";
            std::cout << std::fixed << std::setprecision(3);

            size_t limit = std::min(static_cast<size_t>(20), beatGrid.getNumBeats());
            for (size_t i = 0; i < limit; ++i) {
                double timestamp = beatGrid.getBeatAt(i);
                int minutes = static_cast<int>(timestamp) / 60;
                double seconds = timestamp - (minutes * 60);

                std::cout << "  Beat " << std::setw(3) << (i + 1) << ": "
                         << std::setw(2) << minutes << ":"
                         << std::setw(6) << seconds << " ("
                         << timestamp << "s)\n";
            }

            if (beatGrid.getNumBeats() > 20) {
                std::cout << "  ... and " << (beatGrid.getNumBeats() - 20) << " more beats\n";
            }
        }

        std::cout << "\n========================================\n";
        std::cout << "Analysis complete!\n";
        std::cout << "========================================\n";

        return 0;
    }

    // Sync command
    if (command == "sync") {
        if (argc < 4) {
            std::cerr << "Error: Missing video or audio file\n\n";
            std::cout << "Usage: " << argv[0] << " sync <video_file> <audio_file> [options]\n";
            return 1;
        }

        std::string videoFile = argv[2];
        std::string audioFile = argv[3];
        std::string outputFile = "output_synced.mp4";
        double sensitivity = 0.5;
        double clipDuration = 0.0; // 0 = auto (until next beat)

        // Parse options
        for (int i = 4; i < argc; ++i) {
            if ((std::strcmp(argv[i], "-o") == 0 || std::strcmp(argv[i], "--output") == 0) && i + 1 < argc) {
                outputFile = argv[++i];
            } else if ((std::strcmp(argv[i], "-s") == 0 || std::strcmp(argv[i], "--sensitivity") == 0) && i + 1 < argc) {
                sensitivity = std::stod(argv[++i]);
            } else if ((std::strcmp(argv[i], "-d") == 0 || std::strcmp(argv[i], "--duration") == 0) && i + 1 < argc) {
                clipDuration = std::stod(argv[++i]);
            }
        }

        std::cout << "========================================\n";
        std::cout << "BeatSync Video Sync\n";
        std::cout << "========================================\n\n";
        std::cout << "Video: " << videoFile << "\n";
        std::cout << "Audio: " << audioFile << "\n";
        std::cout << "Output: " << outputFile << "\n";
        std::cout << "Sensitivity: " << sensitivity << "\n";
        std::cout << "Clip duration: " << (clipDuration > 0 ? std::to_string(clipDuration) + "s" : "auto") << "\n\n";

        // Analyze audio
        std::cout << "Step 1: Analyzing audio for beats...\n";
        AudioAnalyzer analyzer;
        analyzer.setSensitivity(sensitivity);
        BeatGrid beatGrid = analyzer.analyze(audioFile);

        if (beatGrid.isEmpty()) {
            std::cerr << "Error: " << analyzer.getLastError() << "\n";
            return 1;
        }

        std::cout << "Found " << beatGrid.getNumBeats() << " beats at " << beatGrid.getBPM() << " BPM\n\n";

        // Sync video
        std::cout << "Step 2: Syncing video with beats...\n";
        std::string tempVideoOnly = "beatsync_sync_temp.mp4";
        VideoWriter writer;
        writer.setProgressCallback([](double progress) {
            int percent = static_cast<int>(progress * 100);
            std::cout << "\rProgress: " << percent << "%" << std::flush;
        });

        if (!writer.cutAtBeats(videoFile, beatGrid, tempVideoOnly, clipDuration)) {
            std::cerr << "\nError: " << writer.getLastError() << "\n";
            std::remove(tempVideoOnly.c_str());
            return 1;
        }

        // Step 3: Add the audio track
        std::cout << "\n\nStep 3: Adding audio track...\n";
        if (!writer.addAudioTrack(tempVideoOnly, audioFile, outputFile, true)) {
            std::cerr << "Error: " << writer.getLastError() << "\n";
            std::remove(tempVideoOnly.c_str());
            return 1;
        }

        // Cleanup temp file
        std::remove(tempVideoOnly.c_str());

        std::cout << "\n========================================\n";
        std::cout << "Sync complete!\n";
        std::cout << "Output: " << outputFile << "\n";
        std::cout << "Audio: " << audioFile << "\n";
        std::cout << "========================================\n";

        return 0;
    }

    // Split command
    if (command == "split") {
        if (argc < 4) {
            std::cerr << "Error: Missing video or audio file\n\n";
            std::cout << "Usage: " << argv[0] << " split <video_file> <audio_file> [options]\n";
            return 1;
        }

        std::string videoFile = argv[2];
        std::string audioFile = argv[3];
        std::string outputPattern = "clip_%03d.mp4";
        double sensitivity = 0.5;

        // Parse options
        for (int i = 4; i < argc; ++i) {
            if ((std::strcmp(argv[i], "-o") == 0 || std::strcmp(argv[i], "--output") == 0) && i + 1 < argc) {
                outputPattern = argv[++i];
            } else if ((std::strcmp(argv[i], "-s") == 0 || std::strcmp(argv[i], "--sensitivity") == 0) && i + 1 < argc) {
                sensitivity = std::stod(argv[++i]);
            }
        }

        std::cout << "========================================\n";
        std::cout << "BeatSync Video Split\n";
        std::cout << "========================================\n\n";
        std::cout << "Video: " << videoFile << "\n";
        std::cout << "Audio: " << audioFile << "\n";
        std::cout << "Output pattern: " << outputPattern << "\n";
        std::cout << "Sensitivity: " << sensitivity << "\n\n";

        // Analyze audio
        std::cout << "Step 1: Analyzing audio for beats...\n";
        AudioAnalyzer analyzer;
        analyzer.setSensitivity(sensitivity);
        BeatGrid beatGrid = analyzer.analyze(audioFile);

        if (beatGrid.isEmpty()) {
            std::cerr << "Error: " << analyzer.getLastError() << "\n";
            return 1;
        }

        std::cout << "Found " << beatGrid.getNumBeats() << " beats\n\n";

        // Split video
        std::cout << "Step 2: Splitting video at beats...\n";
        VideoWriter writer;

        if (!writer.splitVideo(videoFile, beatGrid.getBeats(), outputPattern)) {
            std::cerr << "Error: " << writer.getLastError() << "\n";
            return 1;
        }

        std::cout << "\n========================================\n";
        std::cout << "Split complete!\n";
        std::cout << "========================================\n";

        return 0;
    }

    // Multiclip command
    if (command == "multiclip") {
        if (argc < 4) {
            std::cerr << "Error: Missing clips folder or audio file\n\n";
            std::cout << "Usage: " << argv[0] << " multiclip <clips_folder> <audio_file> [options]\n";
            return 1;
        }

        std::string clipsFolder = argv[2];
        std::string audioFile = argv[3];
        std::string outputFile = "output_multiclip.mp4";
        double sensitivity = 0.5;
        double clipDuration = 0.0; // 0 = auto (until next beat)

        // Parse options
        for (int i = 4; i < argc; ++i) {
            if ((std::strcmp(argv[i], "-o") == 0 || std::strcmp(argv[i], "--output") == 0) && i + 1 < argc) {
                outputFile = argv[++i];
            } else if ((std::strcmp(argv[i], "-s") == 0 || std::strcmp(argv[i], "--sensitivity") == 0) && i + 1 < argc) {
                sensitivity = std::stod(argv[++i]);
            } else if ((std::strcmp(argv[i], "-d") == 0 || std::strcmp(argv[i], "--duration") == 0) && i + 1 < argc) {
                clipDuration = std::stod(argv[++i]);
            }
        }

        std::cout << "========================================\n";
        std::cout << "BeatSync Multi-Clip Sync\n";
        std::cout << "========================================\n\n";

        // Find all video files in folder
        std::vector<std::string> videoFiles;
        try {
            for (const auto& entry : fs::directory_iterator(clipsFolder)) {
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (ext == ".mp4" || ext == ".avi" || ext == ".mov" || ext == ".mkv") {
                        videoFiles.push_back(entry.path().string());
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error reading clips folder: " << e.what() << "\n";
            return 1;
        }

        if (videoFiles.empty()) {
            std::cerr << "Error: No video files found in " << clipsFolder << "\n";
            return 1;
        }

        // Sort for consistent ordering
        std::sort(videoFiles.begin(), videoFiles.end());

        std::cout << "Found " << videoFiles.size() << " video clips\n";
        std::cout << "Audio: " << audioFile << "\n";
        std::cout << "Output: " << outputFile << "\n";
        std::cout << "Sensitivity: " << sensitivity << "\n";
        std::cout << "Clip duration: " << (clipDuration > 0 ? std::to_string(clipDuration) + "s" : "auto") << "\n\n";

        // Analyze audio
        std::cout << "Step 1: Analyzing audio for beats...\n";
        AudioAnalyzer analyzer;
        analyzer.setSensitivity(sensitivity);
        BeatGrid beatGrid = analyzer.analyze(audioFile);

        if (beatGrid.isEmpty()) {
            std::cerr << "Error: " << analyzer.getLastError() << "\n";
            return 1;
        }

        std::cout << "Found " << beatGrid.getNumBeats() << " beats at " << beatGrid.getBPM() << " BPM\n\n";

        // Get audio duration for padding
        double audioDuration = beatGrid.getAudioDuration();

        // Create segments from beats, cycling through video clips
        std::cout << "Step 2: Creating beat-synced segments...\n";
        std::vector<std::string> tempFiles;
        const auto& beats = beatGrid.getBeats();
        double totalVideoDuration = 0.0;

        for (size_t i = 0; i < beats.size(); ++i) {
            // Pick video file (cycle through clips)
            std::string videoFile = videoFiles[i % videoFiles.size()];

            double startTime = beats[i];
            double duration;

            if (clipDuration > 0) {
                duration = clipDuration;
            } else if (i + 1 < beats.size()) {
                duration = beats[i + 1] - beats[i];
            } else {
                // Last beat: extend to end of audio instead of fixed 2 seconds
                duration = audioDuration - beats[i];
                if (duration < 0.5) duration = 0.5;  // Minimum duration
            }

            totalVideoDuration += duration;

            // Create temp filename
            std::ostringstream tempFile;
            tempFile << "beatsync_multi_" << std::setw(5) << std::setfill('0') << i << ".mp4";

            if (i % 100 == 0 || i == beats.size() - 1) {
                std::cout << "\rProcessing beat " << (i + 1) << "/" << beats.size() << std::flush;
            }

            // Extract a segment from the beginning of the clip
            // (We'll use the first 'duration' seconds of each clip)
            VideoWriter writer;
            if (!writer.copySegmentFast(videoFile, 0.0, duration, tempFile.str())) {
                // If fast copy fails, try with a small offset
                if (!writer.copySegmentFast(videoFile, 0.1, duration, tempFile.str())) {
                    std::cerr << "\nWarning: Failed to extract segment from " << videoFile << "\n";
                    continue;
                }
            }

            tempFiles.push_back(tempFile.str());
        }

        // Check if we need padding to match audio duration
        double remainingTime = audioDuration - totalVideoDuration;
        if (remainingTime > 0.5) {
            std::cout << "\n  Adding " << remainingTime << "s padding to match audio length...\n";

            // Add padding segments by cycling through clips
            size_t padIndex = beats.size();
            while (remainingTime > 0.1) {
                std::string videoFile = videoFiles[padIndex % videoFiles.size()];
                double duration = std::min(remainingTime, 2.0);  // Max 2 second segments for padding

                std::ostringstream tempFile;
                tempFile << "beatsync_multi_" << std::setw(5) << std::setfill('0') << padIndex << ".mp4";

                VideoWriter writer;
                if (writer.copySegmentFast(videoFile, 0.0, duration, tempFile.str())) {
                    tempFiles.push_back(tempFile.str());
                    remainingTime -= duration;
                } else {
                    // Try next clip if this one fails
                }
                padIndex++;

                // Safety limit to prevent infinite loop
                if (padIndex > beats.size() + 1000) break;
            }
        }

        std::cout << "\n\nStep 3: Concatenating " << tempFiles.size() << " segments...\n";

        // Concatenate all segments to a temporary file first (video only)
        std::string tempVideoOnly = "beatsync_video_only_temp.mp4";
        VideoWriter writer;
        writer.setProgressCallback([](double progress) {
            int percent = static_cast<int>(progress * 100);
            std::cout << "\rProgress: " << percent << "%" << std::flush;
        });

        bool success = writer.concatenateVideos(tempFiles, tempVideoOnly);

        // Cleanup segment temp files
        for (const auto& f : tempFiles) {
            std::remove(f.c_str());
        }

        if (!success) {
            std::cerr << "\nError: " << writer.getLastError() << "\n";
            std::remove(tempVideoOnly.c_str());
            return 1;
        }

        // Step 4: Add the audio track (don't trim - video should match audio length now)
        std::cout << "\n\nStep 4: Adding audio track...\n";
        success = writer.addAudioTrack(tempVideoOnly, audioFile, outputFile, false);

        // Cleanup temp video-only file
        std::remove(tempVideoOnly.c_str());

        if (!success) {
            std::cerr << "Error: " << writer.getLastError() << "\n";
            return 1;
        }

        std::cout << "\n========================================\n";
        std::cout << "Multi-clip sync complete!\n";
        std::cout << "Output: " << outputFile << "\n";
        std::cout << "Used " << videoFiles.size() << " different clips\n";
        std::cout << "Created " << tempFiles.size() << " beat-synced segments\n";
        std::cout << "Audio: " << audioFile << "\n";
        std::cout << "========================================\n";

        return 0;
    }

    // Unknown command
    std::cerr << "Error: Unknown command '" << command << "'\n\n";
    printUsage(argv[0]);
    return 1;
}
