#include "VideoWriter.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <chrono>
#include <ctime>
#include <cstring>

// Cross-platform popen/pclose
#ifdef _WIN32
#define popen_compat _popen
#define pclose_compat _pclose
#else
#define popen_compat popen
#define pclose_compat pclose
#endif

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/timestamp.h>
}

namespace BeatSync {

namespace {
// Lightweight logger to capture FFmpeg command, exit code, and recent output.
void appendFfmpegLog(const std::string& logFile,
                     const std::string& label,
                     const std::string& command,
                     int exitCode,
                     const std::string& output,
                     const std::string& extra = "") {
    FILE* log = fopen(logFile.c_str(), "a");
    if (!log) {
        return;
    }

    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char timeBuf[64] = {0};
#ifdef _WIN32
    ctime_s(timeBuf, sizeof(timeBuf), &now);
#else
    std::strftime(timeBuf, sizeof(timeBuf), "%c", std::localtime(&now));
#endif

    // Trim trailing newline from ctime output
    size_t len = std::strlen(timeBuf);
    if (len > 0 && timeBuf[len - 1] == '\n') {
        timeBuf[len - 1] = '\0';
    }

    fprintf(log, "\n[%s] %s\n", timeBuf, label.c_str());
    fprintf(log, "cmd: %s\n", command.c_str());
    fprintf(log, "exit: %d\n", exitCode);
    if (!extra.empty()) {
        fprintf(log, "extra: %s\n", extra.c_str());
    }

    // Avoid huge logs by only keeping the tail of the output if very long.
    const size_t maxTail = 4000;
    if (output.size() <= maxTail) {
        fprintf(log, "output:\n%s\n", output.c_str());
    } else {
        fprintf(log, "output (last %zu chars):\n%s\n", maxTail, output.substr(output.size() - maxTail).c_str());
    }

    fclose(log);
}
} // namespace

VideoWriter::VideoWriter()
{
}

VideoWriter::~VideoWriter() {
}

std::string VideoWriter::resolveFfmpegPath() const {
    return getFFmpegPath();
}

std::string VideoWriter::getFFmpegPath() const {
    // 1. Check environment variable first
    const char* envPath = std::getenv("BEATSYNC_FFMPEG_PATH");
    if (envPath != nullptr && envPath[0] != '\0') {
        return envPath;
    }

    // 2. Try to find ffmpeg in PATH
#ifdef _WIN32
    FILE* pipe = popen_compat("where ffmpeg 2>nul", "r");
#else
    FILE* pipe = popen("which ffmpeg 2>/dev/null", "r");
#endif
    if (pipe) {
        char buffer[512];
        std::string result;
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            result += buffer;
        }
#ifdef _WIN32
        pclose_compat(pipe);
#else
        pclose(pipe);
#endif

        // Get first line (first match)
        size_t newline = result.find('\n');
        if (newline != std::string::npos) {
            result = result.substr(0, newline);
        }

        // If we found something, use it
        if (!result.empty() && result.find("ffmpeg") != std::string::npos) {
            return result;
        }
    }

    // 3. Fall back to platform-specific hardcoded path
#ifdef _WIN32
    return "C:\\ffmpeg-dev\\ffmpeg-master-latest-win64-gpl-shared\\bin\\ffmpeg.exe";
#elif defined(__APPLE__)
    return "/opt/homebrew/bin/ffmpeg";
#else
    return "/usr/bin/ffmpeg";
#endif
}

bool VideoWriter::cutAtBeats(const std::string& inputVideo,
                             const BeatGrid& beatGrid,
                             const std::string& outputVideo,
                             double clipDuration) {
    if (beatGrid.isEmpty()) {
        m_lastError = "Beat grid is empty";
        return false;
    }

    // Create segments from beats
    std::vector<VideoSegment> segments;
    const auto& beats = beatGrid.getBeats();

    for (size_t i = 0; i < beats.size(); ++i) {
        VideoSegment seg;
        seg.startTime = beats[i];

        if (clipDuration > 0) {
            seg.endTime = beats[i] + clipDuration;
        } else if (i + 1 < beats.size()) {
            seg.endTime = beats[i + 1];
        } else {
            // Last beat - use a default duration
            seg.endTime = beats[i] + 2.0;
        }

        segments.push_back(seg);
    }

    return extractSegments(inputVideo, segments, outputVideo);
}

bool VideoWriter::extractSegments(const std::string& inputVideo,
                                  const std::vector<VideoSegment>& segments,
                                  const std::string& outputVideo) {
    m_lastError.clear();

    if (segments.empty()) {
        m_lastError = "No segments to extract";
        return false;
    }

    // Create temporary files for each segment
    std::vector<std::string> tempFiles;
    char tempPattern[] = "beatsync_temp_XXXXXX.mp4";

    std::cout << "Extracting " << segments.size() << " segments...\n";

    for (size_t i = 0; i < segments.size(); ++i) {
        const auto& seg = segments[i];
        double duration = seg.endTime - seg.startTime;

        // Create temp filename
        std::ostringstream tempFile;
        tempFile << "beatsync_segment_" << std::setw(5) << std::setfill('0') << i << ".mp4";

        std::cout << "  Segment " << (i + 1) << "/" << segments.size()
                  << ": " << seg.startTime << "s - " << seg.endTime << "s\n";

        if (!copySegmentFast(inputVideo, seg.startTime, duration, tempFile.str())) {
            // Fallback to precise method if fast fails
            if (!copySegmentPrecise(inputVideo, seg.startTime, duration, tempFile.str())) {
                // Cleanup temp files
                for (const auto& f : tempFiles) {
                    std::remove(f.c_str());
                }
                return false;
            }
        }

        tempFiles.push_back(tempFile.str());

        if (m_progressCallback) {
            reportProgress((i + 1) / (double)segments.size() * 0.9);
        }
    }

    // Concatenate all segments
    std::cout << "Concatenating segments...\n";
    bool result = concatenateVideos(tempFiles, outputVideo);

    // Cleanup temp files
    for (const auto& f : tempFiles) {
        std::remove(f.c_str());
    }

    if (m_progressCallback) {
        reportProgress(1.0);
    }

    return result;
}

bool VideoWriter::splitVideo(const std::string& inputVideo,
                             const std::vector<double>& timestamps,
                             const std::string& outputPattern) {
    m_lastError.clear();

    if (timestamps.empty()) {
        m_lastError = "No timestamps provided";
        return false;
    }

    VideoProcessor processor;
    if (!processor.open(inputVideo)) {
        m_lastError = "Could not open input video: " + processor.getLastError();
        return false;
    }

    VideoInfo info = processor.getInfo();
    processor.close();

    // Create segments between timestamps
    std::vector<double> splitPoints = timestamps;
    splitPoints.insert(splitPoints.begin(), 0.0);
    splitPoints.push_back(info.duration);

    std::cout << "Splitting video into " << (splitPoints.size() - 1) << " parts...\n";

    for (size_t i = 0; i < splitPoints.size() - 1; ++i) {
        double start = splitPoints[i];
        double duration = splitPoints[i + 1] - start;

        // Create output filename
        char outFile[512];
        snprintf(outFile, sizeof(outFile), outputPattern.c_str(), (int)i);

        std::cout << "  Part " << (i + 1) << ": " << start << "s - "
                  << (start + duration) << "s -> " << outFile << "\n";

        if (!copySegmentFast(inputVideo, start, duration, outFile)) {
            if (!copySegmentPrecise(inputVideo, start, duration, outFile)) {
                return false;
            }
        }

        if (m_progressCallback) {
            reportProgress((i + 1) / (double)(splitPoints.size() - 1));
        }
    }

    return true;
}

std::string VideoWriter::getLastError() const {
    return m_lastError;
}

void VideoWriter::setProgressCallback(std::function<void(double)> callback) {
    m_progressCallback = callback;
}

void VideoWriter::setOutputSettings(int width, int height, int fps) {
    m_outputWidth = width;
    m_outputHeight = height;
    m_outputFps = fps;
}

bool VideoWriter::copySegmentFast(const std::string& inputVideo,
                                   double startTime,
                                   double duration,
                                   const std::string& outputVideo) {
    std::cout << "Extracting segment: " << inputVideo << " @ " << startTime << "s for " << duration << "s -> " << outputVideo << "\n";

    // Use FFmpeg command-line for reliable segment extraction
    // Note: popen_compat() on Windows passes commands to cmd.exe, so we need proper quote escaping
    //
    // FIX: Normalize ALL clips to same resolution (1920x1080), frame rate (24fps),
    // and pixel format to prevent freezing from mixed source formats
    std::string ffmpegPath = getFFmpegPath();
    std::ostringstream cmd;
    cmd << "\"" << ffmpegPath << "\" -i \"" << inputVideo << "\""
        << " -ss " << startTime
        << " -t " << duration
        << " -vf \"scale=" << m_outputWidth << ":" << m_outputHeight
        << ":force_original_aspect_ratio=decrease,pad=" << m_outputWidth << ":" << m_outputHeight
        << ":(ow-iw)/2:(oh-ih)/2,setsar=1,fps=" << m_outputFps << "\""
        << " -c:v libx264 -preset ultrafast -crf 18 -pix_fmt yuv420p"
        << " -c:a aac -b:a 192k -ar 44100"
        << " -video_track_timescale 90000"
        << " -avoid_negative_ts make_zero"
        << " -y \"" << outputVideo << "\"";

    // DEBUG: Print command for first failure
    static int debugCount = 0;
    bool shouldDebug = (debugCount < 2);

    // Execute FFmpeg (redirect stderr to stdout to capture all output)
#ifdef _WIN32
    // Wrap with cmd /C to avoid PowerShell/cmd parsing quirks when filters contain parentheses/commas.
    std::string fullCmd = "cmd /C \"" + cmd.str() + " 2>&1\"";
#else
    std::string fullCmd = cmd.str() + " 2>&1";
#endif
    FILE* pipe = popen_compat(fullCmd.c_str(), "r");
    if (!pipe) {
        m_lastError = "Failed to execute FFmpeg";
        appendFfmpegLog("beatsync_ffmpeg_extract.log", "copySegmentFast::popen_compat_failed", fullCmd, -1, "", "start=" + std::to_string(startTime) + ", dur=" + std::to_string(duration));
        return false;
    }

    // Read output
    char buffer[256];
    std::string ffmpegOutput;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        ffmpegOutput += buffer;
    }

    int exitCode = pclose_compat(pipe);

    // Check file size regardless of exit code for better diagnostics
    long fileSize = -1;
    {
        FILE* test = fopen(outputVideo.c_str(), "rb");
        if (test) {
            fseek(test, 0, SEEK_END);
            fileSize = ftell(test);
            fclose(test);
        }
    }

    if (exitCode != 0) {
        m_lastError = "Segment extraction failed";
        std::string extra = "start=" + std::to_string(startTime) + ", dur=" + std::to_string(duration) + ", size=" + std::to_string(fileSize);
        appendFfmpegLog("beatsync_ffmpeg_extract.log", "copySegmentFast", cmd.str(), exitCode, ffmpegOutput, extra);

        // Check if it's a genuine error or just warnings
        if (ffmpegOutput.find("Output file is empty") != std::string::npos ||
            ffmpegOutput.find("No such file or directory") != std::string::npos ||
            ffmpegOutput.find("Invalid data found") != std::string::npos) {
            if (shouldDebug) {
                std::cerr << "\nDEBUG Failed extraction #" << (++debugCount) << ":\n";
                std::cerr << "Command: " << cmd.str() << "\n";
                std::cerr << "Exit code: " << exitCode << "\n";
                std::cerr << "Output: " << ffmpegOutput << "\n";
            }
            return false;
        }
        // FFmpeg often returns non-zero for warnings, but file might still be created
        // Check if output file exists AND has content
        if (fileSize > 1024) {  // Minimum viable video file (1KB)
            return true;  // File was created with content despite non-zero exit
        }
        if (shouldDebug) {
            std::cerr << "\nDEBUG Failed extraction #" << (++debugCount) << ":\n";
            std::cerr << "Command: " << cmd.str() << "\n";
            std::cerr << "Exit code: " << exitCode << "\n";
            std::cerr << "File check failed for: " << outputVideo << "\n";
            std::cerr << "Output (last 500 chars): " << ffmpegOutput.substr(ffmpegOutput.length() > 500 ? ffmpegOutput.length() - 500 : 0) << "\n";
        }
        return false;
    }

    // If exit code is zero but file is suspiciously small, log it for debugging.
    if (fileSize >= 0 && fileSize <= 1024) {
        std::string extra = "start=" + std::to_string(startTime) + ", dur=" + std::to_string(duration) + ", size=" + std::to_string(fileSize);
        appendFfmpegLog("beatsync_ffmpeg_extract.log", "copySegmentFast_small_file", cmd.str(), exitCode, ffmpegOutput, extra);
    }

    return true;
}

bool VideoWriter::copySegmentPrecise(const std::string& inputVideo,
                                     double startTime,
                                     double duration,
                                     const std::string& outputVideo) {
    // Use FFmpeg with re-encoding for frame-accurate extraction
    // This is slower but more precise than stream copy
    //
    // FIX: Normalize ALL clips to same resolution, frame rate, and pixel format
    std::string ffmpegPath = getFFmpegPath();
    std::ostringstream cmd;
    cmd << "\"" << ffmpegPath << "\" -i \"" << inputVideo << "\""
        << " -ss " << startTime
        << " -t " << duration
        << " -vf \"scale=" << m_outputWidth << ":" << m_outputHeight
        << ":force_original_aspect_ratio=decrease,pad=" << m_outputWidth << ":" << m_outputHeight
        << ":(ow-iw)/2:(oh-ih)/2,setsar=1,fps=" << m_outputFps << "\""
        << " -c:v libx264 -preset ultrafast -crf 18 -pix_fmt yuv420p"
        << " -c:a aac -b:a 192k -ar 44100"
        << " -video_track_timescale 90000"
        << " -avoid_negative_ts make_zero"
        << " -y \"" << outputVideo << "\"";

    // Execute FFmpeg (redirect stderr to stdout to capture all output)
#ifdef _WIN32
    std::string fullCmd = "cmd /C \"" + cmd.str() + " 2>&1\"";
#else
    std::string fullCmd = cmd.str() + " 2>&1";
#endif
    FILE* pipe = popen_compat(fullCmd.c_str(), "r");
    if (!pipe) {
        m_lastError = "Failed to execute FFmpeg for precise copy";
        appendFfmpegLog("beatsync_ffmpeg_extract.log", "copySegmentPrecise::popen_compat_failed", fullCmd, -1, "", "start=" + std::to_string(startTime) + ", dur=" + std::to_string(duration));
        return false;
    }

    // Read output
    char buffer[256];
    std::string ffmpegOutput;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        ffmpegOutput += buffer;
    }

    int exitCode = pclose_compat(pipe);

    long fileSize = -1;
    {
        FILE* test = fopen(outputVideo.c_str(), "rb");
        if (test) {
            fseek(test, 0, SEEK_END);
            fileSize = ftell(test);
            fclose(test);
        }
    }

    if (exitCode != 0) {
        m_lastError = "Precise segment extraction failed";
        std::string extra = "start=" + std::to_string(startTime) + ", dur=" + std::to_string(duration) + ", size=" + std::to_string(fileSize);
        appendFfmpegLog("beatsync_ffmpeg_extract.log", "copySegmentPrecise", cmd.str(), exitCode, ffmpegOutput, extra);
        if (fileSize > 1024) {
            return true;  // File was created despite non-zero exit
        }
        return false;
    }

    if (fileSize >= 0 && fileSize <= 1024) {
        std::string extra = "start=" + std::to_string(startTime) + ", dur=" + std::to_string(duration) + ", size=" + std::to_string(fileSize);
        appendFfmpegLog("beatsync_ffmpeg_extract.log", "copySegmentPrecise_small_file", cmd.str(), exitCode, ffmpegOutput, extra);
    }

    return true;
}

bool VideoWriter::concatenateVideos(const std::vector<std::string>& inputVideos,
                                   const std::string& outputVideo) {
    if (inputVideos.empty()) {
        m_lastError = "No input videos to concatenate";
        return false;
    }

    // Create concat list file
    std::string listFile = "beatsync_concat_list.txt";
    FILE* f = fopen(listFile.c_str(), "w");
    if (!f) {
        m_lastError = "Could not create concat list file";
        return false;
    }

    // Log what we're concatenating for debugging
    std::cout << "Creating concat list with " << inputVideos.size() << " videos:\n";

    FILE* debugLog = fopen("tripsitter_debug.log", "a");
    if (debugLog) {
        fprintf(debugLog, "\n=== Concatenation Step ===\n");
        fprintf(debugLog, "Creating concat list with %zu videos:\n", inputVideos.size());
        fclose(debugLog);
    }

    int missingCount = 0;
    for (const auto& video : inputVideos) {
        fprintf(f, "file '%s'\n", video.c_str());
        std::cout << "  - " << video;

        // Check if file exists
        FILE* check = fopen(video.c_str(), "rb");
        if (!check) {
            std::cout << " [MISSING!]\n";
            missingCount++;

            debugLog = fopen("tripsitter_debug.log", "a");
            if (debugLog) {
                fprintf(debugLog, "  - %s [MISSING!]\n", video.c_str());
                fclose(debugLog);
            }
        } else {
            // Get file size
            fseek(check, 0, SEEK_END);
            long size = ftell(check);
            std::cout << " [OK, " << size << " bytes]\n";
            fclose(check);

            debugLog = fopen("tripsitter_debug.log", "a");
            if (debugLog) {
                fprintf(debugLog, "  - %s [OK, %ld bytes]\n", video.c_str(), size);
                fclose(debugLog);
            }
        }
    }
    fclose(f);

    if (missingCount > 0) {
        m_lastError = "Cannot concatenate: " + std::to_string(missingCount) + " segment files are missing!";

        debugLog = fopen("tripsitter_debug.log", "a");
        if (debugLog) {
            fprintf(debugLog, "ERROR: %d segment files are missing!\n", missingCount);
            fclose(debugLog);
        }

        std::remove(listFile.c_str());
        return false;
    }

    // Use FFmpeg command-line to concatenate pre-normalized segments
    // Since all segments are now normalized to same resolution/fps/format,
    // we prefer stream copy for fast concatenation but capture FFmpeg output
    // and fall back to a re-encode if we detect timestamp/PTS/DTS problems.
    std::string ffmpegPath = getFFmpegPath();
    std::ostringstream cmd;
    cmd << "\"" << ffmpegPath << "\" -fflags +genpts+igndts -f concat -safe 0 -i \"" << listFile
        << "\" -c copy -video_track_timescale 90000 -y \"" << outputVideo << "\"";

    // Execute FFmpeg and capture output (redirect stderr to stdout)
#ifdef _WIN32
    std::string fullCmd = "cmd /C \"" + cmd.str() + " 2>&1\"";
#else
    std::string fullCmd = cmd.str() + " 2>&1";
#endif
    FILE* pipe = popen_compat(fullCmd.c_str(), "r");
    if (!pipe) {
        m_lastError = "Failed to execute FFmpeg";
        std::remove(listFile.c_str());
        return false;
    }

    char buffer[512];
    std::string ffmpegOutput;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        ffmpegOutput += buffer;
    }

    int exitCode = pclose_compat(pipe);

    // Persist FFmpeg output for debugging
    {
        FILE* logf = fopen("beatsync_ffmpeg_concat.log", "a");
        if (logf) {
            fprintf(logf, "\n--- FFmpeg concat run ---\ncmd: %s\nexit: %d\noutput:\n%s\n", cmd.str().c_str(), exitCode, ffmpegOutput.c_str());
            fclose(logf);
        }
    }

    // Check for suspicious warnings that can indicate timestamp/frame issues
    // NOTE: Don't delete list file yet - fallback re-encode might need it
    bool suspicious = false;
    const char* patterns[] = {
        "Non-monotonic DTS",
        "Non monotonic DTS",
        "Invalid pts",
        "Non-monotonic PTS",
        "Non monotonic PTS",
        "Dropping frame",
        "duplicate",
        "Output file is empty",
        "Error while decoding",
        "invalid pts"
    };
    for (const char* p : patterns) {
        if (ffmpegOutput.find(p) != std::string::npos) {
            suspicious = true;
            break;
        }
    }

    if (exitCode != 0 || suspicious) {
        // Attempt a safe re-encode fallback (slower but normalizes timestamps)
        std::ostringstream reencodeCmd;
        reencodeCmd << "\"" << ffmpegPath << "\" -fflags +genpts -f concat -safe 0 -i \"" << listFile
                    << "\" -c:v libx264 -preset ultrafast -crf 18 -r " << m_outputFps << " -pix_fmt yuv420p"
                    << " -c:a aac -b:a 192k -video_track_timescale 90000 -y \"" << outputVideo << "\"";

        // Execute re-encode (redirect stderr to stdout)
    #ifdef _WIN32
        std::string fullReencodeCmd = "cmd /C \"" + reencodeCmd.str() + " 2>&1\"";
    #else
        std::string fullReencodeCmd = reencodeCmd.str() + " 2>&1";
    #endif
        FILE* pipe2 = popen_compat(fullReencodeCmd.c_str(), "r");
        std::string reencodeOutput;
        if (!pipe2) {
            m_lastError = "FFmpeg re-encode fallback failed to start";
            return false;
        }
        while (fgets(buffer, sizeof(buffer), pipe2) != nullptr) {
            reencodeOutput += buffer;
        }
        int rc2 = pclose_compat(pipe2);

        // Log re-encode output
        FILE* logf2 = fopen("beatsync_ffmpeg_concat.log", "a");
        if (logf2) {
            fprintf(logf2, "\n--- FFmpeg re-encode run ---\ncmd: %s\nexit: %d\noutput:\n%s\n", reencodeCmd.str().c_str(), rc2, reencodeOutput.c_str());
            fclose(logf2);
        }

        if (rc2 != 0) {
            m_lastError = "FFmpeg concatenation and re-encode both failed";
            // Attach the last line of the re-encode output for clearer debugging
            size_t lastNewline = reencodeOutput.rfind('\n');
            if (lastNewline != std::string::npos && lastNewline + 1 < reencodeOutput.size()) {
                m_lastError += ": " + reencodeOutput.substr(lastNewline + 1);
            }
            std::remove(listFile.c_str());  // Clean up before returning error
            return false;
        }

        // Re-encode succeeded - clean up list file
        std::remove(listFile.c_str());
        return true;
    }

    // Success without needing re-encode - clean up list file
    std::remove(listFile.c_str());
    return true;
}

bool VideoWriter::addAudioTrack(const std::string& inputVideo,
                                 const std::string& audioFile,
                                 const std::string& outputVideo,
                                 bool trimToShortest,
                                 double audioStart,
                                 double audioEnd) {
    m_lastError.clear();

    std::string ffmpegPath = getFFmpegPath();
    std::ostringstream cmd;

    // Combine video from first input with audio from second input
    // -c:v copy = stream copy video (fast, no re-encode)
    // -c:a aac = encode audio as AAC
    // -shortest = trim output to shorter of video/audio (optional)
    double clipStart = std::max(0.0, audioStart);
    bool clipAudio = (audioEnd > 0.0 && audioEnd > audioStart + 1e-3);
    double clipDur = clipAudio ? (audioEnd - audioStart) : 0.0;

    cmd << "\"" << ffmpegPath << "\" -i \"" << inputVideo << "\"";
    if (clipAudio) {
        cmd << " -ss " << clipStart << " -t " << clipDur;
    }
    cmd << " -i \"" << audioFile << "\""
        << " -c:v copy -c:a aac -b:a 192k"
        << " -map 0:v:0 -map 1:a:0";  // Take video from first input, audio from second

    if (trimToShortest) {
        cmd << " -shortest";
    }

    cmd << " -y \"" << outputVideo << "\"";

    std::cout << "Adding audio track...\n";

    // Execute FFmpeg (redirect stderr to stdout)
#ifdef _WIN32
    std::string fullCmd = "cmd /C \"" + cmd.str() + " 2>&1\"";
#else
    std::string fullCmd = cmd.str() + " 2>&1";
#endif
    FILE* pipe = popen_compat(fullCmd.c_str(), "r");
    if (!pipe) {
        m_lastError = "Failed to execute FFmpeg for audio muxing";
        return false;
    }

    // Read output
    char buffer[256];
    std::string ffmpegOutput;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        ffmpegOutput += buffer;
    }

    int exitCode = pclose_compat(pipe);

    // Persist muxing output for troubleshooting
    {
        FILE* logf = fopen("beatsync_ffmpeg_concat.log", "a");
        if (logf) {
            fprintf(logf, "\n--- FFmpeg audio mux run ---\ncmd: %s\nexit: %d\noutput:\n%s\n", cmd.str().c_str(), exitCode, ffmpegOutput.c_str());
            fclose(logf);
        }
    }

    if (exitCode != 0) {
        m_lastError = "FFmpeg audio muxing failed";
        // Check if output file was created anyway
        FILE* test = fopen(outputVideo.c_str(), "rb");
        if (test) {
            fclose(test);
            return true;  // File was created despite non-zero exit
        }
        return false;
    }

    return true;
}

void VideoWriter::reportProgress(double progress) {
    if (m_progressCallback) {
        m_progressCallback(progress);
    }
}

void VideoWriter::setEffectsConfig(const EffectsConfig& config) {
    m_effects = config;
}

std::string VideoWriter::getColorGradeFilter(const std::string& preset) const {
    if (preset == "warm") {
        return "colorbalance=rs=0.1:gs=0.05:bs=-0.05";
    } else if (preset == "cool") {
        return "colorbalance=rs=-0.05:gs=0.0:bs=0.1";
    } else if (preset == "vintage") {
        return "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131";
    } else if (preset == "vibrant") {
        return "eq=saturation=1.4:contrast=1.1";
    }
    return "";
}

std::string VideoWriter::buildEffectsFilterChain() const {
    std::vector<std::string> filters;

    // Color grading
    if (m_effects.enableColorGrade && m_effects.colorPreset != "none") {
        std::string colorFilter = getColorGradeFilter(m_effects.colorPreset);
        if (!colorFilter.empty()) {
            filters.push_back(colorFilter);
        }
    }

    // Vignette
    if (m_effects.enableVignette) {
        std::ostringstream vig;
        vig << "vignette=PI/" << (4.0 / m_effects.vignetteStrength);
        filters.push_back(vig.str());
    }

    // Blur
    if (m_effects.enableBlur) {
        std::ostringstream blur;
        blur << "gblur=sigma=" << m_effects.blurStrength;
        filters.push_back(blur.str());
    }

    // Beat zoom pulse effect
    if (m_effects.enableBeatZoom && m_effects.bpm > 0) {
        std::ostringstream zoom;
        // Subtle zoom pulse synced to BPM
        zoom << "zoompan=z='1.0+0.03*sin(t*PI*" << (m_effects.bpm / 30.0) << ")'"
             << ":d=1:s=" << m_outputWidth << "x" << m_outputHeight;
        filters.push_back(zoom.str());
    }

    // Join filters with commas
    if (filters.empty()) {
        return "";
    }

    std::string result = filters[0];
    for (size_t i = 1; i < filters.size(); ++i) {
        result += "," + filters[i];
    }
    return result;
}

bool VideoWriter::applyEffects(const std::string& inputVideo, const std::string& outputVideo) {
    std::string filterChain = buildEffectsFilterChain();

    // If no effects enabled, just copy
    if (filterChain.empty() && !m_effects.enableBeatFlash) {
        // Simple copy
        std::string ffmpegPath = getFFmpegPath();
        std::ostringstream cmd;
        cmd << "\"" << ffmpegPath << "\" -i \"" << inputVideo << "\""
            << " -c copy -y \"" << outputVideo << "\"";

        std::string fullCmd = cmd.str() + " 2>&1";
        FILE* pipe = popen_compat(fullCmd.c_str(), "r");
        if (!pipe) {
            m_lastError = "Failed to execute FFmpeg for effects copy";
            return false;
        }
        char buffer[256];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {}
        int exitCode = pclose_compat(pipe);
        return exitCode == 0;
    }

    // Apply effects with re-encoding
    std::string ffmpegPath = getFFmpegPath();
    std::ostringstream cmd;
    cmd << "\"" << ffmpegPath << "\" -i \"" << inputVideo << "\"";

    // Build video filter
    std::string vf;
    if (!filterChain.empty()) {
        vf = filterChain;
    }

    // Beat flash effect (white flash on beat intervals)
    if (m_effects.enableBeatFlash && m_effects.bpm > 0) {
        double beatInterval = 60.0 / m_effects.bpm;
        std::ostringstream flash;
        // Create brief white flash every beat
        flash << "split[main][flash];"
              << "[flash]drawbox=c=white@0.3:t=fill,fade=t=out:st=0:d=0.08[fl];"
              << "[main][fl]overlay=enable='lt(mod(t," << beatInterval << "),0.08)'";
        if (!vf.empty()) {
            vf = vf + "," + flash.str();
        } else {
            vf = flash.str();
        }
    }

    if (!vf.empty()) {
        cmd << " -vf \"" << vf << "\"";
    }

    cmd << " -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p"
        << " -c:a copy"
        << " -y \"" << outputVideo << "\"";

    std::cout << "Applying effects...\n";

    std::string fullCmd = cmd.str() + " 2>&1";
    FILE* pipe = popen_compat(fullCmd.c_str(), "r");
    if (!pipe) {
        m_lastError = "Failed to execute FFmpeg for effects";
        return false;
    }

    char buffer[256];
    std::string ffmpegOutput;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        ffmpegOutput += buffer;
    }

    int exitCode = pclose_compat(pipe);

    if (exitCode != 0) {
        m_lastError = "FFmpeg effects processing failed";
        // Check if output file was created anyway
        FILE* test = fopen(outputVideo.c_str(), "rb");
        if (test) {
            fclose(test);
            return true;
        }
        return false;
    }

    return true;
}

} // namespace BeatSync
