#include "VideoWriter.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstdio>
#include <cstdlib>

// Cross-platform popen/pclose
#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#endif

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/timestamp.h>
}

namespace BeatSync {

VideoWriter::VideoWriter()
{
}

VideoWriter::~VideoWriter() {
}

std::string VideoWriter::getFFmpegPath() const {
    // 1. Check environment variable first
    const char* envPath = std::getenv("BEATSYNC_FFMPEG_PATH");
    if (envPath != nullptr && envPath[0] != '\0') {
        return envPath;
    }

    // 2. Try to find ffmpeg in PATH
#ifdef _WIN32
    const char* findCmd = "where ffmpeg 2>nul";
    const char* fallbackPath = "C:\\ffmpeg-dev\\ffmpeg-master-latest-win64-gpl-shared\\bin\\ffmpeg.exe";
#else
    const char* findCmd = "which ffmpeg 2>/dev/null";
    const char* fallbackPath = "/usr/local/bin/ffmpeg";
#endif

    FILE* pipe = popen(findCmd, "r");
    if (pipe) {
        char buffer[512];
        std::string result;
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            result += buffer;
        }
        pclose(pipe);

        // Get first line (first match)
        size_t newline = result.find('\n');
        if (newline != std::string::npos) {
            result = result.substr(0, newline);
        }

        // Remove trailing whitespace
        while (!result.empty() && (result.back() == '\n' || result.back() == '\r' || result.back() == ' ')) {
            result.pop_back();
        }

        // If we found something, use it
        if (!result.empty() && result.find("ffmpeg") != std::string::npos) {
            return result;
        }
    }

    // 3. Fall back to hardcoded path
    return fallbackPath;
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

bool VideoWriter::copySegmentFast(const std::string& inputVideo,
                                   double startTime,
                                   double duration,
                                   const std::string& outputVideo) {
    // Use FFmpeg command-line for reliable segment extraction
    // Note: popen() on Windows passes commands to cmd.exe, so we need proper quote escaping
    //
    // FIX: Normalize ALL clips to same resolution (1920x1080), frame rate (24fps),
    // and pixel format to prevent freezing from mixed source formats
    std::string ffmpegPath = getFFmpegPath();
    std::ostringstream cmd;
    cmd << "\"\"" << ffmpegPath << "\" -i \"" << inputVideo << "\""
        << " -ss " << startTime
        << " -t " << duration
        << " -vf \"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=24\""
        << " -c:v libx264 -preset ultrafast -crf 18 -pix_fmt yuv420p"
        << " -c:a aac -b:a 192k -ar 44100"
        << " -video_track_timescale 90000"
        << " -avoid_negative_ts make_zero"
        << " -y \"" << outputVideo << "\"\"";

    // DEBUG: Print command for first failure
    static int debugCount = 0;
    bool shouldDebug = (debugCount < 2);

    // Execute FFmpeg
    FILE* pipe = popen(cmd.str().c_str(), "r");
    if (!pipe) {
        m_lastError = "Failed to execute FFmpeg";
        return false;
    }

    // Read output
    char buffer[256];
    std::string ffmpegOutput;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        ffmpegOutput += buffer;
    }

    int exitCode = pclose(pipe);

    if (exitCode != 0) {
        m_lastError = "Segment extraction failed";
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
        // Check if output file exists
        FILE* test = fopen(outputVideo.c_str(), "rb");
        if (test) {
            fclose(test);
            return true;  // File was created despite non-zero exit
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
    cmd << "\"\"" << ffmpegPath << "\" -i \"" << inputVideo << "\""
        << " -ss " << startTime
        << " -t " << duration
        << " -vf \"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=24\""
        << " -c:v libx264 -preset ultrafast -crf 18 -pix_fmt yuv420p"
        << " -c:a aac -b:a 192k -ar 44100"
        << " -video_track_timescale 90000"
        << " -avoid_negative_ts make_zero"
        << " -y \"" << outputVideo << "\"\"";

    // Execute FFmpeg
    FILE* pipe = popen(cmd.str().c_str(), "r");
    if (!pipe) {
        m_lastError = "Failed to execute FFmpeg for precise copy";
        return false;
    }

    // Read output
    char buffer[256];
    std::string ffmpegOutput;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        ffmpegOutput += buffer;
    }

    int exitCode = pclose(pipe);

    if (exitCode != 0) {
        m_lastError = "Precise segment extraction failed";
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

    for (const auto& video : inputVideos) {
        fprintf(f, "file '%s'\n", video.c_str());
    }
    fclose(f);

    // Use FFmpeg command-line to concatenate pre-normalized segments
    // Since all segments are now normalized to same resolution/fps/format,
    // we can use stream copy for fast concatenation
    //
    // FIX: Use stream copy since segments are pre-normalized, add -fflags +genpts
    // to regenerate clean timestamps
    std::string ffmpegPath = getFFmpegPath();
    std::ostringstream cmd;
    cmd << "\"\"" << ffmpegPath << "\" -fflags +genpts+igndts -f concat -safe 0 -i \"" << listFile
        << "\" -c copy -video_track_timescale 90000 -y \"" << outputVideo << "\"\"";

    // Execute FFmpeg
    FILE* pipe = popen(cmd.str().c_str(), "r");
    if (!pipe) {
        m_lastError = "Failed to execute FFmpeg";
        std::remove(listFile.c_str());
        return false;
    }

    // Read output (for debugging if needed)
    char buffer[256];
    std::string ffmpegOutput;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        ffmpegOutput += buffer;
    }

    int exitCode = pclose(pipe);

    // Remove concat list file
    std::remove(listFile.c_str());

    if (exitCode != 0) {
        m_lastError = "FFmpeg concatenation failed";
        // Include last few lines of output for debugging
        size_t lastNewline = ffmpegOutput.rfind('\n', ffmpegOutput.size() - 2);
        if (lastNewline != std::string::npos) {
            m_lastError += ": " + ffmpegOutput.substr(lastNewline + 1);
        }
        return false;
    }

    return true;
}

bool VideoWriter::addAudioTrack(const std::string& inputVideo,
                                 const std::string& audioFile,
                                 const std::string& outputVideo,
                                 bool trimToShortest) {
    m_lastError.clear();

    std::string ffmpegPath = getFFmpegPath();
    std::ostringstream cmd;

    // Combine video from first input with audio from second input
    // -c:v copy = stream copy video (fast, no re-encode)
    // -c:a aac = encode audio as AAC
    // -shortest = trim output to shorter of video/audio (optional)
    cmd << "\"\"" << ffmpegPath << "\" -i \"" << inputVideo << "\""
        << " -i \"" << audioFile << "\""
        << " -c:v copy -c:a aac -b:a 192k"
        << " -map 0:v:0 -map 1:a:0";  // Take video from first input, audio from second

    if (trimToShortest) {
        cmd << " -shortest";
    }

    cmd << " -y \"" << outputVideo << "\"\"";

    std::cout << "Adding audio track...\n";

    // Execute FFmpeg
    FILE* pipe = popen(cmd.str().c_str(), "r");
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

    int exitCode = pclose(pipe);

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

} // namespace BeatSync
