#include "VideoWriter.h"
#include "TransitionLibrary.h"
#include "OnnxFrameInterpolator.h"
#include "OnnxUpscaler.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <set>
#include <chrono>
#include <ctime>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <atomic>
#include <thread>
#include <mutex>
#include <atomic>
#include <random>
#include <algorithm>
#include <cmath>
#include <fstream>

// libavformat for audio stream probing
extern "C" {
#include <libavformat/avformat.h>
}

#include <algorithm>

// Cross-platform popen/pclose
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <vector>
#define popen_compat _popen
#define pclose_compat _pclose

// Run a command hidden (no console window) and capture output.
// Returns exit code; output is appended to 'output'.
// If cancelFlag is provided and becomes non-zero, the process is terminated.
static int runHiddenCommand(const std::string& cmdLine, std::string& output, const int* cancelFlag = nullptr) {
    if (cmdLine.empty()) {
        output = "Error: empty command line";
        return -1;
    }

    SECURITY_ATTRIBUTES sa;
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    sa.lpSecurityDescriptor = NULL;

    HANDLE hReadPipe = NULL, hWritePipe = NULL;
    if (!CreatePipe(&hReadPipe, &hWritePipe, &sa, 0)) {
        DWORD err = GetLastError();
        output = "Error: CreatePipe failed with error " + std::to_string(err);
        return -1;
    }

    if (!SetHandleInformation(hReadPipe, HANDLE_FLAG_INHERIT, 0)) {
        DWORD err = GetLastError();
        // Clean up handles to avoid leaks
        if (hReadPipe) { CloseHandle(hReadPipe); hReadPipe = NULL; }
        if (hWritePipe) { CloseHandle(hWritePipe); hWritePipe = NULL; }
        output = "Error: SetHandleInformation failed with error " + std::to_string(err);
        return -1;
    }

    STARTUPINFOA si = {0};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESHOWWINDOW | STARTF_USESTDHANDLES;
    si.wShowWindow = SW_HIDE;
    si.hStdOutput = hWritePipe;
    si.hStdError = hWritePipe;

    PROCESS_INFORMATION pi = {0};

    // CreateProcess works best when we pass NULL for lpApplicationName
    // and let it parse the command line itself. This handles quoted paths correctly.
    // We need a mutable copy of the command line.
    std::vector<char> cmdBuf(cmdLine.begin(), cmdLine.end());
    cmdBuf.push_back('\0');

    BOOL ok = CreateProcessA(
        NULL,             // Let CreateProcess parse the executable from command line
        cmdBuf.data(),    // Command line (must be mutable)
        NULL, NULL, TRUE,
        CREATE_NO_WINDOW,
        NULL, NULL,
        &si, &pi
    );

    // Close write end of pipe immediately after CreateProcess
    CloseHandle(hWritePipe);
    hWritePipe = NULL;

    if (!ok) {
        DWORD err = GetLastError();
        CloseHandle(hReadPipe);
        output = "Error: CreateProcess failed with error " + std::to_string(err) +
                 " for command: " + cmdLine.substr(0, 200);
        return -1;
    }

    // Read output in chunks with non-blocking check for cancellation
    char buf[4096];
    DWORD bytesRead;
    while (ReadFile(hReadPipe, buf, sizeof(buf) - 1, &bytesRead, NULL) && bytesRead > 0) {
        buf[bytesRead] = '\0';
        output += buf;

        // Check for cancellation during output reading
        if (cancelFlag && *cancelFlag != 0) {
            TerminateProcess(pi.hProcess, 2);
            CloseHandle(hReadPipe);
            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
            output += "\nCancelled by user";
            return -2;  // Special code for cancellation
        }
    }
    CloseHandle(hReadPipe);

    // Wait for process to complete with periodic cancellation checks
    // Instead of a single 5-minute wait, poll every 100ms
    DWORD exitCode = 0;
    constexpr DWORD pollIntervalMs = 100;
    constexpr DWORD maxWaitMs = 300000;  // 5 minutes
    DWORD totalWaitMs = 0;

    while (totalWaitMs < maxWaitMs) {
        // Check for cancellation
        if (cancelFlag && *cancelFlag != 0) {
            TerminateProcess(pi.hProcess, 2);
            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
            output += "\nCancelled by user";
            return -2;
        }

        DWORD waitResult = WaitForSingleObject(pi.hProcess, pollIntervalMs);
        if (waitResult == WAIT_OBJECT_0) {
            // Process finished
            GetExitCodeProcess(pi.hProcess, &exitCode);
            break;
        } else if (waitResult == WAIT_TIMEOUT) {
            totalWaitMs += pollIntervalMs;
            continue;
        } else {
            // Unexpected error
            exitCode = 1;
            output += "\nError: WaitForSingleObject failed";
            break;
        }
    }

    if (totalWaitMs >= maxWaitMs) {
        TerminateProcess(pi.hProcess, 1);
        exitCode = 1;
        output += "\nError: Process timed out after 5 minutes";
    }

    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    return static_cast<int>(exitCode);
}
#else
#define popen_compat popen
#define pclose_compat pclose
#endif

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/timestamp.h>
}

namespace BeatSync {

namespace {
// Helper to get temp directory path with trailing slash
// Uses Windows-native API to avoid std::filesystem exceptions that can crash the app
std::string getTempDir() {
#ifdef _WIN32
    wchar_t tempPath[MAX_PATH + 1];
    DWORD len = GetTempPathW(MAX_PATH + 1, tempPath);
    if (len > 0 && len < MAX_PATH) {
        char narrowPath[MAX_PATH + 1];
        int result = WideCharToMultiByte(CP_UTF8, 0, tempPath, -1, narrowPath, MAX_PATH + 1, nullptr, nullptr);
        if (result > 0) {
            std::string tempDir(narrowPath);
            if (!tempDir.empty() && tempDir.back() != '\\') {
                tempDir += '\\';
            }
            return tempDir;
        }
    }
    // Fallback: try TEMP/TMP environment variables before using hardcoded path
    const char* tempEnv = std::getenv("TEMP");
    if (!tempEnv) tempEnv = std::getenv("TMP");
    if (tempEnv && tempEnv[0] != '\0') {
        std::string tempDir(tempEnv);
        if (!tempDir.empty() && tempDir.back() != '\\') {
            tempDir += '\\';
        }
        return tempDir;
    }
    // Last resort fallback
    return "C:\\Temp\\";
#else
    try {
        std::string tempDir = std::filesystem::temp_directory_path().string();
        if (!tempDir.empty() && tempDir.back() != '/') {
            tempDir += '/';
        }
        return tempDir;
    } catch (...) {
        return "/tmp/";
    }
#endif
}

// Force release of any lingering FFmpeg CUDA contexts
// This is necessary because repeated FFmpeg invocations can leak CUDA memory
// The function runs nvidia-smi to query memory, which forces the driver to clean up
// orphaned contexts from terminated FFmpeg processes
void flushGpuMemory() {
#ifdef _WIN32
    // Sleep briefly to allow async CUDA cleanup from recent FFmpeg processes
    Sleep(100);

    // Running nvidia-smi query forces NVIDIA driver to clean up orphaned CUDA contexts
    // This is a best-effort cleanup that helps prevent memory accumulation
    STARTUPINFOW si = { sizeof(si) };
    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    PROCESS_INFORMATION pi = {0};

    // nvidia-smi -q -d MEMORY queries memory state, which triggers driver cleanup
    wchar_t cmdLine[] = L"nvidia-smi -q -d MEMORY";
    if (CreateProcessW(nullptr, cmdLine, nullptr, nullptr, FALSE,
                       CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) {
        // Wait up to 2 seconds for query to complete
        WaitForSingleObject(pi.hProcess, 2000);
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
    }

    // Additional brief sleep to let cleanup propagate
    Sleep(50);
#else
    // On Linux/macOS, brief sleep using standard thread features
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
#endif
}

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
    // Cache the result to avoid repeated system calls
    static std::string s_cachedFfmpegPath;
    static std::once_flag s_ffmpegFlag;

    std::call_once(s_ffmpegFlag, []() {
        // 1. Check environment variable first
        const char* envPath = std::getenv("BEATSYNC_FFMPEG_PATH");
        if (envPath != nullptr && envPath[0] != '\0') {
            s_cachedFfmpegPath = envPath;
            return;
        }

        // 2. Try to find ffmpeg in PATH (hidden to avoid console flash)
#ifdef _WIN32
        std::string result;
        int rc = runHiddenCommand("where ffmpeg", result);
        if (rc == 0 && !result.empty()) {
            // Get first line (first match)
            size_t newline = result.find('\n');
            if (newline != std::string::npos) {
                result = result.substr(0, newline);
            }
            // Trim trailing whitespace
            while (!result.empty() && (result.back() == '\r' || result.back() == ' ')) {
                result.pop_back();
            }
            // If we found something, cache and use it
            if (!result.empty() && result.find("ffmpeg") != std::string::npos) {
                s_cachedFfmpegPath = result;
                return;
            }
        }
#else
        FILE* pipe = popen("which ffmpeg 2>/dev/null", "r");
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

            // If we found something, cache and use it
            if (!result.empty() && result.find("ffmpeg") != std::string::npos) {
                s_cachedFfmpegPath = result;
                return;
            }
        }
#endif

        // 3. Fall back to platform-specific hardcoded path
#ifdef _WIN32
        s_cachedFfmpegPath = "C:\\ffmpeg-dev\\ffmpeg-master-latest-win64-gpl-shared\\bin\\ffmpeg.exe";
#elif defined(__APPLE__)
        s_cachedFfmpegPath = "/opt/homebrew/bin/ffmpeg";
#else
        s_cachedFfmpegPath = "/usr/bin/ffmpeg";
#endif
    });

    return s_cachedFfmpegPath;
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

    // Per-clip speed ramps: assign a deterministic multiplier to each segment.
    // The source-footage guard is applied per-clip inside extractSegments.
    if (m_speed.enabled && !segments.empty()) {
        std::vector<double> speeds = computeClipSpeeds(segments.size(), m_speed);
        for (size_t i = 0; i < segments.size(); ++i) {
            segments[i].speed = speeds[i];
        }
    }

    return extractSegments(inputVideo, segments, outputVideo);
}

bool VideoWriter::extractSegments(const std::string& inputVideo,
                                  const std::vector<VideoSegment>& segments,
                                  const std::string& outputVideo) {
    m_lastError.clear();

    // Reset GPU segment counter at start of batch operation
    resetSegmentCounter();

    // Log GPU capabilities once on first call
    static bool loggedGpuCaps = false;
    if (!loggedGpuCaps) {
        logGpuCapabilities();
        loggedGpuCaps = true;
    }

    if (segments.empty()) {
        m_lastError = "No segments to extract";
        return false;
    }

    // Per-clip speed ramp setup: if any segment requests a speed != 1.0, probe
    // the source length once so the source-footage guard can clamp speed-ups
    // that would overrun the end of the video.
    m_speedClampCount.store(0);
    bool anySpeed = false;
    for (const auto& s : segments) {
        if (std::abs(s.speed - 1.0) > 1e-6) { anySpeed = true; break; }
    }
    double sourceLen = 0.0;
    if (anySpeed) {
        VideoProcessor proc;
        if (proc.open(inputVideo)) {
            sourceLen = proc.getInfo().duration;
            proc.close();
        }
    }

    // Use OS temp directory for intermediate segment files to avoid write-permission
    // issues when the executable runs from a protected folder or read-only install.
    std::string tempDir = getTempDir();
    if (tempDir.empty()) {
        m_lastError = "Could not resolve temporary directory";
        return false;
    }
    std::filesystem::create_directories(tempDir);

    // Pre-allocate temp file names (must be in order for concatenation)
    std::vector<std::string> tempFiles(segments.size());
    for (size_t i = 0; i < segments.size(); ++i) {
        std::ostringstream tempFile;
        tempFile << tempDir << "beatsync_segment_" << std::setw(5) << std::setfill('0') << i << ".mp4";
        tempFiles[i] = tempFile.str();
    }

    std::cout << "Extracting " << segments.size() << " segments to " << tempDir << "...\n";

    // Parallel segment extraction with thread pool
    // Use up to 4 concurrent FFmpeg processes (balances I/O and GPU encoder utilization)
    const size_t maxConcurrent = std::min(static_cast<size_t>(4),
                                          static_cast<size_t>(std::thread::hardware_concurrency()));
    const size_t workerCount = maxConcurrent == 0 ? 1 : maxConcurrent;

    std::atomic<size_t> nextSegment{0};
    std::atomic<size_t> completedSegments{0};
    std::atomic<bool> hasError{false};
    std::mutex errorMutex;
    std::string firstError;

    std::atomic<bool> cancelled{false};

    // Worker function - each thread processes segments until done
    auto processSegments = [&]() {
        while (!hasError && !cancelled) {
            // Check for cancellation
            if (isCancelled()) {
                cancelled = true;
                break;
            }

            // Atomically claim the next segment
            size_t i = nextSegment.fetch_add(1);
            if (i >= segments.size()) break;

            const auto& seg = segments[i];
            double duration = seg.endTime - seg.startTime;
            const std::string& outFile = tempFiles[i];

            if (m_progressCallback) {
                reportProgress(i / (double)segments.size() * 0.9);
            }

            // Per-clip speed ramp: a clip with speed != 1.0 must re-encode
            // (stream copy cannot retime), so skip the fast path. The output
            // clip stays `duration` long; source consumed = duration * speed.
            double clipSpeed = seg.speed;
            if (std::abs(clipSpeed - 1.0) > 1e-6) {
                // Source-footage guard (speed-up only; slow-mo consumes less).
                if (clipSpeed > 1.0 && sourceLen > 0.0) {
                    double needed = duration * clipSpeed;
                    double available = sourceLen - seg.startTime;
                    if (needed > available) {
                        m_speedClampCount.fetch_add(1);
                        double maxSpeed = (duration > 0.0) ? (available / duration) : 1.0;
                        clipSpeed = (m_speed.guardClampToAvailable && maxSpeed > 1.001)
                                        ? std::min(clipSpeed, maxSpeed)
                                        : 1.0;
                    }
                }
            }

            bool success;
            if (std::abs(clipSpeed - 1.0) > 1e-6) {
                success = extractSpeedClip(inputVideo, seg.startTime, duration, clipSpeed, outFile);
            } else {
                // Try fast copy, then precise copy as fallback
                success = copySegmentFast(inputVideo, seg.startTime, duration, outFile);
                if (!success) {
                    success = copySegmentPrecise(inputVideo, seg.startTime, duration, outFile);
                }
            }

            if (!success) {
                std::lock_guard<std::mutex> lock(errorMutex);
                if (!hasError) {
                    hasError = true;
                    firstError = "Failed to extract segment " + std::to_string(i) +
                                 " (" + std::to_string(seg.startTime) + "s - " +
                                 std::to_string(seg.endTime) + "s)";
                }
                break;
            }

            // Update progress
            size_t completed = completedSegments.fetch_add(1) + 1;
            std::cout << "  Segment " << completed << "/" << segments.size()
                      << ": " << seg.startTime << "s - " << seg.endTime << "s [done]\n";

            if (m_progressCallback) {
                reportProgress(completed / (double)segments.size() * 0.9);
            }
        }
    };

    // Launch worker threads
    std::vector<std::thread> workers;
    workers.reserve(workerCount);
    for (size_t t = 0; t < workerCount; ++t) {
        workers.emplace_back(processSegments);
    }

    // Wait for all workers to complete
    for (auto& worker : workers) {
        worker.join();
    }

    // Check for cancellation or errors
    if (cancelled) {
        // Cleanup any temp files that were created
        for (const auto& f : tempFiles) {
            std::remove(f.c_str());
        }
        m_lastError = "Operation cancelled by user";
        return false;
    }

    if (hasError) {
        // Cleanup any temp files that were created
        for (const auto& f : tempFiles) {
            std::remove(f.c_str());
        }
        m_lastError = firstError;
        return false;
    }

    // Concatenate all segments (must be sequential - order matters)
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

void VideoWriter::setCancelFlag(const int* flag) {
    m_cancelFlag = flag;
}

bool VideoWriter::isCancelled() const {
    return m_cancelFlag != nullptr && *m_cancelFlag != 0;
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
    // DEBUG: Version marker to confirm correct DLL is loaded (v2025.01.19)
    static bool versionLogged = false;
    if (!versionLogged) {
        std::cout << "[VideoWriter] DLL version: 2025.01.19-fix-scientific-notation\n";
        versionLogged = true;
    }

    // Clamp very small start times to zero - values like 2e-05 (0.00002s) are essentially zero
    // and can cause FFmpeg errors with scientific notation even with std::fixed in some cases
    if (startTime < 0.001) {
        std::cout << "[VideoWriter] Clamping startTime from " << std::scientific << startTime << " to 0.0\n" << std::fixed;
        startTime = 0.0;
    }

    std::cout << "Extracting segment: " << inputVideo << " @ " << std::fixed << std::setprecision(6) << startTime << "s for " << duration << "s -> " << outputVideo << std::defaultfloat << "\n";

    // Use FFmpeg command-line for reliable segment extraction
    // Note: popen_compat() on Windows passes commands to cmd.exe, so we need proper quote escaping
    //
    // FIX: Normalize ALL clips to same resolution (1920x1080), frame rate (24fps),
    // and pixel format to prevent freezing from mixed source formats
    std::string ffmpegPath = getFFmpegPath();
    std::ostringstream cmd;
    cmd << "\"" << ffmpegPath << "\" -nostdin";

    // GPU acceleration: Use CUDA hardware decoding if available
    // IMPORTANT: Periodically force CPU mode to release GPU memory and prevent CUDA crashes
    // Store in member variable so copySegmentPrecise fallback can use the same decision
    m_allowGpuThisSegment = shouldUseGpuForSegment();
    bool allowGpuThisSegment = m_allowGpuThisSegment;
    bool cudaAvailable = hasCudaHwaccel();
    bool useCuda = allowGpuThisSegment && cudaAvailable;
    bool scaleCudaAvailable = hasScaleCudaFilter();
    bool useScaleCuda = useCuda && scaleCudaAvailable;

    // DEBUG: Log GPU decision for every segment
    static std::atomic<int> segmentNum{0};
    int currentSegment = ++segmentNum;
    std::cout << "[GPU DEBUG] Segment " << currentSegment << ": allowGpu=" << allowGpuThisSegment
              << " cudaHwaccel=" << cudaAvailable << " scaleCuda=" << scaleCudaAvailable
              << " -> useCuda=" << useCuda << " useScaleCuda=" << useScaleCuda << "\n";

    if (useCuda) {
        cmd << " -hwaccel cuda -hwaccel_device 0";
        // Keep frames on GPU if using NVENC encoder and scale_cuda
        if (useScaleCuda && probeEncoder("h264_nvenc")) {
            cmd << " -hwaccel_output_format cuda";
        }
    }

    // Use fixed-point notation for time values - FFmpeg doesn't accept scientific notation (e.g., 2e-05)
    cmd << std::fixed << std::setprecision(6);
    cmd << " -i \"" << inputVideo << "\""
        << " -ss " << startTime
        << " -t " << duration;
    cmd << std::defaultfloat;  // Reset to default formatting

    // Build filter chain: Use GPU filters when available, CPU fallback otherwise
    if (useScaleCuda) {
        // GPU-accelerated filter chain with scale_cuda
        // Note: pad filter has no CUDA equivalent, so we need hwdownload->pad->hwupload
        // Add setsar=1 for SAR consistency with CPU path
        cmd << " -vf \"scale_cuda=" << m_outputWidth << ":" << m_outputHeight
            << ":force_original_aspect_ratio=decrease,hwdownload,format=nv12"
            << ",pad=" << m_outputWidth << ":" << m_outputHeight
            << ":(ow-iw)/2:(oh-ih)/2,setsar=1,fps=" << m_outputFps << "\"";
    } else {
        // CPU filter chain (original behavior)
        cmd << " -vf \"scale=" << m_outputWidth << ":" << m_outputHeight
            << ":force_original_aspect_ratio=decrease,pad=" << m_outputWidth << ":" << m_outputHeight
            << ":(ow-iw)/2:(oh-ih)/2,setsar=1,fps=" << m_outputFps << "\"";
    }

    // Always use best available encoder (GPU preferred)
    std::string encoderArgs = getEncoderArgs("ultrafast");
    std::cout << "[GPU DEBUG] Using encoder: " << encoderArgs << "\n";
    cmd << " " << encoderArgs;
    cmd << " -c:a aac -b:a 192k -ar 44100"
        << " -video_track_timescale 90000"
        << " -avoid_negative_ts make_zero"
        << " -y \"" << outputVideo << "\"";

    // DEBUG: Print command for first failure
    static int debugCount = 0;
    bool shouldDebug = (debugCount < 2);

    // Execute FFmpeg hidden (no console flash on Windows)
    std::string ffmpegOutput;
    int exitCode;
#ifdef _WIN32
    exitCode = runHiddenCommand(cmd.str(), ffmpegOutput, m_cancelFlag);
    if (exitCode == -2) {
        m_lastError = "Cancelled by user";
        return false;  // Cancelled
    }
#else
    std::string fullCmd = cmd.str() + " 2>&1";
    FILE* pipe = popen_compat(fullCmd.c_str(), "r");
    if (!pipe) {
        m_lastError = "Failed to execute FFmpeg";
        appendFfmpegLog("beatsync_ffmpeg_extract.log", "copySegmentFast::popen_compat_failed", fullCmd, -1, "", "start=" + std::to_string(startTime) + ", dur=" + std::to_string(duration));
        return false;
    }
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        ffmpegOutput += buffer;
    }
    exitCode = pclose_compat(pipe);
#endif

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
                                     const std::string& outputVideo,
                                     double speed,
                                     int smoothing) {
    // Clamp very small start times to zero - values like 2e-05 (0.00002s) are essentially zero
    // and can cause FFmpeg errors with scientific notation even with std::fixed in some cases
    if (startTime < 0.001) {
        startTime = 0.0;
    }

    // Use FFmpeg with re-encoding for frame-accurate extraction
    // This is slower but more precise than stream copy
    //
    // FIX: Normalize ALL clips to same resolution, frame rate, and pixel format
    std::string ffmpegPath = getFFmpegPath();
    std::ostringstream cmd;
    cmd << "\"" << ffmpegPath << "\" -nostdin";

    // GPU acceleration: Use CUDA hardware decoding if available
    // IMPORTANT: Periodically force CPU mode to release GPU memory and prevent CUDA crashes
    // Note: copySegmentPrecise is only called as fallback from copySegmentFast, which already
    // set m_allowGpuThisSegment via shouldUseGpuForSegment(). Use that explicit flag.
    bool allowGpuThisSegment = m_allowGpuThisSegment;
    bool useCuda = allowGpuThisSegment && hasCudaHwaccel();
    bool useScaleCuda = useCuda && hasScaleCudaFilter();

    if (useCuda) {
        cmd << " -hwaccel cuda -hwaccel_device 0";
        // Keep frames on GPU if using NVENC encoder and scale_cuda
        if (useScaleCuda && probeEncoder("h264_nvenc")) {
            cmd << " -hwaccel_output_format cuda";
        }
    }

    // Per-clip speed ramp: setpts retimes the video so that `duration` seconds
    // of OUTPUT are produced while FFmpeg consumes `duration * speed` of source.
    // The output `-t duration` (below) caps the slot, so the source amount is
    // governed automatically. speed is clamped so a single atempo keeps audio
    // length consistent for concatenation. See SpeedRampConfig.
    bool applySpeed = (speed > 0.0 && std::abs(speed - 1.0) > 1e-6);
    double clampedSpeed = speed;
    if (applySpeed) {
        clampedSpeed = std::max(static_cast<double>(SpeedRampConfig::kMinSpeed),
                                std::min(static_cast<double>(SpeedRampConfig::kMaxSpeed), clampedSpeed));
    }
    const double ptsFactor = applySpeed ? (1.0 / clampedSpeed) : 1.0;

    // Build the video-retime + framerate suffix appended to the filter chain.
    // - dup-frame smoothing: setpts then plain fps (may judder on slow-mo).
    // - minterpolate smoothing: motion-compensated frame interpolation to fps.
    std::ostringstream retimeFps;
    retimeFps << std::fixed << std::setprecision(6);
    if (applySpeed) {
        retimeFps << ",setpts=" << ptsFactor << "*(PTS-STARTPTS)";
    }
    if (applySpeed && smoothing == 1) {
        retimeFps << ",minterpolate=fps=" << m_outputFps << ":mi_mode=mci:me_mode=bidir:vsbmc=1";
    } else {
        retimeFps << ",fps=" << m_outputFps;
    }
    const std::string retimeFpsStr = retimeFps.str();

    // Use fixed-point notation for time values - FFmpeg doesn't accept scientific notation (e.g., 2e-05)
    cmd << std::fixed << std::setprecision(6);
    cmd << " -i \"" << inputVideo << "\""
        << " -ss " << startTime
        << " -t " << duration;
    cmd << std::defaultfloat;  // Reset to default formatting

    // Build filter chain: Use GPU filters when available, CPU fallback otherwise
    if (useScaleCuda) {
        // GPU-accelerated filter chain with scale_cuda
        // Add setsar=1 for SAR consistency with CPU path
        cmd << " -vf \"scale_cuda=" << m_outputWidth << ":" << m_outputHeight
            << ":force_original_aspect_ratio=decrease,hwdownload,format=nv12"
            << ",pad=" << m_outputWidth << ":" << m_outputHeight
            << ":(ow-iw)/2:(oh-ih)/2,setsar=1" << retimeFpsStr << "\"";
    } else {
        // CPU filter chain (original behavior)
        cmd << " -vf \"scale=" << m_outputWidth << ":" << m_outputHeight
            << ":force_original_aspect_ratio=decrease,pad=" << m_outputWidth << ":" << m_outputHeight
            << ":(ow-iw)/2:(oh-ih)/2,setsar=1" << retimeFpsStr << "\"";
    }

    // Retime audio to match the new video length so this segment file stays
    // internally A/V-consistent for concatenation. atempo is valid for the
    // clamped [0.5, 2.0] speed range. (The final master audio is muxed later.)
    if (applySpeed) {
        cmd << " -af \"atempo=" << std::fixed << std::setprecision(6) << clampedSpeed << "\"";
        cmd << std::defaultfloat;
    }

    // Always use best available encoder (GPU preferred)
    cmd << " " << getEncoderArgs("ultrafast");
    cmd << " -c:a aac -b:a 192k -ar 44100"
        << " -video_track_timescale 90000"
        << " -avoid_negative_ts make_zero"
        << " -y \"" << outputVideo << "\"";

    // Execute FFmpeg hidden (no console flash on Windows)
    std::string ffmpegOutput;
    int exitCode;
#ifdef _WIN32
    exitCode = runHiddenCommand(cmd.str(), ffmpegOutput, m_cancelFlag);
    if (exitCode == -2) {
        m_lastError = "Cancelled by user";
        return false;  // Cancelled
    }
    if (exitCode == -1) {
        m_lastError = "Failed to execute FFmpeg for precise copy";
        appendFfmpegLog("beatsync_ffmpeg_extract.log", "copySegmentPrecise::runHiddenCommand_failed", cmd.str(), -1, "", "start=" + std::to_string(startTime) + ", dur=" + std::to_string(duration));
        return false;
    }
#else
    std::string fullCmd = cmd.str() + " 2>&1";
    FILE* pipe = popen_compat(fullCmd.c_str(), "r");
    if (!pipe) {
        m_lastError = "Failed to execute FFmpeg for precise copy";
        appendFfmpegLog("beatsync_ffmpeg_extract.log", "copySegmentPrecise::popen_compat_failed", fullCmd, -1, "", "start=" + std::to_string(startTime) + ", dur=" + std::to_string(duration));
        return false;
    }
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        ffmpegOutput += buffer;
    }
    exitCode = pclose_compat(pipe);
#endif

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

bool VideoWriter::normalizeVideo(const std::string& inputVideo, const std::string& outputVideo) {
    std::cout << "Pre-normalizing video: " << inputVideo << " -> " << outputVideo << "\n";

    // RAII wrapper for FILE* to ensure proper cleanup on all return paths
    struct FileGuard {
        FILE* file = nullptr;
        ~FileGuard() { if (file) fclose(file); }
    };

    // File-based diagnostic logging with RAII cleanup
    FileGuard diagLogGuard;
    diagLogGuard.file = fopen((getTempDir() + "beatsync_normalize_detail.log").c_str(), "a");
    FILE* diagLog = diagLogGuard.file;  // Alias for convenience
    if (diagLog) {
        fprintf(diagLog, "\n=== normalizeVideo ENTER ===\n");
        fprintf(diagLog, "  input: %s\n", inputVideo.c_str());
        fprintf(diagLog, "  output: %s\n", outputVideo.c_str());
        fflush(diagLog);
    }

    std::string ffmpegPath = getFFmpegPath();
    if (diagLog) {
        fprintf(diagLog, "  ffmpegPath: %s\n", ffmpegPath.c_str());
        fflush(diagLog);
    }

    std::ostringstream cmd;
    cmd << "\"" << ffmpegPath << "\" -nostdin";
    bool cudaAvail = hasCudaHwaccel();
    bool scaleCudaAvail = hasScaleCudaFilter();
    bool nvencAvail = probeEncoder("h264_nvenc");
    bool useGpuNormalize = cudaAvail && scaleCudaAvail && nvencAvail;

    if (diagLog) {
        fprintf(diagLog, "  GPU normalize: cuda=%d scaleCuda=%d nvenc=%d -> useGpu=%d\n",
                cudaAvail, scaleCudaAvail, nvencAvail, useGpuNormalize);
        fflush(diagLog);
    }

    if (useGpuNormalize) {
        // GPU-accelerated pipeline: CUDA decode -> scale_cuda -> NVENC encode
        // Keep frames on GPU throughout for maximum performance
        cmd << " -hwaccel cuda -hwaccel_device 0 -hwaccel_output_format cuda";
        cmd << " -i \"" << inputVideo << "\"";

        // GPU filter chain using scale_cuda, then hwdownload for pad (no pad_cuda with aspect handling)
        // Use scale_cuda for the heavy lifting, then CPU pad for letterboxing
        cmd << " -vf \"scale_cuda=" << m_outputWidth << ":" << m_outputHeight
            << ":force_original_aspect_ratio=decrease,hwdownload,format=nv12"
            << ",pad=" << m_outputWidth << ":" << m_outputHeight
            << ":(ow-iw)/2:(oh-ih)/2,setsar=1,fps=" << m_outputFps << "\"";

        // NVENC encoding for GPU-accelerated output
        cmd << " -c:v h264_nvenc -preset p4 -rc vbr -cq 18 -pix_fmt yuv420p"
            << " -c:a aac -b:a 192k -ar 44100"
            << " -video_track_timescale 90000"
            << " -y \"" << outputVideo << "\"";
    } else {
        // CPU fallback for systems without CUDA/NVENC
        cmd << " -i \"" << inputVideo << "\"";

        // CPU filter chain
        cmd << " -vf \"scale=" << m_outputWidth << ":" << m_outputHeight
            << ":force_original_aspect_ratio=decrease,pad=" << m_outputWidth << ":" << m_outputHeight
            << ":(ow-iw)/2:(oh-ih)/2,setsar=1,fps=" << m_outputFps << "\"";

        // libx264 CPU encoding
        cmd << " -c:v libx264 -preset fast -crf 18"
            << " -c:a aac -b:a 192k -ar 44100"
            << " -video_track_timescale 90000"
            << " -y \"" << outputVideo << "\"";
    }

    std::string ffmpegOutput;
    int exitCode;

    if (diagLog) {
        fprintf(diagLog, "  Executing FFmpeg command...\n");
        fprintf(diagLog, "  cmd: %s\n", cmd.str().substr(0, 500).c_str());
        fflush(diagLog);
    }

#ifdef _WIN32
    exitCode = runHiddenCommand(cmd.str(), ffmpegOutput, m_cancelFlag);
    if (exitCode == -2) {
        m_lastError = "Cancelled by user";
        if (diagLog) { fprintf(diagLog, "  CANCELLED by user\n"); fclose(diagLog); }
        return false;
    }
#else
    std::string fullCmd = cmd.str() + " 2>&1";

    FILE* pipe = popen_compat(fullCmd.c_str(), "r");
    if (!pipe) {
        m_lastError = "Failed to execute FFmpeg for video normalization";
        if (diagLog) { fprintf(diagLog, "  ERROR: popen failed\n"); }
        return false;
    }
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        ffmpegOutput += buffer;
    }
    exitCode = pclose_compat(pipe);
#endif

    if (diagLog) {
        fprintf(diagLog, "  exitCode: %d\n", exitCode);
        fprintf(diagLog, "  ffmpegOutput (first 300): %s\n", ffmpegOutput.substr(0, 300).c_str());
        fflush(diagLog);
    }

    // Log the normalization
    appendFfmpegLog("beatsync_ffmpeg_normalize.log", "normalizeVideo", cmd.str(), exitCode, ffmpegOutput, "");

    if (exitCode != 0) {
        // Check if file was created anyway
        FILE* test = fopen(outputVideo.c_str(), "rb");
        if (test) {
            fseek(test, 0, SEEK_END);
            long fileSize = ftell(test);
            fclose(test);
            if (fileSize > 1024) {
                std::cout << "  Normalization completed (non-zero exit but file OK: " << fileSize << " bytes)\n";
                if (diagLog) { fprintf(diagLog, "  File created despite error: %ld bytes\n", fileSize); }
                return true;
            }
        }
        m_lastError = "Video normalization failed: " + ffmpegOutput.substr(0, 200);
        if (diagLog) { fprintf(diagLog, "  FAILED: %s\n", m_lastError.c_str()); }
        return false;
    }

    // Verify file was created
    FILE* verify = fopen(outputVideo.c_str(), "rb");
    if (verify) {
        fseek(verify, 0, SEEK_END);
        long fileSize = ftell(verify);
        fclose(verify);
        if (diagLog) { fprintf(diagLog, "  SUCCESS: Output file %ld bytes\n", fileSize); }
    } else {
        if (diagLog) { fprintf(diagLog, "  WARNING: Output file not found after success exit code!\n"); }
    }

    std::cout << "  Normalization complete\n";
    return true;
}

bool VideoWriter::normalizeVideos(const std::vector<std::string>& inputVideos,
                                   std::vector<std::string>& normalizedPaths) {
    normalizedPaths.clear();

    std::cout << "[BeatSync] normalizeVideos called with " << inputVideos.size() << " videos\n";

    if (inputVideos.empty()) {
        return true;  // Nothing to normalize
    }

    std::string tempDir;
    try {
        tempDir = getTempDir();
        std::cout << "[BeatSync] Using temp dir: " << tempDir << "\n";
    } catch (const std::exception& e) {
        m_lastError = std::string("Failed to get temp directory: ") + e.what();
        std::cerr << "[BeatSync] " << m_lastError << "\n";
        return false;
    }

    int index = 0;
    int skippedCount = 0;

    for (const auto& video : inputVideos) {
        std::cout << "[BeatSync] Processing video " << index << ": " << video << "\n";

        // Generate a unique normalized filename in temp directory
        // Use simple string manipulation instead of std::filesystem to avoid potential issues
        std::string baseName;
        try {
            size_t lastSlash = video.find_last_of("/\\");
            size_t lastDot = video.find_last_of('.');
            if (lastSlash != std::string::npos) {
                if (lastDot != std::string::npos && lastDot > lastSlash) {
                    baseName = video.substr(lastSlash + 1, lastDot - lastSlash - 1);
                } else {
                    baseName = video.substr(lastSlash + 1);
                }
            } else {
                if (lastDot != std::string::npos) {
                    baseName = video.substr(0, lastDot);
                } else {
                    baseName = video;
                }
            }
        } catch (...) {
            baseName = "video";
        }

        std::string normalizedName = "beatsync_normalized_" + std::to_string(index++) + "_" +
                                     baseName + ".mp4";
        std::string normalizedPath = tempDir + normalizedName;
        std::cout << "[BeatSync] Output path: " << normalizedPath << "\n";

        // Optional neural upscale of the source before normalization. Doing it
        // here means each clip is enlarged once, however many times it is later
        // cycled into the edit, and the normalize pass below then scales the
        // result to the output resolution. Any failure (model missing, source
        // already large enough) is non-fatal: we normalize the original.
        std::string normalizeSource = video;
        std::string upscaledPath;
        if (m_upscale.enabled) {
            upscaledPath = tempDir + "beatsync_upscaled_" + std::to_string(index - 1) + "_" +
                           baseName + ".mp4";
            if (upscaleVideo(video, upscaledPath)) {
                normalizeSource = upscaledPath;
            } else {
                std::cout << "[BeatSync] Upscale skipped for " << video << ": " << m_lastError << "\n";
                std::remove(upscaledPath.c_str());
                upscaledPath.clear();
            }
        }

        struct UpscaleTempCleanup {
            const std::string& path;
            ~UpscaleTempCleanup() { if (!path.empty()) std::remove(path.c_str()); }
        } upscaleCleanup{upscaledPath};

        if (!normalizeVideo(normalizeSource, normalizedPath)) {
            // A single unreadable/corrupt source (e.g. truncated MP4 with a
            // missing moov atom, or a malformed VLC partial recording) must not
            // abort the entire export. Skip it and keep going with the rest.
            ++skippedCount;
            std::cerr << "[BeatSync] WARNING: skipping video that failed to normalize: "
                      << video << " (" << m_lastError << ")\n";
            // Best-effort: remove any partial output left behind for this clip.
            std::remove(normalizedPath.c_str());

            // Report progress for the skipped slot so the bar keeps advancing.
            if (m_progressCallback) {
                reportProgress(static_cast<double>(index) / inputVideos.size() * 0.1);
            }
            continue;
        }

        normalizedPaths.push_back(normalizedPath);

        // Report progress
        if (m_progressCallback) {
            reportProgress(static_cast<double>(index) / inputVideos.size() * 0.1);  // 10% for normalization
        }
    }

    if (skippedCount > 0) {
        std::cout << "[BeatSync] normalizeVideos skipped " << skippedCount
                  << " unreadable source(s); " << normalizedPaths.size() << " normalized OK\n";
    }

    // Only a hard failure if EVERY input was unusable.
    if (normalizedPaths.empty()) {
        m_lastError = "All " + std::to_string(inputVideos.size()) +
                      " source videos failed to normalize (corrupt or unreadable inputs)";
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

    // Create concat list file in temp directory
    std::string listFile = getTempDir() + "beatsync_concat_list.txt";
    FILE* f = fopen(listFile.c_str(), "w");
    if (!f) {
        m_lastError = "Could not create concat list file";
        return false;
    }

    // Log what we're concatenating for debugging
    std::cout << "Creating concat list with " << inputVideos.size() << " videos:\n";

    // Open debug log once for all logging in this section (avoids repeated file opens)
    FILE* debugLog = fopen((getTempDir() + "tripsitter_debug.log").c_str(), "a");
    if (debugLog) {
        fprintf(debugLog, "\n=== Concatenation Step ===\n");
        fprintf(debugLog, "Creating concat list with %zu videos:\n", inputVideos.size());
    }

    int missingCount = 0;
    for (const auto& video : inputVideos) {
        // FFmpeg resolves relative entries against the list file's directory (the
        // temp dir), not the process cwd - write absolute paths so callers can pass either.
        std::error_code absEc;
        std::filesystem::path absVideo = std::filesystem::absolute(video, absEc);
        const std::string videoEntry = absEc ? video : absVideo.string();
        fprintf(f, "file '%s'\n", videoEntry.c_str());
        std::cout << "  - " << video;

        // Check if file exists
        FILE* check = fopen(video.c_str(), "rb");
        if (!check) {
            std::cout << " [MISSING!]\n";
            missingCount++;

            if (debugLog) {
                fprintf(debugLog, "  - %s [MISSING!]\n", video.c_str());
            }
        } else {
            // Get file size
            fseek(check, 0, SEEK_END);
            long size = ftell(check);
            std::cout << " [OK, " << size << " bytes]\n";
            fclose(check);

            if (debugLog) {
                fprintf(debugLog, "  - %s [OK, %ld bytes]\n", video.c_str(), size);
            }
        }
    }
    fclose(f);

    if (missingCount > 0) {
        m_lastError = "Cannot concatenate: " + std::to_string(missingCount) + " segment files are missing!";

        if (debugLog) {
            fprintf(debugLog, "ERROR: %d segment files are missing!\n", missingCount);
            fclose(debugLog);
        }

        std::remove(listFile.c_str());
        return false;
    }

    // Close debug log after successful validation
    if (debugLog) {
        fclose(debugLog);
    }

    // Use FFmpeg command-line to concatenate pre-normalized segments
    // Since all segments are now normalized to same resolution/fps/format,
    // we prefer stream copy for fast concatenation but capture FFmpeg output
    // and fall back to a re-encode if we detect timestamp/PTS/DTS problems.
    std::string ffmpegPath = getFFmpegPath();

    // If transitions are enabled and there are two or more input videos,
    // attempt to build a chained gltransition filter_complex to transition
    // between adjacent clips. This handles N>=2 transparently.
    if (m_effects.enableTransitions && inputVideos.size() >= 2) {
        // Resolve transitions directory (same heuristic used elsewhere)
        std::string transitionsDir;
#ifdef _WIN32
        char exePath[MAX_PATH] = {0};
        if (GetModuleFileNameA(NULL, exePath, MAX_PATH)) {
            std::filesystem::path exeDir = std::filesystem::path(exePath).parent_path();
            transitionsDir = (exeDir / "assets" / "transitions").string();
        }
#endif
        if (transitionsDir.empty()) {
            // Fallback: current working directory + assets/transitions
            transitionsDir = (std::filesystem::current_path() / "assets" / "transitions").string();
        }

        TransitionLibrary library;
        if (!library.loadFromDirectory(transitionsDir)) {
            // Log and fall back to standard concat
            FILE* logf = fopen((getTempDir() + "beatsync_ffmpeg_concat.log").c_str(), "a");
            if (logf) {
                fprintf(logf, "Transitions enabled but failed to load library: %s\n", library.getLastError().c_str());
                fclose(logf);
            }
        } else {
            const TransitionShader* t = library.findByName(m_effects.transitionType);
            if (!t) {
                // Try default 'fade' as fallback
                t = library.findByName("fade");
            }

            if (t) {
                // IMPORTANT: For very long videos with many beats (600+ segments),
                // transitions are disabled to avoid command line length limits and memory issues.
                // The Windows command line limit is ~8191 chars, and with 600 inputs at ~60 chars each,
                // plus filter_complex, the command would exceed 50KB+ causing crashes.
                const size_t MAX_TRANSITION_INPUTS = 100;  // Safe limit for transitions
                if (inputVideos.size() > MAX_TRANSITION_INPUTS) {
                    FILE* logf = fopen((getTempDir() + "beatsync_ffmpeg_concat.log").c_str(), "a");
                    if (logf) {
                        fprintf(logf, "WARNING: %zu inputs exceeds transition limit of %zu. Falling back to standard concat.\n",
                                inputVideos.size(), MAX_TRANSITION_INPUTS);
                        fclose(logf);
                    }
                    std::cerr << "[Beatsync] " << inputVideos.size() << " segments exceeds transition limit. Using standard concat.\n";
                    // Fall through to standard concat below
                } else {
                    // Build a chained filter_complex for N inputs
                    std::string filterComplex = buildGlTransitionFilterComplex(inputVideos.size(), t->name, m_effects.transitionDuration);

                    // Build ffmpeg command with all inputs
                    std::ostringstream cmd;
                    cmd << "\"" << ffmpegPath << "\" -nostdin";
                    for (const auto &v : inputVideos) {
                        cmd << " -i \"" << v << "\"";
                    }

                    // Build audio portion of filter_complex
                    // For audio, we'll use a simpler approach: just use anullsrc to avoid probing hundreds of files
                    // This is more reliable for long sequences and avoids memory pressure from libavformat probing
                    std::ostringstream audioFilter;

                    // Check all files for audio - any file having audio means we should concatenate audio
                    // We stop at first file with audio to avoid excessive probing
                    bool sourceHasAudio = false;
                    for (const auto& videoPath : inputVideos) {
                        AVFormatContext* probeCtx = nullptr;
                        int openErr = avformat_open_input(&probeCtx, videoPath.c_str(), nullptr, nullptr);
                        if (openErr == 0) {
                            int infoErr = avformat_find_stream_info(probeCtx, nullptr);
                            if (infoErr >= 0) {
                                for (unsigned int s = 0; s < probeCtx->nb_streams; ++s) {
                                    if (probeCtx->streams[s]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
                                        sourceHasAudio = true;
                                        break;
                                    }
                                }
                            }
                            avformat_close_input(&probeCtx);
                        }
                        if (sourceHasAudio) break;  // Found audio, no need to check more files
                    }

                    // If source has audio, concatenate audio from all inputs
                    // Otherwise use a single anullsrc for the whole output (simpler and faster)
                    if (sourceHasAudio) {
                        for (size_t i = 0; i < inputVideos.size(); ++i) {
                            audioFilter << "[" << i << ":a]asetpts=PTS-STARTPTS[ain" << i << "];";
                        }
                        for (size_t i = 0; i < inputVideos.size(); ++i) {
                            audioFilter << "[ain" << i << "]";
                        }
                        audioFilter << "concat=n=" << inputVideos.size() << ":v=0:a=1[aout]";
                    } else {
                        // Use single anullsrc with -shortest flag to match video length
                        constexpr double fallbackLargeDuration = 36000.0; // 10 hours max
                        audioFilter << "anullsrc=channel_layout=stereo:sample_rate=44100:duration=" << fallbackLargeDuration << "[aout]";
                    }

                    // Combine video transitions and audio concat into full filter_complex
                    std::string fullFilter = filterComplex + ";" + audioFilter.str();
                    std::string finalVideoLabel = "[t" + std::to_string(inputVideos.size()-1) + "]";

                    // Use filter_complex_script file if command is too long (Windows limit ~8191 chars)
                    std::string filterScriptPath;
                    bool useFilterScript = fullFilter.length() > 4000;  // Leave margin for rest of command

                    if (useFilterScript) {
                        filterScriptPath = getTempDir() + "beatsync_filter_complex.txt";
                        FILE* scriptFile = fopen(filterScriptPath.c_str(), "w");
                        if (scriptFile) {
                            fprintf(scriptFile, "%s", fullFilter.c_str());
                            fclose(scriptFile);
                            cmd << " -filter_complex_script \"" << filterScriptPath << "\"";
                        } else {
                            // Fallback to inline (may fail if too long)
                            cmd << " -filter_complex \"" << fullFilter << "\"";
                            useFilterScript = false;
                        }
                    } else {
                        cmd << " -filter_complex \"" << fullFilter << "\"";
                    }

                    cmd << " -map \"" << finalVideoLabel << "\" -map \"[aout]\"";
                    cmd << " " << getEncoderArgs("fast") << " -c:a aac -b:a 192k";
                    if (!sourceHasAudio) {
                        cmd << " -shortest";  // Trim anullsrc to video length
                    }
                    cmd << " -y \"" << outputVideo << "\"";

                    // Execute the command
                    std::string ffmpegOutput;
                    int exitCode;
#ifdef _WIN32
                    exitCode = runHiddenCommand(cmd.str(), ffmpegOutput);
#else
                    std::string fullCmd = cmd.str() + " 2>&1";
                    FILE* pipe = popen_compat(fullCmd.c_str(), "r");
                    if (!pipe) {
                        m_lastError = "Failed to execute FFmpeg for transition";
                        std::remove(listFile.c_str());
                        if (useFilterScript && !filterScriptPath.empty()) {
                            std::remove(filterScriptPath.c_str());
                        }
                        return false;
                    }
                    char buffer[512];
                    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                        ffmpegOutput += buffer;
                    }
                    exitCode = pclose_compat(pipe);
#endif

                    // Clean up filter script
                    if (useFilterScript && !filterScriptPath.empty()) {
                        std::remove(filterScriptPath.c_str());
                    }

                    // Persist log
                    FILE* logf = fopen((getTempDir() + "beatsync_ffmpeg_concat.log").c_str(), "a");
                    if (logf) {
                        fprintf(logf, "\n--- FFmpeg transition run (chained) ---\n");
                        fprintf(logf, "Using filter_complex_script: %s\n", useFilterScript ? "yes" : "no");
                        fprintf(logf, "Filter length: %zu chars\n", fullFilter.length());
                        fprintf(logf, "cmd: %s\nexit: %d\noutput:\n%s\n", cmd.str().c_str(), exitCode, ffmpegOutput.c_str());
                        fclose(logf);
                    }

                    if (exitCode != 0) {
                        m_lastError = "FFmpeg transition chain failed: " + ffmpegOutput.substr(0, 200);
                        // Fall back to standard concat below (do not remove listFile yet)
                    } else {
                        std::remove(listFile.c_str());
                        return true;
                    }
                }  // End of transition processing block
            }
        }
        // If we get here, transition attempt failed - fall back to normal concat
    }

    std::ostringstream cmd;
    cmd << "\"" << ffmpegPath << "\" -nostdin -fflags +genpts+igndts -f concat -safe 0 -i \"" << listFile
        << "\" -c copy -video_track_timescale 90000 -y \"" << outputVideo << "\"";

    // Execute FFmpeg hidden (no console flash on Windows)
    std::string ffmpegOutput;
    int exitCode;
#ifdef _WIN32
    exitCode = runHiddenCommand(cmd.str(), ffmpegOutput, m_cancelFlag);
    if (exitCode == -2) {
        m_lastError = "Cancelled by user";
        std::remove(listFile.c_str());
        return false;
    }
    if (exitCode == -1) {
        m_lastError = "Failed to execute FFmpeg";
        std::remove(listFile.c_str());
        return false;
    }
#else
    std::string fullCmd = cmd.str() + " 2>&1";
    FILE* pipe = popen_compat(fullCmd.c_str(), "r");
    if (!pipe) {
        m_lastError = "Failed to execute FFmpeg";
        std::remove(listFile.c_str());
        return false;
    }
    char buffer[512];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        ffmpegOutput += buffer;
    }
    exitCode = pclose_compat(pipe);
#endif

    // Persist FFmpeg output for debugging
    {
        FILE* logf = fopen((getTempDir() + "beatsync_ffmpeg_concat.log").c_str(), "a");
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
        // Use GPU acceleration if available for faster re-encoding
        std::ostringstream reencodeCmd;
        reencodeCmd << "\"" << ffmpegPath << "\" -nostdin";

        // Add CUDA hardware acceleration for decoding if available
        if (hasCudaHwaccel()) {
            reencodeCmd << " -hwaccel cuda -hwaccel_device 0";
        }

        reencodeCmd << " -fflags +genpts -f concat -safe 0 -i \"" << listFile
                    << "\" " << getEncoderArgs("ultrafast") << " -r " << m_outputFps
                    << " -c:a aac -b:a 192k -video_track_timescale 90000 -y \"" << outputVideo << "\"";

        // Execute re-encode hidden (no console flash on Windows)
        std::string reencodeOutput;
        int rc2;
    #ifdef _WIN32
        rc2 = runHiddenCommand(reencodeCmd.str(), reencodeOutput);
    #else
        std::string fullReencodeCmd = reencodeCmd.str() + " 2>&1";
        FILE* pipe2 = popen_compat(fullReencodeCmd.c_str(), "r");
        if (!pipe2) {
            m_lastError = "FFmpeg re-encode fallback failed to start";
            return false;
        }
        char buffer[512];
        while (fgets(buffer, sizeof(buffer), pipe2) != nullptr) {
            reencodeOutput += buffer;
        }
        rc2 = pclose_compat(pipe2);
    #endif

        // Log re-encode output
        FILE* logf2 = fopen((getTempDir() + "beatsync_ffmpeg_concat.log").c_str(), "a");
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
    cmd << "\"" << ffmpegPath << "\" -nostdin";

    // Combine video from first input with audio from second input
    // -c:v copy = stream copy video (fast, no re-encode)
    // -c:a aac = encode audio as AAC
    // -shortest = trim output to shorter of video/audio (optional)
    double clipStart = std::max(0.0, audioStart);
    bool clipAudio = (audioEnd > 0.0 && audioEnd > audioStart + 1e-3);
    double clipDur = clipAudio ? (audioEnd - audioStart) : 0.0;

    cmd << " -i \"" << inputVideo << "\"";
    if (clipAudio) {
        // Use fixed-point notation - FFmpeg doesn't accept scientific notation
        cmd << std::fixed << std::setprecision(6) << " -ss " << clipStart << " -t " << clipDur << std::defaultfloat;
    }
    cmd << " -i \"" << audioFile << "\""
        << " -c:v copy -c:a aac -b:a 192k"
        << " -map 0:v:0 -map 1:a:0";  // Take video from first input, audio from second

    if (trimToShortest) {
        cmd << " -shortest";
    }

    cmd << " -y \"" << outputVideo << "\"";

    std::cout << "Adding audio track...\n";

    // Execute FFmpeg
    std::string ffmpegOutput;
    int exitCode;
#ifdef _WIN32
    exitCode = runHiddenCommand(cmd.str(), ffmpegOutput);
#else
    std::string fullCmd = cmd.str() + " 2>&1";
    FILE* pipe = popen_compat(fullCmd.c_str(), "r");
    if (!pipe) {
        m_lastError = "Failed to execute FFmpeg for audio muxing";
        return false;
    }
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        ffmpegOutput += buffer;
    }
    exitCode = pclose_compat(pipe);
#endif

    // Persist muxing output for troubleshooting
    {
        FILE* logf = fopen((getTempDir() + "beatsync_ffmpeg_concat.log").c_str(), "a");
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

// ==================== GPU Encoder Detection ====================

// Cache all available encoders in a single FFmpeg call (thread-safe, per-path cache)
static std::set<std::string> getAvailableEncoders(const std::string& ffmpegPath) {
    static std::unordered_map<std::string, std::set<std::string>> cachedEncodersMap;
    static std::mutex cachedEncodersMutex;

    {
        std::lock_guard<std::mutex> lock(cachedEncodersMutex);
        auto it = cachedEncodersMap.find(ffmpegPath);
        if (it != cachedEncodersMap.end()) {
            return it->second;
        }
    }

    // Not cached yet - run FFmpeg to get encoder list
    std::string output;
    std::string cmd = "\"" + ffmpegPath + "\" -hide_banner -encoders";
    std::set<std::string> encoders;

#ifdef _WIN32
    int rc = runHiddenCommand(cmd, output);
    if (rc != 0) {
        std::lock_guard<std::mutex> lock(cachedEncodersMutex);
        cachedEncodersMap[ffmpegPath] = encoders;  // Cache empty set
        return encoders;
    }
#else
    std::string fullCmd = cmd + " 2>&1";
    FILE* pipe = popen_compat(fullCmd.c_str(), "r");
    if (!pipe) {
        std::lock_guard<std::mutex> lock(cachedEncodersMutex);
        cachedEncodersMap[ffmpegPath] = encoders;  // Cache empty set
        return encoders;
    }
    char buffer[512];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }
    int rc = pclose_compat(pipe);
    if (rc != 0) {
        std::lock_guard<std::mutex> lock(cachedEncodersMutex);
        cachedEncodersMap[ffmpegPath] = encoders;  // Cache empty set
        return encoders;
    }
#endif

    // Parse encoder list - format: " V..... h264_nvenc           NVIDIA NVENC H.264 encoder"
    // Extract encoder names from lines starting with " V" (video encoders)
    std::istringstream stream(output);
    std::string line;
    while (std::getline(stream, line)) {
        // Skip header lines and non-video encoders
        if (line.size() < 8 || line[0] != ' ' || line[1] != 'V') continue;

        // Extract encoder name: starts after " V..... " (8 chars)
        size_t nameStart = 8;
        while (nameStart < line.size() && line[nameStart] == ' ') nameStart++;
        if (nameStart >= line.size()) continue;

        size_t nameEnd = line.find(' ', nameStart);
        if (nameEnd == std::string::npos) nameEnd = line.size();

        std::string encoderName = line.substr(nameStart, nameEnd - nameStart);
        if (!encoderName.empty()) {
            encoders.insert(encoderName);
        }
    }

    // Cache and return
    {
        std::lock_guard<std::mutex> lock(cachedEncodersMutex);
        cachedEncodersMap[ffmpegPath] = encoders;
    }
    return encoders;
}

bool VideoWriter::probeEncoder(const std::string& encoder) const {
    auto encoders = getAvailableEncoders(getFFmpegPath());
    bool found = encoders.find(encoder) != encoders.end();
    // Log first time each encoder is probed (thread-safe)
    static std::set<std::string> loggedEncoders;
    static std::mutex loggedEncodersMutex;
    {
        std::lock_guard<std::mutex> lock(loggedEncodersMutex);
        if (loggedEncoders.find(encoder) == loggedEncoders.end()) {
            std::cout << "[GPU DEBUG] probeEncoder(" << encoder << ") = " << (found ? "YES" : "NO") << "\n";
            loggedEncoders.insert(encoder);
        }
    }
    return found;
}

GPUEncoderInfo VideoWriter::detectBestEncoder(const std::string& speedPreset) const {
    std::lock_guard<std::recursive_mutex> lock(m_cacheMutex);
    // Return cached result if available
    if (m_encoderCacheValid) {
        // Adjust preset for cached encoder
        GPUEncoderInfo result = m_cachedEncoder;
        if (result.encoderName == "h264_nvenc") {
            result.preset = (speedPreset == "ultrafast") ? "p1" :
                           (speedPreset == "fast") ? "p4" : "p5";
        } else if (result.encoderName == "h264_amf") {
            result.preset = (speedPreset == "ultrafast") ? "speed" :
                           (speedPreset == "fast") ? "balanced" : "quality";
        } else if (result.encoderName == "h264_qsv") {
            result.preset = (speedPreset == "ultrafast") ? "veryfast" :
                           (speedPreset == "fast") ? "fast" : "medium";
        } else {
            result.preset = speedPreset;
        }
        return result;
    }

    // Try NVIDIA NVENC first (most common high-end GPU)
    if (probeEncoder("h264_nvenc")) {
        std::string preset = (speedPreset == "ultrafast") ? "p1" :
                            (speedPreset == "fast") ? "p4" : "p5";
        m_cachedEncoder = {"h264_nvenc", preset, true};
        m_encoderCacheValid = true;
        std::cout << "[GPU] Detected NVIDIA NVENC encoder\n";
        return m_cachedEncoder;
    }

    // Try AMD AMF
    if (probeEncoder("h264_amf")) {
        std::string preset = (speedPreset == "ultrafast") ? "speed" :
                            (speedPreset == "fast") ? "balanced" : "quality";
        m_cachedEncoder = {"h264_amf", preset, true};
        m_encoderCacheValid = true;
        std::cout << "[GPU] Detected AMD AMF encoder\n";
        return m_cachedEncoder;
    }

    // Try Intel Quick Sync
    if (probeEncoder("h264_qsv")) {
        std::string preset = (speedPreset == "ultrafast") ? "veryfast" :
                            (speedPreset == "fast") ? "fast" : "medium";
        m_cachedEncoder = {"h264_qsv", preset, true};
        m_encoderCacheValid = true;
        std::cout << "[GPU] Detected Intel Quick Sync encoder\n";
        return m_cachedEncoder;
    }

    // Fallback to software encoder
    m_cachedEncoder = {"libx264", speedPreset, false};
    m_encoderCacheValid = true;
    std::cout << "[GPU] No hardware encoder found, using software libx264\n";
    return m_cachedEncoder;
}

std::string VideoWriter::getEncoderArgs(const std::string& speedPreset) const {
    GPUEncoderInfo enc = detectBestEncoder(speedPreset);
    std::ostringstream args;

    if (enc.encoderName == "h264_nvenc") {
        // NVIDIA NVENC: Use VBR with CQ mode for quality control
        // -rc vbr -cq gives similar quality to libx264's CRF mode
        args << "-c:v h264_nvenc -preset " << enc.preset
             << " -rc vbr -cq 18 -pix_fmt yuv420p";
    } else if (enc.encoderName == "h264_amf") {
        // AMD AMF: Use CQP mode for quality control
        args << "-c:v h264_amf -quality " << enc.preset
             << " -rc cqp -qp_i 18 -qp_p 18 -pix_fmt yuv420p";
    } else if (enc.encoderName == "h264_qsv") {
        // Intel Quick Sync: Use global_quality for CRF-like mode
        args << "-c:v h264_qsv -preset " << enc.preset
             << " -global_quality 18 -pix_fmt yuv420p";
    } else {
        // Software fallback (libx264)
        args << "-c:v libx264 -preset " << enc.preset
             << " -crf 18 -pix_fmt yuv420p";
    }

    return args.str();
}

bool VideoWriter::hasCudaHwaccel() const {
    std::lock_guard<std::recursive_mutex> lock(m_cacheMutex);
    // Return cached result if available
    if (m_cudaHwaccelCache >= 0) {
        return m_cudaHwaccelCache == 1;
    }

    std::string ffmpegPath = getFFmpegPath();
    std::string cmd = "\"" + ffmpegPath + "\" -hwaccels 2>&1";
    std::string output;

#ifdef _WIN32
    int exitCode = runHiddenCommand(cmd, output);
    (void)exitCode;
#else
    FILE* pipe = popen_compat(cmd.c_str(), "r");
    if (pipe) {
        char buffer[256];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            output += buffer;
        }
        pclose_compat(pipe);
    }
#endif

    m_cudaHwaccelCache = (output.find("cuda") != std::string::npos) ? 1 : 0;
    if (m_cudaHwaccelCache == 1) {
        std::cout << "[GPU] CUDA hardware acceleration available for decoding\n";
    } else {
        std::cout << "[GPU] CUDA hardware acceleration NOT available\n";
    }
    return m_cudaHwaccelCache == 1;
}

bool VideoWriter::hasScaleCudaFilter() const {
    std::lock_guard<std::recursive_mutex> lock(m_cacheMutex);
    // Return cached result if available
    if (m_scaleCudaCache >= 0) {
        return m_scaleCudaCache == 1;
    }

    std::string ffmpegPath = getFFmpegPath();
    std::string cmd = "\"" + ffmpegPath + "\" -filters 2>&1";
    std::string output;

#ifdef _WIN32
    int exitCode = runHiddenCommand(cmd, output);
    (void)exitCode;
#else
    FILE* pipe = popen_compat(cmd.c_str(), "r");
    if (pipe) {
        char buffer[256];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            output += buffer;
        }
        pclose_compat(pipe);
    }
#endif

    m_scaleCudaCache = (output.find("scale_cuda") != std::string::npos) ? 1 : 0;
    if (m_scaleCudaCache == 1) {
        std::cout << "[GPU] scale_cuda filter available\n";
    } else {
        std::cout << "[GPU] scale_cuda filter NOT available (using CPU scaling)\n";
    }
    return m_scaleCudaCache == 1;
}

void VideoWriter::logGpuCapabilities() const {
    std::cout << "[VideoWriter] GPU Capabilities:\n";
    std::cout << "  - CUDA hwaccel: " << (hasCudaHwaccel() ? "YES" : "NO") << "\n";
    std::cout << "  - scale_cuda filter: " << (hasScaleCudaFilter() ? "YES" : "NO") << "\n";

    GPUEncoderInfo enc = detectBestEncoder("fast");
    std::cout << "  - Encoder: " << enc.encoderName << " (Hardware: " << (enc.isHardware ? "YES" : "NO") << ")\n";
    std::cout << "  - GPU reset interval: " << GPU_RESET_INTERVAL << " segments\n";
}

bool VideoWriter::shouldUseGpuForSegment() {
    // Periodically fall back to CPU to force FFmpeg to reset CUDA contexts and avoid resource leaks
    // See header for details. This helps long-running jobs on some drivers.
    std::lock_guard<std::recursive_mutex> lock(m_cacheMutex);
    if (++m_segmentsSinceGpuReset >= GPU_RESET_INTERVAL) {
        m_segmentsSinceGpuReset = 0;
        return false; // Use CPU for this segment
    }
    return true;
}

void VideoWriter::resetSegmentCounter() {
    std::lock_guard<std::recursive_mutex> lock(m_cacheMutex);
    m_segmentsSinceGpuReset = 0;
}

// ==================== End GPU Encoder Detection ====================

void VideoWriter::setEffectsConfig(const EffectsConfig& config) {
    m_effects = config;
}

void VideoWriter::setSpeedConfig(const SpeedRampConfig& config) {
    m_speed = config;
    m_speedClampCount.store(0);
}

std::vector<double> VideoWriter::computeClipSpeeds(size_t clipCount, const SpeedRampConfig& config) {
    std::vector<double> speeds(clipCount, 1.0);
    if (!config.enabled || clipCount == 0) {
        return speeds;
    }

    // Clamp configuration into valid ranges.
    auto clampSpeed = [](float v) {
        return std::max(SpeedRampConfig::kMinSpeed, std::min(SpeedRampConfig::kMaxSpeed, v));
    };
    const double slowLo = clampSpeed(std::min(config.slowMin, config.slowMax));
    const double slowHi = clampSpeed(std::max(config.slowMin, config.slowMax));
    const double fastLo = clampSpeed(std::min(config.fastMin, config.fastMax));
    const double fastHi = clampSpeed(std::max(config.fastMin, config.fastMax));
    const float affected = std::max(0.0f, std::min(1.0f, config.affectedFraction));
    const float upFrac = std::max(0.0f, std::min(1.0f, config.speedUpFraction));
    const int everyN = std::max(1, config.everyN);

    // Single deterministic RNG seeded by config.seed → reproducible edits.
    std::mt19937 rng(config.seed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    // Energy-band mode: speed comes straight from the clip's band, no randomness,
    // so calm sections melt and drops stay at full rate.
    if (config.selectionMode == 3 && !config.beatBands.empty()) {
        for (size_t i = 0; i < clipCount; ++i) {
            const int band = (i < config.beatBands.size()) ? config.beatBands[i] : 1;
            const double s = config.bandSpeed[(band >= 0 && band <= 2) ? band : 1];
            speeds[i] = clampSpeed(static_cast<float>(s));
        }
        return speeds;
    }

    // Decide which clip indices are affected.
    std::vector<size_t> selected;
    if (config.selectionMode == 1 || config.selectionMode == 2) {
        // Deterministic every-Nth selection (mode 2 is treated the same here;
        // beat-divisor nuance can be layered in by the caller via everyN).
        for (size_t i = 0; i < clipCount; ++i) {
            if ((i % static_cast<size_t>(everyN)) == 0) {
                selected.push_back(i);
            }
        }
    } else {
        // Random selection of round(affected * clipCount) distinct clips.
        size_t target = static_cast<size_t>(std::llround(affected * static_cast<double>(clipCount)));
        target = std::min(target, clipCount);
        std::vector<size_t> pool(clipCount);
        for (size_t i = 0; i < clipCount; ++i) pool[i] = i;
        std::shuffle(pool.begin(), pool.end(), rng);
        selected.assign(pool.begin(), pool.begin() + target);
    }

    // Assign direction + amount to each selected clip.
    for (size_t idx : selected) {
        const bool speedUp = (unit(rng) < upFrac);
        double s;
        if (speedUp) {
            s = (fastHi > fastLo) ? (fastLo + unit(rng) * (fastHi - fastLo)) : fastLo;
        } else {
            s = (slowHi > slowLo) ? (slowLo + unit(rng) * (slowHi - slowLo)) : slowLo;
        }
        speeds[idx] = clampSpeed(static_cast<float>(s));
    }

    return speeds;
}

bool VideoWriter::extractSpeedClip(const std::string& inputVideo,
                                   double sourceStart,
                                   double outputDuration,
                                   double speed,
                                   const std::string& outputVideo) {
    // Neural interpolation (RIFE) for smooth slow-mo. Only meaningful for
    // slow-mo (speed < 1.0); a speed-up discards frames so dup/minterpolate is
    // fine. Fall back to minterpolate if the model/pipeline is unavailable.
    if (m_speed.smoothing == 2 && speed < 1.0) {
        if (extractSpeedClipInterpolated(inputVideo, sourceStart, outputDuration, speed, outputVideo)) {
            return true;
        }
        std::cout << "[RIFE] Interpolation unavailable (" << getLastError()
                  << "); falling back to minterpolate for this clip\n";
        return copySegmentPrecise(inputVideo, sourceStart, outputDuration, outputVideo, speed, 1);
    }

    // Speed clips always re-encode (stream copy cannot retime). Honor the
    // configured smoothing mode (0 dup / 1 minterpolate). copySegmentPrecise
    // handles speed == 1.0 too.
    int smoothing = (m_speed.smoothing == 2) ? 1 : m_speed.smoothing;
    return copySegmentPrecise(inputVideo, sourceStart, outputDuration,
                              outputVideo, speed, smoothing);
}

void VideoWriter::setInterpolationModelPath(const std::string& path) {
    std::lock_guard<std::mutex> lock(m_interpMutex);
    if (path != m_interpModelPath) {
        m_interpModelPath = path;
        m_interpolator.reset();
        m_interpLoadAttempted = false;
    }
}

void VideoWriter::setUpscaleConfig(const UpscaleConfig& config) {
    std::lock_guard<std::mutex> lock(m_upscaleMutex);
    m_upscale = config;
    // Config change invalidates a previously loaded model
    m_upscaler.reset();
    m_upscaleLoadAttempted = false;
}

bool VideoWriter::ensureUpscaler() {
    std::lock_guard<std::mutex> lock(m_upscaleMutex);
    if (m_upscaler && m_upscaler->isLoaded()) return true;
    if (m_upscaleLoadAttempted) return m_upscaler && m_upscaler->isLoaded();

    m_upscaleLoadAttempted = true;
    if (!OnnxUpscaler::isAvailable()) {
        m_lastError = "ONNX Runtime not compiled in (no upscaler)";
        return false;
    }
    std::string modelPath = m_upscale.modelPath.empty() ? std::string("models/upscale.onnx")
                                                        : m_upscale.modelPath;
    if (!std::filesystem::exists(modelPath)) {
        m_lastError = "Upscale model not found: " + modelPath;
        return false;
    }
    m_upscaler = std::make_unique<OnnxUpscaler>();
    m_upscaler->setTileSize(m_upscale.tileSize);
    if (!m_upscaler->loadModel(modelPath, /*useGPU=*/true)) {
        m_lastError = "Upscale model load failed: " + m_upscaler->getLastError();
        m_upscaler.reset();
        return false;
    }
    std::cout << "[Upscale] Model loaded (" << m_upscaler->getScale() << "x): " << modelPath << "\n";
    return true;
}

bool VideoWriter::upscaleVideo(const std::string& inputVideo, const std::string& outputVideo) {
    if (!m_upscale.enabled) {
        m_lastError = "Upscaling not enabled";
        return false;
    }

    // Probe the source: skip work when it is already large enough, and keep the
    // native frame rate so no frames are dropped or duplicated here.
    int srcW = 0, srcH = 0;
    double srcFps = 0.0;
    {
        VideoProcessor proc;
        if (proc.open(inputVideo)) {
            const auto info = proc.getInfo();
            srcW = info.width;
            srcH = info.height;
            srcFps = info.fps;
            proc.close();
        }
    }
    if (srcW <= 0 || srcH <= 0) {
        m_lastError = "Upscale: could not probe source dimensions";
        return false;
    }
    if (std::max(srcW, srcH) >= m_upscale.maxSourceEdge) {
        m_lastError = "Upscale skipped: source already " + std::to_string(srcW) + "x" +
                      std::to_string(srcH);
        return false;
    }
    if (srcFps <= 0.0) srcFps = m_outputFps > 0 ? m_outputFps : 30;

    if (!ensureUpscaler()) {
        return false;  // m_lastError set by ensureUpscaler
    }
    const int scale = m_upscaler->getScale();

    std::string tempDir = getTempDir();
    if (tempDir.empty()) {
        m_lastError = "Upscale: no temp directory";
        return false;
    }
    std::ostringstream tag;
    tag << "upscale_" << std::this_thread::get_id() << "_"
        << std::chrono::steady_clock::now().time_since_epoch().count();
    const std::string srcRaw = tempDir + tag.str() + "_src.rgb";
    const std::string outRaw = tempDir + tag.str() + "_out.rgb";

    struct RawCleanup {
        std::vector<std::string> files;
        ~RawCleanup() { for (auto& f : files) std::remove(f.c_str()); }
    } cleanup{{srcRaw, outRaw}};

    const std::string ffmpegPath = getFFmpegPath();

    // Stage 1: decode to raw RGB24 at native size
    {
        std::ostringstream cmd;
        cmd << "\"" << ffmpegPath << "\" -nostdin -i \"" << inputVideo << "\""
            << " -vf \"format=rgb24\" -f rawvideo -y \"" << srcRaw << "\"";
        std::string out; int rc;
#ifdef _WIN32
        rc = runHiddenCommand(cmd.str(), out, m_cancelFlag);
#else
        // Drain the pipe: ffmpeg writes progress continuously and would block
        // on a full buffer if nobody reads it.
        FILE* p = popen_compat((cmd.str() + " 2>&1").c_str(), "r");
        if (!p) {
            m_lastError = "Upscale: could not start FFmpeg";
            return false;
        }
        char buf[512];
        while (fgets(buf, sizeof(buf), p)) out += buf;
        rc = pclose_compat(p);
#endif
        if (rc != 0) {
            appendFfmpegLog("beatsync_ffmpeg_upscale.log", "upscaleVideo::extract", cmd.str(), rc, out, "");
            size_t lastLine = out.find_last_not_of("\n");
            if (lastLine != std::string::npos) {
                size_t start = out.rfind('\n', lastLine);
                out = out.substr(start == std::string::npos ? 0 : start + 1);
            }
            m_lastError = "Upscale: source frame extraction failed: " + out;
            return false;
        }
    }

    // Stage 2: upscale every frame, streaming to avoid holding a whole clip in RAM
    const size_t inFrameBytes = static_cast<size_t>(srcW) * srcH * 3;
    size_t frameCount = 0;
    {
        std::ifstream in(srcRaw, std::ios::binary);
        std::ofstream out(outRaw, std::ios::binary);
        if (!in || !out) {
            m_lastError = "Upscale: could not open raw frame buffers";
            return false;
        }
        std::vector<uint8_t> frame(inFrameBytes);
        std::vector<uint8_t> upscaled;
        while (in.read(reinterpret_cast<char*>(frame.data()), inFrameBytes)) {
            if (m_cancelFlag && *m_cancelFlag != 0) {
                m_lastError = "Cancelled by user";
                return false;
            }
            if (!m_upscaler->upscale(frame.data(), srcW, srcH, upscaled)) {
                m_lastError = "Upscale inference failed: " + m_upscaler->getLastError();
                return false;
            }
            out.write(reinterpret_cast<const char*>(upscaled.data()),
                      static_cast<std::streamsize>(upscaled.size()));
            ++frameCount;
            if (frameCount % 50 == 0) {
                reportProgress(0.0);  // keeps cancel-aware callers responsive
            }
        }
    }
    if (frameCount == 0) {
        m_lastError = "Upscale: no frames decoded";
        return false;
    }

    // Stage 3: encode the upscaled sequence, carrying the original audio over
    {
        std::ostringstream cmd;
        cmd << "\"" << ffmpegPath << "\" -nostdin"
            << " -f rawvideo -pix_fmt rgb24 -s " << (srcW * scale) << "x" << (srcH * scale)
            << " -r " << srcFps << " -i \"" << outRaw << "\""
            << " -i \"" << inputVideo << "\""
            << " -map 0:v -map \"1:a?\" -shortest "
            << getEncoderArgs("fast")
            << " -c:a aac -b:a 192k -video_track_timescale 90000 -y \"" << outputVideo << "\"";
        std::string out; int rc;
#ifdef _WIN32
        rc = runHiddenCommand(cmd.str(), out, m_cancelFlag);
#else
        FILE* p = popen_compat((cmd.str() + " 2>&1").c_str(), "r");
        if (!p) {
            m_lastError = "Upscale: could not start FFmpeg for encode";
            return false;
        }
        char buf[512];
        while (fgets(buf, sizeof(buf), p)) out += buf;
        rc = pclose_compat(p);
#endif
        if (rc != 0) {
            appendFfmpegLog("beatsync_ffmpeg_upscale.log", "upscaleVideo::encode", cmd.str(), rc, out, "");
            size_t lastLine = out.find_last_not_of("\n");
            if (lastLine != std::string::npos) {
                size_t start = out.rfind('\n', lastLine);
                out = out.substr(start == std::string::npos ? 0 : start + 1);
            }
            m_lastError = "Upscale: encode failed: " + out;
            return false;
        }
    }

    std::cout << "[Upscale] " << srcW << "x" << srcH << " -> " << (srcW * scale) << "x"
              << (srcH * scale) << " (" << frameCount << " frames)\n";
    return true;
}

bool VideoWriter::ensureInterpolator() {
    std::lock_guard<std::mutex> lock(m_interpMutex);
    if (m_interpolator && m_interpolator->isLoaded()) return true;
    if (m_interpLoadAttempted) return m_interpolator && m_interpolator->isLoaded();

    m_interpLoadAttempted = true;
    if (!OnnxFrameInterpolator::isAvailable()) {
        m_lastError = "ONNX Runtime not compiled in (no RIFE)";
        return false;
    }
    if (m_interpModelPath.empty()) {
        m_lastError = "No RIFE model path set";
        return false;
    }
    if (!std::filesystem::exists(m_interpModelPath)) {
        m_lastError = "RIFE model not found: " + m_interpModelPath;
        return false;
    }
    m_interpolator = std::make_unique<OnnxFrameInterpolator>();
    if (!m_interpolator->loadModel(m_interpModelPath, /*useGPU=*/true)) {
        m_lastError = "RIFE model load failed: " + m_interpolator->getLastError();
        m_interpolator.reset();
        return false;
    }
    std::cout << "[RIFE] Model loaded: " << m_interpModelPath << "\n";
    return true;
}

bool VideoWriter::extractSpeedClipInterpolated(const std::string& inputVideo,
                                               double sourceStart,
                                               double outputDuration,
                                               double speed,
                                               const std::string& outputVideo) {
    if (!ensureInterpolator()) {
        return false;  // m_lastError set by ensureInterpolator
    }
    if (sourceStart < 0.001) sourceStart = 0.0;

    // Probe source frame rate (to know how many real frames we have to work with).
    double srcFps = 0.0;
    {
        VideoProcessor proc;
        if (proc.open(inputVideo)) {
            srcFps = proc.getInfo().fps;
            proc.close();
        }
    }
    if (srcFps <= 0.0) srcFps = m_outputFps > 0 ? m_outputFps : 30;

    const int W = m_outputWidth;
    const int H = m_outputHeight;
    const double windowDur = outputDuration * speed;  // source seconds consumed
    if (W <= 0 || H <= 0 || windowDur <= 0.0) {
        m_lastError = "Invalid interpolation parameters";
        return false;
    }

    std::string tempDir = getTempDir();
    if (tempDir.empty()) { m_lastError = "No temp dir for interpolation"; return false; }

    // Unique temp basenames (this runs across worker threads).
    std::ostringstream tag;
    tag << "rife_" << std::this_thread::get_id() << "_"
        << std::chrono::steady_clock::now().time_since_epoch().count();
    std::string srcRaw = tempDir + tag.str() + "_src.rgb";
    std::string outRaw = tempDir + tag.str() + "_out.rgb";

    struct RawCleanup {
        std::vector<std::string> files;
        ~RawCleanup() { for (auto& f : files) std::remove(f.c_str()); }
    } cleanup{{srcRaw, outRaw}};

    // Stage 1: decode the source window to raw RGB24 frames at output WxH and
    // the source frame rate (scaled+padded to match the rest of the pipeline).
    const std::string ffmpegPath = getFFmpegPath();
    {
        std::ostringstream cmd;
        cmd << "\"" << ffmpegPath << "\" -nostdin";
        cmd << std::fixed << std::setprecision(6);
        cmd << " -ss " << sourceStart << " -t " << windowDur;
        cmd << std::defaultfloat;
        cmd << " -i \"" << inputVideo << "\"";
        cmd << " -vf \"scale=" << W << ":" << H
            << ":force_original_aspect_ratio=decrease,pad=" << W << ":" << H
            << ":(ow-iw)/2:(oh-ih)/2,setsar=1,fps=" << srcFps << ",format=rgb24\"";
        cmd << " -f rawvideo -y \"" << srcRaw << "\"";

        std::string out; int rc;
#ifdef _WIN32
        rc = runHiddenCommand(cmd.str(), out, m_cancelFlag);
#else
        FILE* p = popen_compat((cmd.str() + " 2>&1").c_str(), "r");
        rc = p ? pclose_compat(p) : -1;
#endif
        if (rc != 0) { m_lastError = "RIFE: source frame extraction failed"; return false; }
    }

    const size_t frameBytes = static_cast<size_t>(W) * H * 3;
    // Read all source frames into memory (beat clips are short).
    std::vector<std::vector<uint8_t>> srcFrames;
    {
        std::ifstream in(srcRaw, std::ios::binary);
        if (!in) { m_lastError = "RIFE: could not read extracted frames"; return false; }
        std::vector<uint8_t> buf(frameBytes);
        while (in.read(reinterpret_cast<char*>(buf.data()), frameBytes)) {
            srcFrames.push_back(buf);
        }
    }
    const int Ns = static_cast<int>(srcFrames.size());
    if (Ns == 0) { m_lastError = "RIFE: no source frames decoded"; return false; }
    if (Ns == 1) {
        // Single source frame: nothing to interpolate between. Let the caller
        // fall back to the (dup-frame) precise path.
        m_lastError = "RIFE: only one source frame in window";
        return false;
    }

    // Stage 2: synthesize the output frame sequence at the output fps. Output
    // frame k maps to source position s = k*(Ns-1)/(No-1); interpolate between
    // the bracketing source frames at the fractional timestep.
    const int No = std::max(2, static_cast<int>(std::lround(outputDuration * m_outputFps)));
    std::ofstream outFile(outRaw, std::ios::binary);
    if (!outFile) { m_lastError = "RIFE: could not open output frame file"; return false; }

    std::vector<uint8_t> interp;
    for (int k = 0; k < No; ++k) {
        if (isCancelled()) { m_lastError = "Cancelled"; return false; }
        double s = (No > 1) ? (static_cast<double>(k) * (Ns - 1) / (No - 1)) : 0.0;
        int i0 = static_cast<int>(std::floor(s));
        i0 = std::min(i0, Ns - 1);
        double frac = s - i0;

        if (frac < 1e-3 || i0 + 1 >= Ns) {
            const auto& f = srcFrames[std::min(i0, Ns - 1)];
            outFile.write(reinterpret_cast<const char*>(f.data()), frameBytes);
        } else {
            if (!m_interpolator->interpolate(srcFrames[i0].data(), srcFrames[i0 + 1].data(),
                                             W, H, static_cast<float>(frac), interp)) {
                m_lastError = "RIFE: inference failed: " + m_interpolator->getLastError();
                return false;
            }
            outFile.write(reinterpret_cast<const char*>(interp.data()), frameBytes);
        }
    }
    outFile.close();

    // Stage 3: encode the synthesized frames at the output fps into the slot
    // clip (silent AAC track keeps segment streams consistent for concat; the
    // master audio is muxed over the whole timeline later).
    {
        std::ostringstream cmd;
        cmd << "\"" << ffmpegPath << "\" -nostdin";
        cmd << " -f rawvideo -pix_fmt rgb24 -s " << W << "x" << H
            << " -r " << m_outputFps << " -i \"" << outRaw << "\"";
        cmd << " -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100";
        cmd << " -vf format=yuv420p";
        cmd << " " << getEncoderArgs("ultrafast");
        cmd << " -c:a aac -b:a 192k -ar 44100 -shortest";
        cmd << " -video_track_timescale 90000";
        cmd << std::fixed << std::setprecision(6) << " -t " << outputDuration << std::defaultfloat;
        cmd << " -y \"" << outputVideo << "\"";

        std::string out; int rc;
#ifdef _WIN32
        rc = runHiddenCommand(cmd.str(), out, m_cancelFlag);
#else
        FILE* p = popen_compat((cmd.str() + " 2>&1").c_str(), "r");
        rc = p ? pclose_compat(p) : -1;
#endif
        if (rc != 0) { m_lastError = "RIFE: output encode failed"; return false; }
    }

    return true;
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

    // Helper: resolve effective [start, end] for a given per-effect range.
    // If the per-effect end is <= 0, fall back to the global effectStartTime/effectEndTime.
    auto resolveRange = [&](double perStart, double perEnd,
                             double& outStart, double& outEnd) {
        if (perEnd <= 0.0) {
            // Default value — use the global fallback
            outStart = m_effects.effectStartTime;
            outEnd   = m_effects.effectEndTime;
        } else {
            outStart = perStart;
            outEnd   = perEnd;
        }
    };

    // Helper: build a FFmpeg enable expression string for a time range.
    // Returns empty string when the range covers the whole video (no gating needed).
    auto buildEnableExpr = [&](double start, double end) -> std::string {
        bool hasStart = (start > 0.0);
        bool hasEnd   = (end > 0.0);
        if (!hasStart && !hasEnd) return "";  // Whole video — no enable clause

        std::ostringstream expr;
        expr << std::fixed << std::setprecision(6);
        if (hasStart && hasEnd) {
            expr << ":enable='between(t," << start << "," << end << ")'";
        } else if (hasStart) {
            expr << ":enable='gte(t," << start << ")'";
        } else {
            // hasEnd only
            expr << ":enable='lte(t," << end << ")'";
        }
        return expr.str();
    };

    // Color grading
    if (m_effects.enableColorGrade && m_effects.colorPreset != "none") {
        std::string colorFilter = getColorGradeFilter(m_effects.colorPreset);
        if (!colorFilter.empty()) {
            double cStart, cEnd;
            resolveRange(m_effects.colorGradeStartTime, m_effects.colorGradeEndTime, cStart, cEnd);
            colorFilter += buildEnableExpr(cStart, cEnd);
            filters.push_back(colorFilter);
        }
    }

    // Vignette
    if (m_effects.enableVignette) {
        double vStart, vEnd;
        resolveRange(m_effects.vignetteStartTime, m_effects.vignetteEndTime, vStart, vEnd);
        std::ostringstream vig;
        vig << "vignette=PI/" << (4.0 / m_effects.vignetteStrength);
        vig << buildEnableExpr(vStart, vEnd);
        filters.push_back(vig.str());
    }

    // Blur (no per-effect range — blur has no independent range field)
    if (m_effects.enableBlur) {
        std::ostringstream blur;
        blur << "gblur=sigma=" << m_effects.blurStrength;
        filters.push_back(blur.str());
    }

    // Beat zoom pulse effect is now handled in applyEffects() for proper beat filtering
    // This section is intentionally left empty - zoom uses filtered beats in applyEffects

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

std::string VideoWriter::buildGlTransitionFilterComplex(size_t numInputs, const std::string& transitionName, double duration) const {
    if (numInputs < 2) return "";

    // Use TransitionLibrary to resolve transition shader and build per-edge filter
    TransitionLibrary lib;
    std::string transitionsDir;
#ifdef _WIN32
    char exePath[MAX_PATH] = {0};
    if (GetModuleFileNameA(NULL, exePath, MAX_PATH)) {
        std::filesystem::path exeDir = std::filesystem::path(exePath).parent_path();
        transitionsDir = (exeDir / "assets" / "transitions").string();
    }
#endif
    if (transitionsDir.empty()) {
        transitionsDir = (std::filesystem::current_path() / "assets" / "transitions").string();
    }

    if (!lib.loadFromDirectory(transitionsDir)) {
        return "";
    }

    const TransitionShader* t = lib.findByName(transitionName);
    if (!t) {
        t = lib.findByName("fade");
        if (!t) return "";
    }

    std::string transitionFilter = lib.buildGlTransitionFilter(t->name, duration);
    if (transitionFilter.empty()) return "";

    std::ostringstream fc;

    // Build chained transitions: [0:v][1:v] -> [t1]; [t1][2:v] -> [t2]; ...
    for (size_t i = 0; i < numInputs - 1; ++i) {
        std::string inA = (i == 0) ? (std::string("[0:v]")) : (std::string("[t") + std::to_string(i) + "]");
        std::string inB = std::string("[") + std::to_string(i+1) + ":v]";
        std::string out = std::string("[t") + std::to_string(i+1) + "]";
        fc << inA << inB << transitionFilter << out;
        if (i + 1 < numInputs - 1) fc << ";";
    }

    return fc.str();
}

bool VideoWriter::applyEffects(const std::string& inputVideo, const std::string& outputVideo) {
    std::string filterChain = buildEffectsFilterChain();

    // Resolve per-effect time ranges.
    // Rule: if an effect's own end <= 0, fall back to the global effectStartTime/effectEndTime.
    auto resolveEffectRange = [&](double perStart, double perEnd,
                                   double& outStart, double& outEnd) {
        if (perEnd <= 0.0) {
            outStart = m_effects.effectStartTime;
            outEnd   = m_effects.effectEndTime;
        } else {
            outStart = perStart;
            outEnd   = perEnd;
        }
    };

    double flashStart, flashEnd, zoomStart, zoomEnd;
    resolveEffectRange(m_effects.beatFlashStartTime, m_effects.beatFlashEndTime, flashStart, flashEnd);
    resolveEffectRange(m_effects.beatZoomStartTime,  m_effects.beatZoomEndTime,  zoomStart,  zoomEnd);

    // Filter beat times by divisor (using original beat index) and per-effect region.
    // Two separate lists: flashBeats gates the flash filter, zoomBeats gates the zoom filter.
    bool hasOriginalIndices = (m_effects.originalBeatIndices.size() == m_effects.beatTimesInOutput.size());

    // Debug: Pre-filtering log
    FILE* preLog = fopen((getTempDir() + "beatsync_ffmpeg_concat.log").c_str(), "a");
    if (preLog) {
        fprintf(preLog, "\n--- Divisor Filter Debug ---\n");
        fprintf(preLog, "hasOriginalIndices=%d, beatTimesInOutput.size=%zu, originalBeatIndices.size=%zu\n",
                hasOriginalIndices ? 1 : 0, m_effects.beatTimesInOutput.size(), m_effects.originalBeatIndices.size());
        fprintf(preLog, "effectBeatDivisor=%d\n", m_effects.effectBeatDivisor);
        fprintf(preLog, "flashRange=[%.3f, %.3f], zoomRange=[%.3f, %.3f]\n",
                flashStart, flashEnd, zoomStart, zoomEnd);
    }

    std::vector<double> flashBeats;
    std::vector<double> zoomBeats;

    int skippedByDivisor = 0;

    for (size_t i = 0; i < m_effects.beatTimesInOutput.size(); ++i) {
        // Apply beat divisor using ORIGINAL beat index (not filtered array index)
        // This ensures "every 2nd beat" actually means every 2nd musical beat
        if (m_effects.effectBeatDivisor > 1) {
            size_t origIdx = hasOriginalIndices ? m_effects.originalBeatIndices[i] : i;
            if ((origIdx % m_effects.effectBeatDivisor) != 0) {
                skippedByDivisor++;
                if (preLog && i < 10) {
                    fprintf(preLog, "  Beat %zu: origIdx=%zu, %zu%%%d=%zu -> SKIP\n",
                            i, origIdx, origIdx, m_effects.effectBeatDivisor, origIdx % m_effects.effectBeatDivisor);
                }
                continue;
            } else if (preLog && i < 20) {
                fprintf(preLog, "  Beat %zu: origIdx=%zu, %zu%%%d=%zu -> PASS divisor\n",
                        i, origIdx, origIdx, m_effects.effectBeatDivisor, origIdx % m_effects.effectBeatDivisor);
            }
        }
        double bt = m_effects.beatTimesInOutput[i];

        // Flash: include beat only if it falls within the flash range
        if (m_effects.enableBeatFlash) {
            bool afterStart = (flashStart <= 0.0) || (bt >= flashStart);
            bool beforeEnd  = (flashEnd  <= 0.0) || (bt <= flashEnd);
            if (afterStart && beforeEnd) {
                flashBeats.push_back(bt);
            }
        }

        // Zoom: include beat only if it falls within the zoom range
        if (m_effects.enableBeatZoom) {
            bool afterStart = (zoomStart <= 0.0) || (bt >= zoomStart);
            bool beforeEnd  = (zoomEnd  <= 0.0) || (bt <= zoomEnd);
            if (afterStart && beforeEnd) {
                zoomBeats.push_back(bt);
            }
        }
    }

    if (preLog) {
        fprintf(preLog, "Result: skippedByDivisor=%d, flashBeats=%zu, zoomBeats=%zu\n",
                skippedByDivisor, flashBeats.size(), zoomBeats.size());
        fclose(preLog);
    }

    // Debug: Log beat times being used for effects
    {
        FILE* debugLog = fopen((getTempDir() + "beatsync_ffmpeg_concat.log").c_str(), "a");
        if (debugLog) {
            fprintf(debugLog, "\n--- Effects Debug ---\n");
            fprintf(debugLog, "Original beats: %zu, flashBeats: %zu (range=%.2f-%.2f), zoomBeats: %zu (range=%.2f-%.2f)\n",
                    m_effects.beatTimesInOutput.size(),
                    flashBeats.size(), flashStart, flashEnd,
                    zoomBeats.size(), zoomStart, zoomEnd);

            // Log original indices for first 20 beats
            fprintf(debugLog, "Original indices (first 20): ");
            for (size_t i = 0; i < m_effects.originalBeatIndices.size() && i < 20; ++i) {
                fprintf(debugLog, "%zu ", m_effects.originalBeatIndices[i]);
            }
            fprintf(debugLog, "\n");

            fprintf(debugLog, "Flash beat times (first 20):\n");
            for (size_t i = 0; i < flashBeats.size() && i < 20; ++i) {
                fprintf(debugLog, "  Beat %zu: %.3f sec\n", i, flashBeats[i]);
            }
            fprintf(debugLog, "BPM: %.2f, enableBeatFlash: %d (intensity=%.2f), enableBeatZoom: %d (intensity=%.2f)\n",
                    m_effects.bpm, m_effects.enableBeatFlash, m_effects.flashIntensity,
                    m_effects.enableBeatZoom, m_effects.zoomIntensity);
            fclose(debugLog);
        }
    }

    // If no effects enabled, just copy
    if (filterChain.empty() && !m_effects.enableBeatFlash && !m_effects.enableBeatZoom) {
        // Simple copy
        std::string ffmpegPath = getFFmpegPath();
        std::ostringstream cmd;
        cmd << "\"" << ffmpegPath << "\" -nostdin -i \"" << inputVideo << "\""
            << " -c copy -y \"" << outputVideo << "\"";

        std::string ffmpegOutput;
        int exitCode;
#ifdef _WIN32
        exitCode = runHiddenCommand(cmd.str(), ffmpegOutput);
#else
        std::string fullCmd = cmd.str() + " 2>&1";
        FILE* pipe = popen_compat(fullCmd.c_str(), "r");
        if (!pipe) {
            m_lastError = "Failed to execute FFmpeg for effects copy";
            return false;
        }
        char buffer[256];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            ffmpegOutput += buffer;
        }
        exitCode = pclose_compat(pipe);
#endif
        return exitCode == 0;
    }

    // Apply effects with re-encoding
    std::string ffmpegPath = getFFmpegPath();
    std::ostringstream cmd;
    cmd << "\"" << ffmpegPath << "\" -nostdin";
    bool cudaAvailable = hasCudaHwaccel();
    bool scaleCudaAvailable = hasScaleCudaFilter();
    bool nvencAvailable = probeEncoder("h264_nvenc");

    // GPU-accelerated effects pipeline strategy:
    // - Use CUDA decode with hwaccel_output_format cuda to keep frames on GPU
    // - Use scale_cuda and overlay_cuda for zoom effects (keeps frames on GPU)
    // - For brightness (eq filter), we must go through CPU but minimize transfers
    // - Use NVENC for encoding to leverage GPU

    bool useGpuPipeline = cudaAvailable && scaleCudaAvailable && nvencAvailable;
    bool hasZoomEffect = m_effects.enableBeatZoom && !zoomBeats.empty();
    bool hasFlashEffect = m_effects.enableBeatFlash && !flashBeats.empty();

    // Log GPU pipeline decision
    {
        FILE* gpuLog = fopen((getTempDir() + "beatsync_ffmpeg_concat.log").c_str(), "a");
        if (gpuLog) {
            fprintf(gpuLog, "\n--- GPU Pipeline Decision ---\n");
            fprintf(gpuLog, "cudaAvailable=%d, scaleCudaAvailable=%d, nvencAvailable=%d\n",
                    cudaAvailable, scaleCudaAvailable, nvencAvailable);
            fprintf(gpuLog, "useGpuPipeline=%d, hasZoomEffect=%d, hasFlashEffect=%d\n",
                    useGpuPipeline, hasZoomEffect, hasFlashEffect);
            fclose(gpuLog);
        }
    }

    // Add CUDA hardware acceleration for decoding
    if (cudaAvailable) {
        cmd << " -hwaccel cuda -hwaccel_device 0";
        // Keep frames on GPU if we're using GPU pipeline for zoom (no flash, or flash-only with GPU zoom)
        if (useGpuPipeline && hasZoomEffect && !hasFlashEffect) {
            cmd << " -hwaccel_output_format cuda";
        }
    }

    cmd << " -i \"" << inputVideo << "\"";

    // Build video filter
    std::string vf;
    if (!filterChain.empty()) {
        vf = filterChain;
    }

    // Beat effects using chained eq filters with limited beats per filter
    // FFmpeg's expression parser has a limit on expression complexity (~50 terms max).
    // We chain multiple eq filters, each handling a subset of beats.
    static constexpr size_t BEATS_PER_FILTER = 30;  // Safe limit per expression

    // Track if we need to output a label for the zoom filter to consume
    // When prior filters exist and zoom is enabled, we need proper filter graph syntax
    // because FFmpeg filter graphs can only consume an input stream once
    bool zoomWillBeEnabled = hasZoomEffect;
    std::string priorFilterOutput;  // Will be set if prior filters output a label for zoom

    // Beat flash effect using chained eq filters
    // Note: eq filter is CPU-only, but we optimize by:
    // 1. Using CUDA decode (frames downloaded once for eq processing)
    // 2. Using NVENC encode (frames uploaded once after all CPU filters)
    if (hasFlashEffect) {
        double flashDuration = 0.08;
        double intensity = std::max(0.1, std::min(1.0, m_effects.flashIntensity));

        // Split beats into chunks, each handled by a separate eq filter
        for (size_t chunk = 0; chunk * BEATS_PER_FILTER < flashBeats.size(); ++chunk) {
            size_t startIdx = chunk * BEATS_PER_FILTER;
            size_t endIdx = std::min(startIdx + BEATS_PER_FILTER, flashBeats.size());

            // Build enable expression for this chunk using between()
            std::ostringstream enableExpr;
            enableExpr << std::fixed << std::setprecision(6);
            for (size_t i = startIdx; i < endIdx; ++i) {
                if (i > startIdx) enableExpr << "+";
                double bt = flashBeats[i];
                enableExpr << "between(t," << bt << "," << (bt + flashDuration) << ")";
            }

            // eq filter with brightness boost, enabled only during beat windows
            std::ostringstream eqFilter;
            eqFilter << "eq=brightness=" << intensity << ":enable='" << enableExpr.str() << "'";

            if (!vf.empty()) {
                vf = vf + "," + eqFilter.str();
            } else {
                vf = eqFilter.str();
            }
        }

        // If zoom is also enabled, we need to output a label from the prior filters
        // so zoom can consume it (can't use [0:v] twice in a filter graph)
        if (zoomWillBeEnabled) {
            // Wrap existing filters with input/output labels for filter graph syntax
            // [0:v]colorbalance...,eq...[prior_out]
            vf = "[0:v]" + vf + "[prior_out]";
            priorFilterOutput = "[prior_out]";
        }
    }

    // If there are prior filters (filterChain/colorbalance) but no flash, and zoom is enabled,
    // we still need to add labels so zoom can properly chain
    if (priorFilterOutput.empty() && !vf.empty() && zoomWillBeEnabled) {
        vf = "[0:v]" + vf + "[prior_out]";
        priorFilterOutput = "[prior_out]";
    }

    // Beat zoom effect
    // Strategy: Use GPU-accelerated filters when available and no flash effect
    // - scale_cuda + overlay_cuda for pure GPU pipeline
    // - CPU scale + overlay when flash effect is present (frames already on CPU)
    if (hasZoomEffect) {
        double zoomDuration = 0.15;
        double zoomAmount = std::max(0.01, std::min(0.15, m_effects.zoomIntensity));
        // Scale factor: 1.0 + zoomAmount means we scale up then crop back
        double scaleFactor = 1.0 + zoomAmount;
        int scaledW = static_cast<int>(m_outputWidth * scaleFactor);
        int scaledH = static_cast<int>(m_outputHeight * scaleFactor);
        int cropX = (scaledW - m_outputWidth) / 2;
        int cropY = (scaledH - m_outputHeight) / 2;

        std::string prevOutput;  // Track previous chunk's output label
        // Set prevOutput to priorFilterOutput if any prior filters exist
        prevOutput = priorFilterOutput;

        // Calculate total number of chunks upfront so we know which is the last
        size_t totalChunks = (zoomBeats.size() + BEATS_PER_FILTER - 1) / BEATS_PER_FILTER;

        // Determine if we can use GPU zoom filters
        // We can use GPU zoom if:
        // 1. GPU pipeline is available (scale_cuda, overlay_cuda)
        // 2. No flash effect (which forces CPU processing)
        // 3. No prior CPU filters in the chain
        bool useGpuZoom = useGpuPipeline && !hasFlashEffect && filterChain.empty();

        // Log zoom strategy
        {
            FILE* zoomLog = fopen((getTempDir() + "beatsync_ffmpeg_concat.log").c_str(), "a");
            if (zoomLog) {
                fprintf(zoomLog, "Zoom strategy: useGpuZoom=%d (gpuPipeline=%d, noFlash=%d, noFilterChain=%d)\n",
                        useGpuZoom, useGpuPipeline, !hasFlashEffect, filterChain.empty());
                fclose(zoomLog);
            }
        }

        // Split beats into chunks
        for (size_t chunk = 0; chunk * BEATS_PER_FILTER < zoomBeats.size(); ++chunk) {
            size_t startIdx = chunk * BEATS_PER_FILTER;
            size_t endIdx = std::min(startIdx + BEATS_PER_FILTER, zoomBeats.size());
            bool isLastChunk = (chunk == totalChunks - 1);

            // Build enable expression for this chunk
            std::ostringstream enableExpr;
            enableExpr << std::fixed << std::setprecision(6);
            for (size_t i = startIdx; i < endIdx; ++i) {
                if (i > startIdx) enableExpr << "+";
                double bt = zoomBeats[i];
                enableExpr << "between(t," << bt << "," << (bt + zoomDuration) << ")";
            }

            // Determine input: first chunk uses prevOutput (set to priorFilterOutput if any prior filters), otherwise [0:v]
            std::string inputPad;
            if (chunk == 0) {
                inputPad = !prevOutput.empty() ? prevOutput : "[0:v]";
            } else {
                inputPad = prevOutput;
            }
            // Only add output label if not the last chunk - last chunk outputs directly
            // so FFmpeg uses it as the default video output
            std::string outputLabel = isLastChunk ? "" : "[zoom_out" + std::to_string(chunk) + "]";

            std::ostringstream zoomFilter;

            if (useGpuZoom) {
                // GPU-accelerated zoom using scale_cuda and overlay_cuda
                // Note: overlay_cuda doesn't support enable expressions, so we use a different approach:
                // We'll use the standard CPU overlay with enable, but use scale_cuda for the scaling
                // This still provides benefit since scale is the most compute-intensive part

                // For now, use hwdownload before overlay since overlay_cuda doesn't support enable
                // This is still faster than pure CPU because scale_cuda is much faster
                zoomFilter << inputPad << "split[zoom_main" << chunk << "][zoom_in" << chunk << "];"
                          << "[zoom_in" << chunk << "]scale_cuda=" << scaledW << ":" << scaledH
                          << ",hwdownload,format=nv12"
                          << ",crop=" << m_outputWidth << ":" << m_outputHeight << ":" << cropX << ":" << cropY
                          << "[zoom_scaled" << chunk << "];"
                          << "[zoom_main" << chunk << "]hwdownload,format=nv12[zoom_main_cpu" << chunk << "];"
                          << "[zoom_main_cpu" << chunk << "][zoom_scaled" << chunk << "]overlay=enable='" << enableExpr.str() << "'"
                          << outputLabel;
            } else {
                // CPU zoom (when flash effect is present or GPU not available)
                zoomFilter << inputPad << "split[zoom_main" << chunk << "][zoom_in" << chunk << "];"
                          << "[zoom_in" << chunk << "]scale=" << scaledW << ":" << scaledH
                          << ",crop=" << m_outputWidth << ":" << m_outputHeight << ":" << cropX << ":" << cropY
                          << "[zoom_scaled" << chunk << "];"
                          << "[zoom_main" << chunk << "][zoom_scaled" << chunk << "]overlay=enable='" << enableExpr.str() << "'"
                          << outputLabel;
            }

            if (chunk == 0 && !vf.empty()) {
                // First zoom chunk but have prior filters - connect them
                vf = vf + ";" + zoomFilter.str();
            } else if (chunk == 0) {
                vf = zoomFilter.str();
            } else {
                vf = vf + ";" + zoomFilter.str();
            }

            prevOutput = outputLabel;  // Save for next iteration (empty for last chunk)
        }
    }

    // Use filter_script file if the filter is too long for command line (Windows limit ~8191 chars)
    std::string filterScriptPath;
    bool useFilterScript = vf.length() > 6000;  // Leave margin for rest of command
    
    if (!vf.empty()) {
        if (useFilterScript) {
            // Write filter to a temporary file
            filterScriptPath = getTempDir() + "beatsync_filter.txt";
            FILE* scriptFile = fopen(filterScriptPath.c_str(), "w");
            if (scriptFile) {
                fprintf(scriptFile, "%s", vf.c_str());
                fclose(scriptFile);
                cmd << " -filter_complex_script \"" << filterScriptPath << "\"";
            } else {
                // Fallback to command line if file write fails
                cmd << " -filter_complex \"" << vf << "\"";
                useFilterScript = false;
            }
        } else {
            cmd << " -filter_complex \"" << vf << "\"";
        }
    }

    cmd << " " << getEncoderArgs("fast")
        << " -c:a copy"
        << " -y \"" << outputVideo << "\"";

    std::cout << "Applying effects...\n";

    std::string ffmpegOutput;
    int exitCode;
#ifdef _WIN32
    exitCode = runHiddenCommand(cmd.str(), ffmpegOutput, m_cancelFlag);
    if (exitCode == -2) {
        m_lastError = "Cancelled by user";
        return false;
    }
#else
    std::string fullCmd = cmd.str() + " 2>&1";
    FILE* pipe = popen_compat(fullCmd.c_str(), "r");
    if (!pipe) {
        m_lastError = "Failed to execute FFmpeg for effects";
        return false;
    }
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        ffmpegOutput += buffer;
    }
    exitCode = pclose_compat(pipe);
#endif

    // Clean up temp files
    if (useFilterScript && !filterScriptPath.empty()) {
        std::remove(filterScriptPath.c_str());
    }

    // Log the effects command output
    {
        FILE* logf = fopen((getTempDir() + "beatsync_ffmpeg_concat.log").c_str(), "a");
        if (logf) {
            fprintf(logf, "\n--- FFmpeg effects run ---\n");
            fprintf(logf, "Using filter_script: %s\n", useFilterScript ? "yes" : "no");
            fprintf(logf, "Filter length: %zu chars\n", vf.length());
            fprintf(logf, "cmd: %s\nexit: %d\noutput:\n%s\n", cmd.str().c_str(), exitCode, ffmpegOutput.c_str());
            if (useFilterScript) {
                fprintf(logf, "Filter content (first 500 chars): %.500s...\n", vf.c_str());
            }
            fclose(logf);
        }
    }

    if (exitCode != 0) {
        m_lastError = "FFmpeg effects processing failed: " + ffmpegOutput.substr(0, 200);
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
