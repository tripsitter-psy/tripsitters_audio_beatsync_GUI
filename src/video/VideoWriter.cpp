#include "VideoWriter.h"
#include "TransitionLibrary.h"
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

// libavformat for audio stream probing
extern "C" {
#include <libavformat/avformat.h>
}

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
static int runHiddenCommand(const std::string& cmdLine, std::string& output) {
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

    // Read output in chunks
    char buf[4096];
    DWORD bytesRead;
    while (ReadFile(hReadPipe, buf, sizeof(buf) - 1, &bytesRead, NULL) && bytesRead > 0) {
        buf[bytesRead] = '\0';
        output += buf;
    }
    CloseHandle(hReadPipe);

    // Wait for process to complete (with timeout to prevent hangs)
    DWORD waitResult = WaitForSingleObject(pi.hProcess, 300000);  // 5 minute timeout

    DWORD exitCode = 0;
    if (waitResult == WAIT_TIMEOUT) {
        TerminateProcess(pi.hProcess, 1);
        exitCode = 1;
        output += "\nError: Process timed out after 5 minutes";
    } else {
        GetExitCodeProcess(pi.hProcess, &exitCode);
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
    // Fallback to a safe default
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
    // On Linux/macOS, brief sleep is the best we can do without root access
    usleep(150000);  // 150ms
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
    // 1. Check environment variable first
    const char* envPath = std::getenv("BEATSYNC_FFMPEG_PATH");
    if (envPath != nullptr && envPath[0] != '\0') {
        return envPath;
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
        // If we found something, use it
        if (!result.empty() && result.find("ffmpeg") != std::string::npos) {
            return result;
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

        // If we found something, use it
        if (!result.empty() && result.find("ffmpeg") != std::string::npos) {
            return result;
        }
    }
#endif

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
    cmd << "\"" << ffmpegPath << "\"";

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
    static int segmentNum = 0;
    segmentNum++;
    std::cout << "[GPU DEBUG] Segment " << segmentNum << ": allowGpu=" << allowGpuThisSegment
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
    exitCode = runHiddenCommand(cmd.str(), ffmpegOutput);
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
                                     const std::string& outputVideo) {
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
    cmd << "\"" << ffmpegPath << "\"";

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
            << ":(ow-iw)/2:(oh-ih)/2,setsar=1,fps=" << m_outputFps << "\"";
    } else {
        // CPU filter chain (original behavior)
        cmd << " -vf \"scale=" << m_outputWidth << ":" << m_outputHeight
            << ":force_original_aspect_ratio=decrease,pad=" << m_outputWidth << ":" << m_outputHeight
            << ":(ow-iw)/2:(oh-ih)/2,setsar=1,fps=" << m_outputFps << "\"";
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
    exitCode = runHiddenCommand(cmd.str(), ffmpegOutput);
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

    // File-based diagnostic logging
    FILE* diagLog = fopen((getTempDir() + "beatsync_normalize_detail.log").c_str(), "a");
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
    cmd << "\"" << ffmpegPath << "\"";

    // Check GPU availability for hardware-accelerated normalization
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
    exitCode = runHiddenCommand(cmd.str(), ffmpegOutput);
#else
    std::string fullCmd = cmd.str() + " 2>&1";
    FILE* pipe = popen_compat(fullCmd.c_str(), "r");
    if (!pipe) {
        m_lastError = "Failed to execute FFmpeg for video normalization";
        if (diagLog) { fprintf(diagLog, "  ERROR: popen failed\n"); fclose(diagLog); }
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
                if (diagLog) { fprintf(diagLog, "  File created despite error: %ld bytes\n", fileSize); fclose(diagLog); }
                return true;
            }
        }
        m_lastError = "Video normalization failed: " + ffmpegOutput.substr(0, 200);
        if (diagLog) { fprintf(diagLog, "  FAILED: %s\n", m_lastError.c_str()); fclose(diagLog); }
        return false;
    }

    // Verify file was created
    FILE* verify = fopen(outputVideo.c_str(), "rb");
    if (verify) {
        fseek(verify, 0, SEEK_END);
        long fileSize = ftell(verify);
        fclose(verify);
        if (diagLog) { fprintf(diagLog, "  SUCCESS: Output file %ld bytes\n", fileSize); fclose(diagLog); }
    } else {
        if (diagLog) { fprintf(diagLog, "  WARNING: Output file not found after success exit code!\n"); fclose(diagLog); }
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

        if (!normalizeVideo(video, normalizedPath)) {
            // Clean up any already-created normalized files
            for (const auto& path : normalizedPaths) {
                std::remove(path.c_str());
            }
            normalizedPaths.clear();
            return false;
        }

        normalizedPaths.push_back(normalizedPath);

        // Report progress
        if (m_progressCallback) {
            reportProgress(static_cast<double>(index) / inputVideos.size() * 0.1);  // 10% for normalization
        }
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

    FILE* debugLog = fopen((getTempDir() + "tripsitter_debug.log").c_str(), "a");
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

            debugLog = fopen((getTempDir() + "tripsitter_debug.log").c_str(), "a");
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

            debugLog = fopen((getTempDir() + "tripsitter_debug.log").c_str(), "a");
            if (debugLog) {
                fprintf(debugLog, "  - %s [OK, %ld bytes]\n", video.c_str(), size);
                fclose(debugLog);
            }
        }
    }
    fclose(f);

    if (missingCount > 0) {
        m_lastError = "Cannot concatenate: " + std::to_string(missingCount) + " segment files are missing!";

        debugLog = fopen((getTempDir() + "tripsitter_debug.log").c_str(), "a");
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
                    cmd << "\"" << ffmpegPath << "\"";
                    for (const auto &v : inputVideos) {
                        cmd << " -i \"" << v << "\"";
                    }

                    // Build audio portion of filter_complex
                    // For audio, we'll use a simpler approach: just use anullsrc to avoid probing hundreds of files
                    // This is more reliable for long sequences and avoids memory pressure from libavformat probing
                    std::ostringstream audioFilter;

                    // Only probe first file to determine if source has audio
                    bool sourceHasAudio = false;
                    if (!inputVideos.empty()) {
                        AVFormatContext* probeCtx = nullptr;
                        int openErr = avformat_open_input(&probeCtx, inputVideos[0].c_str(), nullptr, nullptr);
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
    cmd << "\"" << ffmpegPath << "\" -fflags +genpts+igndts -f concat -safe 0 -i \"" << listFile
        << "\" -c copy -video_track_timescale 90000 -y \"" << outputVideo << "\"";

    // Execute FFmpeg hidden (no console flash on Windows)
    std::string ffmpegOutput;
    int exitCode;
#ifdef _WIN32
    exitCode = runHiddenCommand(cmd.str(), ffmpegOutput);
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
        reencodeCmd << "\"" << ffmpegPath << "\"";

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

    // Combine video from first input with audio from second input
    // -c:v copy = stream copy video (fast, no re-encode)
    // -c:a aac = encode audio as AAC
    // -shortest = trim output to shorter of video/audio (optional)
    double clipStart = std::max(0.0, audioStart);
    bool clipAudio = (audioEnd > 0.0 && audioEnd > audioStart + 1e-3);
    double clipDur = clipAudio ? (audioEnd - audioStart) : 0.0;

    cmd << "\"" << ffmpegPath << "\" -i \"" << inputVideo << "\"";
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

// Cache all available encoders in a single FFmpeg call (thread-safe init)
static std::set<std::string> getAvailableEncoders(const std::string& ffmpegPath) {
    static std::set<std::string> cachedEncoders;
    static std::once_flag encodersOnce;

    std::call_once(encodersOnce, [ffmpegPath]() {
        std::string output;
        std::string cmd = "\"" + ffmpegPath + "\" -hide_banner -encoders";

#ifdef _WIN32
        int rc = runHiddenCommand(cmd, output);
        if (rc != 0) return;
#else
        std::string fullCmd = cmd + " 2>&1";
        FILE* pipe = popen_compat(fullCmd.c_str(), "r");
        if (!pipe) return; // leave cachedEncoders empty
        char buffer[512];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            output += buffer;
        }
        int rc = pclose_compat(pipe);
        if (rc != 0) return;
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
                cachedEncoders.insert(encoderName);
            }
        }
    });

    return cachedEncoders;
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

    // Filter beat times by divisor (using original beat index) and region
    std::vector<double> filteredBeats;
    bool hasOriginalIndices = (m_effects.originalBeatIndices.size() == m_effects.beatTimesInOutput.size());
    
    // Debug: Pre-filtering log
    FILE* preLog = fopen((getTempDir() + "beatsync_ffmpeg_concat.log").c_str(), "a");
    if (preLog) {
        fprintf(preLog, "\n--- Divisor Filter Debug ---\n");
        fprintf(preLog, "hasOriginalIndices=%d, beatTimesInOutput.size=%zu, originalBeatIndices.size=%zu\n",
                hasOriginalIndices ? 1 : 0, m_effects.beatTimesInOutput.size(), m_effects.originalBeatIndices.size());
        fprintf(preLog, "effectBeatDivisor=%d\n", m_effects.effectBeatDivisor);
    }
    
    int skippedByDivisor = 0, skippedByRegion = 0, included = 0;
    
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
        // Apply effect region filter
        if (m_effects.effectStartTime > 0 && bt < m_effects.effectStartTime) {
            skippedByRegion++;
            continue;
        }
        if (m_effects.effectEndTime > 0 && bt > m_effects.effectEndTime) {
            skippedByRegion++;
            continue;
        }
        included++;
        filteredBeats.push_back(bt);
    }
    
    if (preLog) {
        fprintf(preLog, "Result: skippedByDivisor=%d, skippedByRegion=%d, included=%d\n", 
                skippedByDivisor, skippedByRegion, included);
        fclose(preLog);
    }

    // Debug: Log beat times being used for effects
    {
        FILE* debugLog = fopen((getTempDir() + "beatsync_ffmpeg_concat.log").c_str(), "a");
        if (debugLog) {
            fprintf(debugLog, "\n--- Effects Debug ---\n");
            fprintf(debugLog, "Original beats: %zu, After filter: %zu (divisor=%d, region=%.2f-%.2f, hasOrigIdx=%d)\n", 
                    m_effects.beatTimesInOutput.size(), filteredBeats.size(),
                    m_effects.effectBeatDivisor, m_effects.effectStartTime, m_effects.effectEndTime,
                    hasOriginalIndices ? 1 : 0);
            
            // Log original indices for first 20 beats
            fprintf(debugLog, "Original indices (first 20): ");
            for (size_t i = 0; i < m_effects.originalBeatIndices.size() && i < 20; ++i) {
                fprintf(debugLog, "%zu ", m_effects.originalBeatIndices[i]);
            }
            fprintf(debugLog, "\n");
            
            fprintf(debugLog, "Filtered beat times:\n");
            for (size_t i = 0; i < filteredBeats.size() && i < 20; ++i) {
                fprintf(debugLog, "  Beat %zu: %.3f sec\n", i, filteredBeats[i]);
            }
            if (filteredBeats.size() > 20) {
                fprintf(debugLog, "  ... and %zu more\n", filteredBeats.size() - 20);
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
        cmd << "\"" << ffmpegPath << "\" -i \"" << inputVideo << "\""
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
    cmd << "\"" << ffmpegPath << "\"";

    // Check GPU capabilities for optimal pipeline selection
    bool cudaAvailable = hasCudaHwaccel();
    bool scaleCudaAvailable = hasScaleCudaFilter();
    bool nvencAvailable = probeEncoder("h264_nvenc");

    // GPU-accelerated effects pipeline strategy:
    // - Use CUDA decode with hwaccel_output_format cuda to keep frames on GPU
    // - Use scale_cuda and overlay_cuda for zoom effects (keeps frames on GPU)
    // - For brightness (eq filter), we must go through CPU but minimize transfers
    // - Use NVENC for encoding to leverage GPU

    bool useGpuPipeline = cudaAvailable && scaleCudaAvailable && nvencAvailable;
    bool hasZoomEffect = m_effects.enableBeatZoom && !filteredBeats.empty();
    bool hasFlashEffect = m_effects.enableBeatFlash && !filteredBeats.empty();

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

    std::string sendcmdPath;  // Placeholder for cleanup compatibility

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
        for (size_t chunk = 0; chunk * BEATS_PER_FILTER < filteredBeats.size(); ++chunk) {
            size_t startIdx = chunk * BEATS_PER_FILTER;
            size_t endIdx = std::min(startIdx + BEATS_PER_FILTER, filteredBeats.size());

            // Build enable expression for this chunk using between()
            std::ostringstream enableExpr;
            enableExpr << std::fixed << std::setprecision(6);
            for (size_t i = startIdx; i < endIdx; ++i) {
                if (i > startIdx) enableExpr << "+";
                double bt = filteredBeats[i];
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
        size_t totalChunks = (filteredBeats.size() + BEATS_PER_FILTER - 1) / BEATS_PER_FILTER;

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
        for (size_t chunk = 0; chunk * BEATS_PER_FILTER < filteredBeats.size(); ++chunk) {
            size_t startIdx = chunk * BEATS_PER_FILTER;
            size_t endIdx = std::min(startIdx + BEATS_PER_FILTER, filteredBeats.size());
            bool isLastChunk = (chunk == totalChunks - 1);

            // Build enable expression for this chunk
            std::ostringstream enableExpr;
            enableExpr << std::fixed << std::setprecision(6);
            for (size_t i = startIdx; i < endIdx; ++i) {
                if (i > startIdx) enableExpr << "+";
                double bt = filteredBeats[i];
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
    exitCode = runHiddenCommand(cmd.str(), ffmpegOutput);
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
