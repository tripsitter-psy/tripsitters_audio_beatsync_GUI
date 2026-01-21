#include "tracing/Tracing.h"
#include <chrono>
#include <fstream>
#include <iostream>

#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <iomanip>
#include <sstream>
#include <filesystem>

namespace BeatSync {
namespace tracing {

struct Span::Impl {
    std::string name;
    std::chrono::steady_clock::time_point start;
    bool ended = false;
};



static std::mutex g_mutex;
static std::unique_ptr<std::ofstream> g_out;
static std::string g_outfile_path;

enum class TracingFlushMode { Never, Periodic, Shutdown };
static std::atomic<TracingFlushMode> g_flush_mode{TracingFlushMode::Shutdown};
static std::thread g_flush_thread;
static std::atomic<bool> g_flush_thread_running{false};
static std::condition_variable g_flush_cv;
static std::mutex g_flush_cv_mutex;
    static std::atomic<int> g_flush_period_ms{1000}; // Default 1s for periodic

static std::mutex g_flush_start_mutex;
void SetTracingFlushMode(TracingFlushMode mode, int period_ms = 1000) {
    g_flush_mode = mode;
    if (mode == TracingFlushMode::Periodic) {
        g_flush_period_ms.store(period_ms, std::memory_order_relaxed);
        std::lock_guard<std::mutex> startlk(g_flush_start_mutex);
        if (!g_flush_thread_running) {
            g_flush_thread_running = true;
            g_flush_thread = std::thread([]() {
                while (g_flush_thread_running) {
                    std::unique_lock<std::mutex> lk(g_flush_cv_mutex);
                    g_flush_cv.wait_for(lk, std::chrono::milliseconds(g_flush_period_ms.load(std::memory_order_relaxed)));
                    if (!g_flush_thread_running) break;
                    std::lock_guard<std::mutex> outlk(g_mutex);
                    if (g_out) g_out->flush();
                }
            });
        }
    } else {
        // Stop background flusher if running
        std::lock_guard<std::mutex> startlk(g_flush_start_mutex);
        if (g_flush_thread_running) {
            g_flush_thread_running = false;
            g_flush_cv.notify_all();
            if (g_flush_thread.joinable()) g_flush_thread.join();
        }
    }
}

static std::string timestamp() {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto t = system_clock::to_time_t(now);
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    std::tm tm_buf;
#if defined(_WIN32)
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S") << "." << std::setw(3) << std::setfill('0') << ms.count();
    return oss.str();
}

void InitTracing(const std::string& outfile) {
    std::unique_lock<std::mutex> lk(g_mutex);
    std::filesystem::path new_path;
    if (!outfile.empty()) new_path = outfile;
    else {
        std::error_code ec;
        std::filesystem::path td = std::filesystem::temp_directory_path(ec);
        if (!ec) {
            new_path = td / "beatsync-trace.log";
        } else {
            std::cerr << "Warning: unable to determine temp directory: " << ec.message() << ". Using current directory for trace file.\n";
            new_path = std::filesystem::path("beatsync-trace.log");
        }
    }


    // If already open, check if we need to switch files
    if (g_out) {
        if (g_outfile_path == new_path.string()) {
            // Already initialized to this file, do nothing
            return;
        } else {
            // Switching to a new file: close and reset
            bool need_stop_flusher = (g_flush_mode == TracingFlushMode::Periodic);
            std::unique_ptr<std::ofstream> local_out;
            if (need_stop_flusher) {
                lk.unlock();
                SetTracingFlushMode(TracingFlushMode::Never); // Stop flusher
                lk.lock();
            }
            local_out = std::move(g_out);
            g_outfile_path.clear();
            // Release lock before flushing/closing
            lk.unlock();
            if (local_out) {
                local_out->flush();
                local_out->close();
                local_out.reset();
            }
            lk.lock();
        }
    }

    g_out = std::make_unique<std::ofstream>(new_path.string(), std::ios::app);
    if (!g_out->is_open()) {
        std::cerr << "Warning: could not open tracing file: " << new_path.string() << "\n";
        g_out.reset();
        g_outfile_path.clear();
    } else {
        g_outfile_path = new_path.string();
        // Optionally start flusher if mode is periodic
        if (g_flush_mode == TracingFlushMode::Periodic) {
            SetTracingFlushMode(TracingFlushMode::Periodic, g_flush_period_ms);
        }
    }
}

void ShutdownTracing() {
    // Stop background flusher if running
    SetTracingFlushMode(TracingFlushMode::Never);
    std::lock_guard<std::mutex> lk(g_mutex);
    if (g_out) {
        g_out->flush();
        g_out->close();
        g_out.reset();
    }
    // Also clear the cached outfile path so re-initialization behaves correctly
    g_outfile_path.clear();
}


Span::Span(const char* name) : impl_(std::make_unique<Impl>()) {
    impl_->name = name ? name : "";
    impl_->start = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(g_mutex);
    if (g_out) {
        (*g_out) << timestamp() << " START " << impl_->name << " thread=" << std::this_thread::get_id() << "\n";
        // No per-span flush; handled by flusher or shutdown
    }
}

Span::~Span() {
    End();
}


void Span::End() {
    if (!impl_ || impl_->ended) return;
    impl_->ended = true;
    auto end = std::chrono::steady_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - impl_->start).count();
    std::lock_guard<std::mutex> lk(g_mutex);
    if (g_out) {
        (*g_out) << timestamp() << " END   " << impl_->name << " thread=" << std::this_thread::get_id() << " duration=" << dur << "ms\n";
        // No per-span flush; handled by flusher or shutdown
    }
}

// Optional: C API for flush mode
extern "C" void SetTracingFlushModeC(int mode, int period_ms) {
    // 0 = Never, 1 = Periodic, 2 = Shutdown
    TracingFlushMode safeMode = TracingFlushMode::Never;
    if (mode >= 0 && mode <= 2) {
        safeMode = static_cast<TracingFlushMode>(mode);
    } else {
        fprintf(stderr, "Invalid flush mode: %d, using Never.\n", mode);
    }
    SetTracingFlushMode(safeMode, period_ms);
}

} // namespace tracing
} // namespace BeatSync
