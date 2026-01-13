#include "tracing/Tracing.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>
#include <iomanip>
#include <sstream>
#include <filesystem>

namespace BeatSync {
namespace tracing {

struct Span::Impl {
    std::string name;
    std::chrono::steady_clock::time_point start;
    bool ended = false;
    bool ended = false;
};


static std::mutex g_mutex;
static std::unique_ptr<std::ofstream> g_out;
static std::string g_outfile_path;

static std::string timestamp() {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto t = system_clock::to_time_t(now);
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M:%S") << "." << std::setw(3) << std::setfill('0') << ms.count();
    return oss.str();
}

void InitTracing(const std::string& outfile) {
    std::lock_guard<std::mutex> lk(g_mutex);
    std::filesystem::path new_path;
    if (!outfile.empty()) new_path = outfile;
    else new_path = std::filesystem::temp_directory_path() / "beatsync-trace.log";

    // If already open, check if we need to switch files
    if (g_out) {
        if (g_outfile_path == new_path.string()) {
            // Already initialized to this file, do nothing
            return;
        } else {
            // Switching to a new file: close and reset
            g_out->flush();
            g_out->close();
            g_out.reset();
            g_outfile_path.clear();
        }
    }

    g_out = std::make_unique<std::ofstream>(new_path.string(), std::ios::app);
    if (!g_out->is_open()) {
        std::cerr << "Warning: could not open tracing file: " << new_path.string() << "\n";
        g_out.reset();
        g_outfile_path.clear();
    } else {
        g_outfile_path = new_path.string();
    }
}

void ShutdownTracing() {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (g_out) {
        g_out->flush();
        g_out->close();
        g_out.reset();
    }
}

Span::Span(const char* name) : impl_(new Impl()) {
    impl_->name = name ? name : "";
    impl_->start = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(g_mutex);
    if (g_out) {
        (*g_out) << timestamp() << " START " << impl_->name << " thread=" << std::this_thread::get_id() << "\n";
        g_out->flush();
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
        g_out->flush();
    }
}

} // namespace tracing
} // namespace BeatSync
