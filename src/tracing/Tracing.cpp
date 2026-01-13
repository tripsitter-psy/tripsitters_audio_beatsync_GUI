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
    std::ofstream* out;
    bool ended = false;
};

static std::mutex g_mutex;
static std::unique_ptr<std::ofstream> g_out;

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
    if (!g_out) {
        std::filesystem::path path;
        if (!outfile.empty()) path = outfile;
        else {
            path = std::filesystem::temp_directory_path() / "beatsync-trace.log";
        }
        g_out = std::make_unique<std::ofstream>(path.string(), std::ios::app);
        if (!g_out->is_open()) {
            std::cerr << "Warning: could not open tracing file: " << path.string() << "\n";
            g_out.reset();
        }
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
    impl_->out = nullptr;
    std::lock_guard<std::mutex> lk(g_mutex);
    if (g_out) impl_->out = g_out.get();
    if (impl_->out) {
        (*impl_->out) << timestamp() << " START " << impl_->name << " thread=" << std::this_thread::get_id() << "\n";
        impl_->out->flush();
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
    if (impl_->out) {
        (*impl_->out) << timestamp() << " END " << impl_->name << " dur_ms=" << dur << " thread=" << std::this_thread::get_id() << "\n";
        impl_->out->flush();
    }
}

} // namespace tracing
} // namespace BeatSync
