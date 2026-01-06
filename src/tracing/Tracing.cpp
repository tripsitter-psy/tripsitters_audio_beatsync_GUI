#include "Tracing.h"
#include <fstream>
#include <mutex>
#include <thread>
#include <iomanip>
#include <sstream>
#include <cstdio>

namespace Tracing {

static std::mutex s_mutex;
static std::ofstream s_out;
static std::string s_serviceName = "";

void Init(const std::string& serviceName) {
    std::lock_guard<std::mutex> lk(s_mutex);
    if (s_out.is_open()) return; // already initialized
    s_serviceName = serviceName;
    s_out.open("traces.jsonl", std::ios::app);
    if (!s_out.is_open()) {
        // Fall back to stderr if file can't be opened
        fprintf(stderr, "Tracing: failed to open traces.jsonl for append\n");
    }
}

void Shutdown() {
    std::lock_guard<std::mutex> lk(s_mutex);
    if (s_out.is_open()) {
        s_out.flush();
        s_out.close();
    }
}

Span::Span(const std::string& name)
    : m_name(name), m_start(std::chrono::steady_clock::now()) {
}

Span::~Span() {
    auto end = std::chrono::steady_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - m_start).count();

    std::ostringstream ss;
    ss << std::fixed;
    ss << "{\"service\":\"" << s_serviceName << "\",";
    ss << "\"span\":\"" << m_name << "\",";
    ss << "\"thread\":\"" << std::this_thread::get_id() << "\",";
    ss << "\"duration_ms\":" << dur << ",";
    ss << "\"timestamp_ms\":" << std::chrono::duration_cast<std::chrono::milliseconds>(m_start.time_since_epoch()).count() << ",";

    // attrs
    ss << "\"attrs\":{";
    bool first = true;
    for (const auto& kv : m_attrs) {
        if (!first) ss << ",";
        first = false;
        ss << "\"" << kv.first << "\":\"" << kv.second << "\"";
    }
    ss << "}}\n";

    std::lock_guard<std::mutex> lk(s_mutex);
    if (s_out.is_open()) {
        s_out << ss.str();
        s_out.flush();
    } else {
        fprintf(stderr, "%s", ss.str().c_str());
    }
}

void Span::addAttribute(const std::string& key, const std::string& value) {
    m_attrs[key] = value;
}

} // namespace Tracing
