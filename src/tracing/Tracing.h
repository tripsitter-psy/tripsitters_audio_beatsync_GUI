#pragma once

#include <string>
#include <chrono>
#include <map>

namespace Tracing {

// Initialize tracing subsystem; serviceName will be recorded in spans
void Init(const std::string& serviceName);

// Shutdown tracing subsystem, flush logs
void Shutdown();

// Lightweight RAII init helper
struct ScopedInit {
    explicit ScopedInit(const std::string& serviceName) { Init(serviceName); }
    ~ScopedInit() { Shutdown(); }
};

// Simple span object - records start time and logs on destruction
class Span {
public:
    explicit Span(const std::string& name);
    ~Span();

    void addAttribute(const std::string& key, const std::string& value);

private:
    std::string m_name;
    std::chrono::steady_clock::time_point m_start;
    std::map<std::string, std::string> m_attrs;
};

// Convenience macro to create a scoped span
#define TRACED_SCOPE(name) Tracing::Span ANON_SPAN(__COUNTER__)(name)
// Helper to give unique local name
#define ANON_SPAN(x) span_##x

} // namespace Tracing
