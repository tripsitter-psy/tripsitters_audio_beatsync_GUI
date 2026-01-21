#include "tracing.h"
#include <iostream>
#include <cstdlib>
#include <mutex>

#if defined(BEATSYNC_ENABLE_TRACING)
// If the OpenTelemetry SDK is available, include the headers and set up an OTLP exporter.
// Note: Exact include paths may vary by distribution; leave this guarded so builds don't fail
// when the SDK is missing. The CMake detect above sets BEATSYNC_ENABLE_TRACING when the
// package was found.


#include <opentelemetry/sdk/trace/tracer_provider.h>
#include <opentelemetry/sdk/resource/resource.h>
#include <opentelemetry/trace/provider.h>
#include <opentelemetry/exporters/otlp/otlp_grpc_exporter.h>
#include <opentelemetry/sdk/trace/batch_span_processor.h>
#include <opentelemetry/trace/noop.h>

namespace {
using opentelemetry::nostd::shared_ptr;
namespace sdktrace = opentelemetry::sdk::trace;
static std::shared_ptr<sdktrace::TracerProvider> g_provider;
static std::mutex g_providerMutex;  // Protects g_provider access
}

namespace BeatSync {

bool InitializeTracing(const std::string& serviceName) {
    const char* env = std::getenv("OTEL_EXPORTER_OTLP_ENDPOINT");
    // Use port 4317 for gRPC (OTLP/gRPC default), not 4318 (OTLP/HTTP)
    std::string endpoint = env ? env : "http://localhost:4317";

    std::lock_guard<std::mutex> lock(g_providerMutex);

    // Check if already initialized (only check our module-level pointer, not GetTracerProvider()
    // which always returns a non-null NoopTracerProvider by default)
    if (g_provider) {
        std::clog << "BeatSync: Tracing already initialized, skipping re-initialization.\n";
        return true;
    }

    try {
        // Create OTLP exporter (gRPC) with configured endpoint
        opentelemetry::exporters::otlp::OtlpGrpcExporterOptions options;
        options.endpoint = endpoint;
        auto exporter = std::unique_ptr<opentelemetry::sdk::trace::SpanExporter>(new opentelemetry::exporters::otlp::OtlpGrpcExporter(options));
        auto processor = std::unique_ptr<opentelemetry::sdk::trace::SpanProcessor>(new opentelemetry::sdk::trace::BatchSpanProcessor(std::move(exporter)));

        // Create resource with service name
        auto resource = opentelemetry::sdk::resource::Resource::Create({{"service.name", serviceName}});
        auto provider = std::make_shared<opentelemetry::sdk::trace::TracerProvider>(std::move(processor), resource);

        opentelemetry::trace::Provider::SetTracerProvider(provider);
        g_provider = provider;
        std::clog << "BeatSync: Tracing initialized (OTLP endpoint=" << endpoint << ", service=" << serviceName << ")\n";
        return true;
    } catch (const std::exception& e) {
        std::cerr << "BeatSync: Failed to initialize tracing: " << e.what() << "\n";
        return false;
    }
}

void ShutdownTracing() {
    std::lock_guard<std::mutex> lock(g_providerMutex);

    if (g_provider) {
        // Attempt to cast to SDK provider for Shutdown
        auto sdk_provider = std::dynamic_pointer_cast<opentelemetry::sdk::trace::TracerProvider>(g_provider);
        if (sdk_provider) {
            bool ok = sdk_provider->Shutdown();
            if (!ok) {
                std::cerr << "BeatSync: Warning - tracing shutdown failed (Shutdown() returned false)\n";
            }
        }
        // Set to no-op provider instead of nullptr
        opentelemetry::trace::Provider::SetTracerProvider(std::make_shared<opentelemetry::trace::NoopTracerProvider>());
        g_provider.reset();
        std::clog << "BeatSync: Tracing shutdown" << std::endl;
    }
}

} // namespace BeatSync

#else

namespace BeatSync {

bool InitializeTracing(const std::string& serviceName) {
    // No-op stub when tracing is disabled or not available
    (void)serviceName;
    std::clog << "BeatSync: Tracing not enabled in this build" << std::endl;
    return false;
}

void ShutdownTracing() {
    // No-op
}

} // namespace BeatSync

#endif
