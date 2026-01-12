#include "tracing.h"
#include <iostream>
#include <cstdlib>

#if defined(BEATSYNC_ENABLE_TRACING)
// If the OpenTelemetry SDK is available, include the headers and set up an OTLP exporter.
// Note: Exact include paths may vary by distribution; leave this guarded so builds don't fail
// when the SDK is missing. The CMake detect above sets BEATSYNC_ENABLE_TRACING when the
// package was found.

#include <opentelemetry/sdk/trace/tracer_provider.h>
#include <opentelemetry/trace/provider.h>
#include <opentelemetry/exporters/otlp/otlp_grpc_exporter.h>
#include <opentelemetry/sdk/trace/simple_processor.h>

namespace {
using namespace opentelemetry;
using sdktrace = sdk::trace;
static nostd::shared_ptr<trace::TracerProvider> g_provider;
}

namespace BeatSync {

bool InitializeTracing(const std::string& serviceName) {
    const char* env = std::getenv("OTEL_EXPORTER_OTLP_ENDPOINT");
    std::string endpoint = env ? env : "http://localhost:4318";

    try {
        // Create OTLP exporter (gRPC) with configured endpoint
        exporters::otlp::OtlpGrpcExporterOptions options;
        options.endpoint = endpoint;
        auto exporter = std::unique_ptr<sdktrace::SpanExporter>(new exporters::otlp::OtlpGrpcExporter(options));
        auto processor = std::unique_ptr<sdktrace::SpanProcessor>(new sdktrace::SimpleSpanProcessor(std::move(exporter)));
        
        // Create resource with service name
        auto resource = sdk::resource::Resource::Create({{"service.name", serviceName}});
        auto provider = std::make_shared<sdktrace::TracerProvider>(resource, std::move(processor));
        
        trace::Provider::SetTracerProvider(provider);
        g_provider = provider;
        std::clog << "BeatSync: Tracing initialized (OTLP endpoint=" << endpoint << ", service=" << serviceName << ")\n";
        return true;
    } catch (const std::exception& e) {
        std::cerr << "BeatSync: Failed to initialize tracing: " << e.what() << "\n";
        return false;
    }
}

void ShutdownTracing() {
    if (g_provider) {
        // Shutdown the provider to flush any pending spans
        auto status = g_provider->Shutdown();
        if (!status.ok()) {
            std::cerr << "BeatSync: Warning - tracing shutdown failed: " << status.message() << "\n";
        }
        // Set to no-op provider instead of nullptr
        trace::Provider::SetTracerProvider(std::make_shared<trace::NoopTracerProvider>());
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
