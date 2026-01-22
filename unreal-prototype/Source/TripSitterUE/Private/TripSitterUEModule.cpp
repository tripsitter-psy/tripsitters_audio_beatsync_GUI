#include "TripSitterUEModule.h"
#include "Modules/ModuleManager.h"
#include "Misc/Paths.h"
#include "HAL/PlatformProcess.h"

IMPLEMENT_MODULE(FTripSitterUEModule, TripSitterUE)

namespace {
using bs_init_tracing_t = int (*)(const char*);
using bs_shutdown_tracing_t = void (*)();

static void* BeatsyncDllHandle = nullptr;

static FString GetBeatsyncDllPath()
{
    FString Filename;
    FString Subdir;
    #if PLATFORM_WINDOWS
        Filename = TEXT("beatsync_backend_shared.dll");
        Subdir = TEXT("x64");
    #elif PLATFORM_MAC
        Filename = TEXT("libbeatsync_backend_shared.dylib");
        // Match TripSitter.Build.cs: use arch-specific subdir
        #if defined(__aarch64__) || defined(__arm64__) || defined(PLATFORM_MAC_ARM64)
            Subdir = TEXT("arm64");
        #else
            Subdir = TEXT("x64");
        #endif
    #else
        Filename = TEXT("libbeatsync_backend_shared.so");
        Subdir = TEXT("Linux");
    #endif
    FString DllPath = FPaths::Combine(FPaths::ProjectDir(), TEXT(".."), TEXT("unreal-prototype"), TEXT("ThirdParty"), TEXT("beatsync"), TEXT("lib"), Subdir, Filename);

    if (!FPaths::FileExists(DllPath)) {
        // Try engine-relative fallback
        FString Relative = FPaths::Combine(FPaths::ConvertRelativePathToFull(FPaths::EngineDir()), TEXT(".."));
        DllPath = FPaths::Combine(Relative, TEXT("unreal-prototype"), TEXT("ThirdParty"), TEXT("beatsync"), TEXT("lib"), Subdir, Filename);
        if (!FPaths::FileExists(DllPath)) {
            UE_LOG(LogTemp, Warning, TEXT("Beatsync DLL not found. Checked project-relative and engine-relative paths. Last attempted: %s (Relative base: %s). FPaths::FileExists returned false."), *DllPath, *Relative);
            // Indicate failure explicitly by returning empty string
            return FString();
        }
    }

    return DllPath;
}

static void CallBackendInitTracing(const FString& ServiceName)
{
    // Guard against multiple initialization - early return if already loaded
    if (BeatsyncDllHandle) {
        UE_LOG(LogTemp, Log, TEXT("TripSitterUEModule: Tracing already initialized, skipping."));
        return;
    }

    // Attempt to locate the backend shared library (same logic as BeatsyncLoader)
    FString DllPath = GetBeatsyncDllPath();

    if (DllPath.IsEmpty()) {
        UE_LOG(LogTemp, Warning, TEXT("TripSitterUEModule: Beatsync DLL not found, skipping tracing init."));
        return;
    }

    BeatsyncDllHandle = FPlatformProcess::GetDllHandle(*DllPath);
    if (!BeatsyncDllHandle) {
        UE_LOG(LogTemp, Warning, TEXT("TripSitterUEModule: Unable to load beatsync backend DLL at %s"), *DllPath);
        return;
    }

    void* Symbol = FPlatformProcess::GetDllExport(BeatsyncDllHandle, TEXT("bs_initialize_tracing"));
    if (!Symbol) {
        UE_LOG(LogTemp, Warning, TEXT("TripSitterUEModule: bs_initialize_tracing not found in backend"));
        FPlatformProcess::FreeDllHandle(BeatsyncDllHandle);
        BeatsyncDllHandle = nullptr;
        return;
    }

    bs_init_tracing_t Init = reinterpret_cast<bs_init_tracing_t>(Symbol);
    int Result = Init(TCHAR_TO_UTF8(*ServiceName));
    if (Result != 0)
    {
        UE_LOG(LogTemp, Error, TEXT("TripSitterUEModule: bs_initialize_tracing failed for service '%s' with code %d"), *ServiceName, Result);
        FPlatformProcess::FreeDllHandle(BeatsyncDllHandle);
        BeatsyncDllHandle = nullptr;
        return;
    }
}

static void CallBackendShutdownTracing()
{
    if (!BeatsyncDllHandle) return;

    void* Symbol = FPlatformProcess::GetDllExport(BeatsyncDllHandle, TEXT("bs_shutdown_tracing"));
    if (!Symbol) {
        UE_LOG(LogTemp, Warning, TEXT("bs_shutdown_tracing symbol not found in Beatsync DLL"));
        return;
    }

    bs_shutdown_tracing_t Shutdown = reinterpret_cast<bs_shutdown_tracing_t>(Symbol);
    Shutdown();
}
} // end anonymous namespace

void FTripSitterUEModule::StartupModule()
{
    // Try to init tracing with service name "tripsitter"
    CallBackendInitTracing(TEXT("tripsitter"));
}

void FTripSitterUEModule::ShutdownModule()
{
    // Try to shutdown tracing
    CallBackendShutdownTracing();

    // Free the DLL handle
    if (BeatsyncDllHandle) {
        FPlatformProcess::FreeDllHandle(BeatsyncDllHandle);
        BeatsyncDllHandle = nullptr;
    }
}
