#include "TripSitterUEModule.h"
#include "Modules/ModuleManager.h"
#include "Misc/Paths.h"
#include "HAL/PlatformProcess.h"
#include "Misc/OutputDeviceNull.h"
#include "LevelEditor.h"
#include "Widgets/Docking/SDockTab.h"
#include "Framework/Application/SlateApplication.h"
#include "Widgets/SWindow.h"
#include "Widgets/Text/STextBlock.h"
#include "Widgets/SBoxPanel.h"

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
    Subdir = TEXT("Mac");
#else
    Filename = TEXT("beatsync_backend_shared.so");
    Subdir = TEXT("Linux");
#endif
    FString DllPath = FPaths::Combine(FPaths::ProjectDir(), TEXT(".."), TEXT("unreal-prototype"), TEXT("ThirdParty"), TEXT("beatsync"), TEXT("lib"), Subdir, Filename);

    if (!FPaths::FileExists(DllPath)) {
        // Try engine-relative fallback
        FString Relative = FPaths::Combine(FPaths::ConvertRelativePathToFull(FPaths::EngineDir()), TEXT(".."));
        DllPath = FPaths::Combine(Relative, TEXT("unreal-prototype"), TEXT("ThirdParty"), TEXT("beatsync"), TEXT("lib"), Subdir, Filename);
    }

    return DllPath;
}

static void CallBackendInitTracing(const FString& ServiceName)
{
    // Attempt to locate the backend shared library (same logic as BeatsyncLoader)
    FString DllPath = GetBeatsyncDllPath();

    // Prevent double-load or leak: free previous handle if already loaded
    if (BeatsyncDllHandle)
    {
        FPlatformProcess::FreeDllHandle(BeatsyncDllHandle);
        BeatsyncDllHandle = nullptr;
    }

    BeatsyncDllHandle = FPlatformProcess::GetDllHandle(*DllPath);
    if (!BeatsyncDllHandle) {
        UE_LOG(LogTemp, Warning, TEXT("TripSitterUEModule: Unable to find beatsync backend DLL at %s"), *DllPath);
        return;
    }

    void* Symbol = FPlatformProcess::GetDllExport(BeatsyncDllHandle, TEXT("bs_initialize_tracing"));
    if (!Symbol) {
        UE_LOG(LogTemp, Warning, TEXT("TripSitterUEModule: bs_initialize_tracing not found in backend"));
        // Cleanup DLL handle to avoid leak
        FPlatformProcess::FreeDllHandle(BeatsyncDllHandle);
        BeatsyncDllHandle = nullptr;
        return;
    }

    bs_init_tracing_t Init = reinterpret_cast<bs_init_tracing_t>(Symbol);
    FTCHARToUTF8 UTF8ServiceName(*ServiceName);
    int Result = Init(UTF8ServiceName.Get());
    // C++ InitializeTracing returns 1 on success, 0 on failure
    if (Result == 0)
    {
        UE_LOG(LogTemp, Error, TEXT("TripSitterUEModule: bs_initialize_tracing failed with code %d for service '%s'"), Result, *ServiceName);
        // Cleanup DLL handle on initialization failure
        FPlatformProcess::FreeDllHandle(BeatsyncDllHandle);
        BeatsyncDllHandle = nullptr;
        return;
    }
}

static void CallBackendShutdownTracing()
{
    if (!BeatsyncDllHandle) return;

    void* Symbol = FPlatformProcess::GetDllExport(BeatsyncDllHandle, TEXT("bs_shutdown_tracing"));
    if (!Symbol) return;

    bs_shutdown_tracing_t Shutdown = reinterpret_cast<bs_shutdown_tracing_t>(Symbol);
    Shutdown();
}
} // end anonymous namespace

void FTripSitterUEModule::AddMenuExtension(FMenuBuilder& MenuBuilder)
{
    MenuBuilder.AddMenuEntry(
        FText::FromString("Open TripSitter"),
        FText::FromString("Open the TripSitter beat sync editor window"),
        FSlateIcon(),
        FUIAction(FExecuteAction::CreateRaw(this, &FTripSitterUEModule::OpenTripSitterWindow))
    );
}

void FTripSitterUEModule::OpenTripSitterWindow()
{
    // Create a simple window with TripSitter content
    TSharedRef<SWindow> Window = SNew(SWindow)
        .Title(FText::FromString("TripSitter - Beat Sync Editor"))
        .ClientSize(FVector2D(800, 600))
        .SupportsMaximize(true)
        .SupportsMinimize(true);

    Window->SetContent(
        SNew(SVerticalBox)
        + SVerticalBox::Slot()
        .FillHeight(1.0f)
        [
            SNew(STextBlock)
            .Text(FText::FromString(TEXT("TripSitter - Beat Sync Editor\n\nThis is a placeholder window. The full TripSitter GUI is implemented in the TripSitter module.")))
        ]
    );

    FSlateApplication::Get().AddWindow(Window);
}

void FTripSitterUEModule::StartupModule()
{
    // Try to init tracing with service name "tripsitter"
    CallBackendInitTracing(TEXT("tripsitter"));

    // Add menu item to open TripSitter window
    if (FModuleManager::Get().IsModuleLoaded("LevelEditor"))
    {
        FLevelEditorModule& LevelEditorModule = FModuleManager::LoadModuleChecked<FLevelEditorModule>("LevelEditor");
        MenuExtender = MakeShareable(new FExtender());
        MenuExtender->AddMenuExtension(
            "WindowLayout",
            EExtensionHook::After,
            nullptr,
            FMenuExtensionDelegate::CreateRaw(this, &FTripSitterUEModule::AddMenuExtension)
        );
        LevelEditorModule.GetMenuExtensibilityManager()->AddExtender(MenuExtender);
    }
}

void FTripSitterUEModule::ShutdownModule()
{
    // Remove menu extender to prevent crashes on shutdown/hot-reload
    if (MenuExtender.IsValid() && FModuleManager::Get().IsModuleLoaded("LevelEditor"))
    {
        FLevelEditorModule& LevelEditorModule = FModuleManager::LoadModuleChecked<FLevelEditorModule>("LevelEditor");
        LevelEditorModule.GetMenuExtensibilityManager()->RemoveExtender(MenuExtender);
        MenuExtender.Reset();
    }

    // Try to shutdown tracing
    CallBackendShutdownTracing();

    // Free the DLL handle
    if (BeatsyncDllHandle) {
        FPlatformProcess::FreeDllHandle(BeatsyncDllHandle);
        BeatsyncDllHandle = nullptr;
    }
}
