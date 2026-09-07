// TripSitter - Program Entry Point
#include "CoreMinimal.h"
#include "RequiredProgramMainCPPInclude.h"
#include "Framework/Application/SlateApplication.h"
#include "StandaloneRenderer.h"
#include "Stats/StatsSystem.h"
#include "Private/STripSitterMainWidget.h"
#include "BeatsyncLoader.h"  // From TripSitterUE plugin
#include "Misc/Paths.h"
#include "HAL/PlatformProcess.h"
#include "Styling/AppStyle.h"
#include "Styling/SlateStyle.h"
#include "Styling/SlateStyleRegistry.h"

#if PLATFORM_WINDOWS
#include "Windows/AllowWindowsPlatformTypes.h"
#include <Windows.h>
#include "Windows/HideWindowsPlatformTypes.h"
// Store custom icon handle for cleanup on exit
static HICON GWindowIcon = nullptr;
#endif

IMPLEMENT_APPLICATION(TripSitter, "TripSitter");

// Slate draws its own title bar for SWindows without an OS border, and puts the
// active app style's "AppIcon" brush (the Unreal logo by default) in its top-left
// corner. This style set shadows just that brush with the TripSitter logo and
// inherits everything else from the core style.
static TSharedPtr<FSlateStyleSet> GTripSitterStyle;

static FString FindResourceFile(const TCHAR* FileName)
{
    const FString ExeDir = FPaths::GetPath(FPlatformProcess::ExecutablePath());
    TArray<FString> Candidates;
    Candidates.Add(FPaths::Combine(ExeDir, TEXT("Resources"), FileName));
    const FString EnvDir = FPlatformMisc::GetEnvironmentVariable(TEXT("BEATSYNC_RESOURCES"));
    if (!EnvDir.IsEmpty())
    {
        Candidates.Add(FPaths::Combine(EnvDir, FileName));
    }
    Candidates.Add(FPaths::Combine(FPaths::ProjectDir(), TEXT("Source"), TEXT("TripSitter"), TEXT("Resources"), FileName));
    Candidates.Add(FPaths::ConvertRelativePathToFull(FPaths::Combine(ExeDir, TEXT(".."), TEXT(".."), TEXT("Source"), TEXT("Programs"), TEXT("TripSitter"), TEXT("Resources"), FileName)));
    for (const FString& Path : Candidates)
    {
        if (FPaths::FileExists(Path))
        {
            return Path;
        }
    }
    return FString();
}

static void RegisterTripSitterAppStyle()
{
    // TitleIcon.png is a pre-scaled (128px) copy of icon.png: the standalone
    // renderer does not filter when downsampling, so the full-size art looks rough.
    FString IconPath = FindResourceFile(TEXT("TitleIcon.png"));
    if (IconPath.IsEmpty())
    {
        IconPath = FindResourceFile(TEXT("icon.png"));
    }
    if (IconPath.IsEmpty())
    {
        UE_LOG(LogTemp, Warning, TEXT("TripSitter: Resources/TitleIcon.png not found; keeping the default title bar icon"));
        return;
    }

    GTripSitterStyle = MakeShared<FSlateStyleSet>(TEXT("TripSitterStyle"));
    GTripSitterStyle->SetParentStyleName(FAppStyle::GetAppStyleSetName());
    // The logo art is 857x779; keep its aspect at the core style's 45px icon height.
    GTripSitterStyle->Set("AppIcon", new FSlateImageBrush(IconPath, FVector2D(49.5f, 45.f)));
    GTripSitterStyle->Set("AppIcon.Small", new FSlateImageBrush(IconPath, FVector2D(26.4f, 24.f)));
    FSlateStyleRegistry::RegisterSlateStyle(*GTripSitterStyle);
    FAppStyle::SetAppStyleSetName(GTripSitterStyle->GetStyleSetName());
    UE_LOG(LogTemp, Log, TEXT("TripSitter: title bar icon set from %s"), *IconPath);
}

static void UnregisterTripSitterAppStyle()
{
    if (GTripSitterStyle.IsValid())
    {
        FSlateStyleRegistry::UnRegisterSlateStyle(*GTripSitterStyle);
        GTripSitterStyle.Reset();
    }
}

int RunTripSitter(const TCHAR* CommandLine)
{
    FTaskTagScope TaskTagScope(ETaskTag::EGameThread);

    // Initialize the engine
    int32 PreInitResult = GEngineLoop.PreInit(CommandLine);
    if (PreInitResult != 0)
    {
        UE_LOG(LogTemp, Error, TEXT("GEngineLoop.PreInit failed with code %d"), PreInitResult);
        FPlatformMisc::RequestExit(true);
        return PreInitResult;
    }

    // Make sure all UObject classes are registered and default properties have been initialized
    ProcessNewlyLoadedUObjects();

    // Tell the module manager it may now process newly-loaded UObjects when new C++ modules are loaded
    FModuleManager::Get().StartProcessingNewlyLoadedObjects();

    // Initialize Slate as standalone application
    FSlateApplication::InitializeAsStandaloneApplication(GetStandardStandaloneRenderer());
    FSlateApplication::InitHighDPI(true);
    RegisterTripSitterAppStyle();

    // Initialize the beatsync backend DLL
    if (!FBeatsyncLoader::Initialize())
    {
        UE_LOG(LogTemp, Warning, TEXT("Failed to initialize beatsync backend. Beat sync features will be disabled."));
    }

    // Create main window
    // Held as a TSharedPtr so the reference can be released before Slate and ICU
    // are torn down (see cleanup below).
    TSharedPtr<SWindow> MainWindow = SNew(SWindow)
        .Title(FText::FromString(TEXT("MTV Trip Sitter")))
        .ClientSize(FVector2D(1400, 900))
        .SupportsMaximize(true)
        .SupportsMinimize(true)
        .IsInitiallyMaximized(false);

    // Create the main widget content
    MainWindow->SetContent(
        SNew(STripSitterMainWidget)
    );

    // Add window and show
    FSlateApplication::Get().AddWindow(MainWindow.ToSharedRef());
    MainWindow->ShowWindow();
    MainWindow->BringToFront();

#if PLATFORM_WINDOWS
    // Set custom window icon (replaces default UE icon)
    if (TSharedPtr<FGenericWindow> NativeWindow = MainWindow->GetNativeWindow())
    {
        HWND Hwnd = (HWND)NativeWindow->GetOSWindowHandle();
        if (Hwnd)
        {
            FString ExeDir = FPaths::GetPath(FPlatformProcess::ExecutablePath());
            FString IconPath = FPaths::Combine(ExeDir, TEXT("Resources"), TEXT("TripSitter.ico"));

            // Fallback: dev builds have no Resources next to the exe; use the
            // engine source location (same fallback the main widget uses for
            // wallpaper/fonts). Without this the dev build shows the default UE icon.
            if (!FPaths::FileExists(IconPath))
            {
                FString DevIconPath = FPaths::Combine(ExeDir, TEXT(".."), TEXT(".."), TEXT("Source"),
                    TEXT("Programs"), TEXT("TripSitter"), TEXT("Resources"), TEXT("TripSitter.ico"));
                DevIconPath = FPaths::ConvertRelativePathToFull(DevIconPath);
                if (FPaths::FileExists(DevIconPath))
                {
                    IconPath = DevIconPath;
                }
            }

            if (FPaths::FileExists(IconPath))
            {
                GWindowIcon = (HICON)LoadImageW(NULL, *IconPath, IMAGE_ICON, 0, 0, LR_LOADFROMFILE | LR_DEFAULTSIZE);
                if (GWindowIcon)
                {
                    ::SendMessageW(Hwnd, WM_SETICON, ICON_BIG, (LPARAM)GWindowIcon);
                    ::SendMessageW(Hwnd, WM_SETICON, ICON_SMALL, (LPARAM)GWindowIcon);
                    UE_LOG(LogTemp, Log, TEXT("Custom window icon set from: %s"), *IconPath);
                }
                else
                {
                    UE_LOG(LogTemp, Warning, TEXT("Failed to load icon from: %s (Error: %d)"), *IconPath, GetLastError());
                }
            }
            else
            {
                UE_LOG(LogTemp, Warning, TEXT("Icon file not found: %s"), *IconPath);
            }
        }
    }
#endif

    // Main application loop
    while (!IsEngineExitRequested())
    {
        BeginExitIfRequested();

        FTaskGraphInterface::Get().ProcessThreadUntilIdle(ENamedThreads::GameThread);
        UE::Stats::FStats::AdvanceFrame(false);
        FTSTicker::GetCoreTicker().Tick(FApp::GetDeltaTime());
        FSlateApplication::Get().PumpMessages();
        FSlateApplication::Get().Tick();
        FPlatformProcess::Sleep(0.01f);

        GFrameCounter++;
    }

    // Cleanup
    FBeatsyncLoader::Shutdown();

    // Release the window before the shutdown calls below. Holding it until this
    // function returns would destruct its widgets - and their text layouts -
    // after AppExit() has unloaded Internationalization, so ICU aborts in
    // ubidi_close() during ~FICUTextBiDi.
    MainWindow.Reset();
    UnregisterTripSitterAppStyle();

#if PLATFORM_WINDOWS
    // Destroy custom window icon to prevent GDI resource leak
    if (GWindowIcon)
    {
        DestroyIcon(GWindowIcon);
        GWindowIcon = nullptr;
    }
#endif

    FCoreDelegates::OnExit.Broadcast();
    FSlateApplication::Shutdown();
    FModuleManager::Get().UnloadModulesAtShutdown();

    GEngineLoop.AppPreExit();
    GEngineLoop.AppExit();

    return 0;
}
