// TripSitter - Alternative Program Entry Point (FTripSitterApplication-based)
// Note: Main entry point is TripSitterApp.cpp with RunTripSitter()

#include "CoreMinimal.h"
#include "RequiredProgramMainCPPInclude.h"
#include "TripSitterApplication.h"

IMPLEMENT_APPLICATION(TripSitterAlt, "TripSitter");

INT32_MAIN_INT32_ARGC_TCHAR_ARGV()
{
    // Initialize the engine with TCHAR arguments (platform-correct)
    int32 PreInitResult = GEngineLoop.PreInit(ArgC, ArgV);
    if (PreInitResult != 0)
    {
        UE_LOG(LogTemp, Error, TEXT("Engine PreInit failed with error code: %d"), PreInitResult);
        return PreInitResult;
    }

    // Create and run the application
    FTripSitterApplication App;

    if (!App.Initialize())
    {
        // Proper teardown on initialization failure
        GEngineLoop.AppPreExit();
        GEngineLoop.AppExit();
        GEngineLoop.Exit();
        return 1;
    }

    int32 ExitCode = App.Run();

    App.Shutdown();

    // Shutdown the engine properly
    GEngineLoop.AppPreExit();
    GEngineLoop.AppExit();

    return ExitCode;
}