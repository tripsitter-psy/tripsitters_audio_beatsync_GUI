#include "CoreMinimal.h"
#include "TripSitterApplication.h"

// Program entry point for TripSitter desktop application
int32 main(int32 argc, char* argv[])
{
    // Initialize the engine
    int32 PreInitResult = GEngineLoop.PreInit(argc, argv);
    if (PreInitResult != 0)
    {
        UE_LOG(LogTemp, Error, TEXT("Engine PreInit failed with error code: %d"), PreInitResult);
        return PreInitResult;
    }

    // Create and run the application
    FTripSitterApplication App;

    if (!App.Initialize())
    {
        return 1;
    }

    int32 ExitCode = App.Run();

    App.Shutdown();

    // Shutdown the engine
    GEngineLoop.Exit();

    return ExitCode;
}