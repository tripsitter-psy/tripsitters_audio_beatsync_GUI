#include "CoreMinimal.h"
#include "TripSitterApplication.h"

// Program entry point for TripSitter desktop application
int32 main(int32 argc, char* argv[])
{
    // Initialize the engine
    GEngineLoop.PreInit(argc, argv);

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