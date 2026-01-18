// TripSitter - Windows Entry Point

#include "CoreMinimal.h"
#include "Windows/WindowsHWrapper.h"

// Forward declaration
int RunTripSitter(const TCHAR* CommandLine);

/**
 * WinMain, called when the application is started
 */
int WINAPI WinMain(_In_ HINSTANCE hInInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR, _In_ int nCmdShow)
{
    // Run TripSitter with the command line
    return RunTripSitter(GetCommandLineW());
}
