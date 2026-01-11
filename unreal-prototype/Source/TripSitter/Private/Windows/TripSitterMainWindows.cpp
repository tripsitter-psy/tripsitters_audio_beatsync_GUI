// TripSitter - Beat Sync Editor (Windows Entry Point)

#include "TripSitterApp.h"
#include "Windows/WindowsHWrapper.h"

/**
 * WinMain, called when the application is started
 */
int WINAPI WinMain(_In_ HINSTANCE hInInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR, _In_ int nCmdShow)
{
	RunTripSitter(GetCommandLineW());
	return 0;
}
