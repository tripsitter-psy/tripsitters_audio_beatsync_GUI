#include "BeatSyncGameMode.h"
#include "BeatSyncHUD.h"

ABeatSyncGameMode::ABeatSyncGameMode()
{
	// Set default HUD class to our custom BeatSync HUD
	HUDClass = ABeatSyncHUD::StaticClass();

	// No default pawn - this is a UI application
	DefaultPawnClass = nullptr;

	// Use default player controller for now
	// PlayerControllerClass = ...
}
