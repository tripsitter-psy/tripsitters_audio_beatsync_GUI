#include "TripSitterUE.h"
#include "BeatsyncLoader.h"

#define LOCTEXT_NAMESPACE "FTripSitterUEModule"

void FTripSitterUEModule::StartupModule()
{
	UE_LOG(LogTemp, Log, TEXT("TripSitterUE: Starting module..."));

	if (FBeatsyncLoader::Initialize())
	{
		UE_LOG(LogTemp, Log, TEXT("TripSitterUE: Beatsync backend loaded successfully"));
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("TripSitterUE: Failed to load beatsync backend - some features may be unavailable"));
	}
}

void FTripSitterUEModule::ShutdownModule()
{
	UE_LOG(LogTemp, Log, TEXT("TripSitterUE: Shutting down module..."));
	FBeatsyncLoader::Shutdown();
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FTripSitterUEModule, TripSitterUE)
