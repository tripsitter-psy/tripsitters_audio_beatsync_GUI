#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"

class FTripSitterUEModule : public IModuleInterface
{
public:
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;

	static inline FTripSitterUEModule& Get()
	{
		return FModuleManager::LoadModuleChecked<FTripSitterUEModule>("TripSitterUE");
	}

	static inline bool IsAvailable()
	{
		return FModuleManager::Get().IsModuleLoaded("TripSitterUE");
	}
};
