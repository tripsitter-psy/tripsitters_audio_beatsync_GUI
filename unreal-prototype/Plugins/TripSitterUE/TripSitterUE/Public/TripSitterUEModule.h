#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"
#include "Framework/Commands/UICommandList.h"
#include "Framework/MultiBox/MultiBoxBuilder.h"

class FExtender;

class TRIPSITTERUE_API FTripSitterUEModule : public IModuleInterface
{
public:
    virtual void StartupModule() override;
    virtual void ShutdownModule() override;

private:
    void AddMenuExtension(FMenuBuilder& MenuBuilder);
    void OpenTripSitterWindow();

    TSharedPtr<FExtender> MenuExtender;
};
