#pragma once

#include "Modules/ModuleManager.h"
#include "Framework/Commands/UICommandList.h"

class FTripSitterUEModule : public IModuleInterface
{
public:
    virtual void StartupModule() override;
    virtual void ShutdownModule() override;

private:
    void AddMenuExtension(FMenuBuilder& MenuBuilder);
    void OpenTripSitterWindow();
};
