using UnrealBuildTool;

public class TripSitterTarget : TargetRules
{
    public TripSitterTarget(TargetInfo Target) : base(Target)
    {
        Type = TargetType.Program;
        LaunchModuleName = "TripSitter";
        DefaultBuildSettings = BuildSettingsVersion.V6;
        IncludeOrderVersion = EngineIncludeOrderVersion.Unreal5_7;

        bCompileAgainstEngine = false;
        bCompileAgainstCoreUObject = true;
        bCompileAgainstApplicationCore = true;
        bUsesSlate = true;
        bIsBuildingConsoleApplication = false;
    }
}