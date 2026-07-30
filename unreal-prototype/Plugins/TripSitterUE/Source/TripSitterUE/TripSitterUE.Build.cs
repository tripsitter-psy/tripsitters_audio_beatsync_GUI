using UnrealBuildTool;

public class TripSitterUE : ModuleRules
{
    public TripSitterUE(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[] {
            "Core",
            "CoreUObject",
            "InputCore",
            "ApplicationCore",
            "Projects"
        });

        PrivateDependencyModuleNames.AddRange(new string[] {
            "Slate",
            "SlateCore",
            "StandaloneRenderer",
            "ImageCore"
        });

        // Engine dependency only needed for Editor and Game targets, not Program
        if (Target.Type != TargetType.Program)
        {
            PublicDependencyModuleNames.Add("Engine");
        }

        // Editor-only dependencies
        if (Target.Type == TargetType.Editor)
        {
            PrivateDependencyModuleNames.Add("UnrealEd");
        }
    }
}