using UnrealBuildTool;

public class TripSitter : ModuleRules
{
    public TripSitter(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        // For Program targets, we need minimal dependencies
        PublicDependencyModuleNames.AddRange(new string[] {
            "Core",
            "CoreUObject",
            "ApplicationCore",
            "Slate",
            "SlateCore",
            "StandaloneRenderer"
        });

        // Enable exceptions for C++ code
        bEnableExceptions = true;

        // Enable Slate UI
        bUsesSlate = true;

        // Add include paths for our backend
        PublicIncludePaths.AddRange(new string[] {
            "$(ProjectDir)/ThirdParty/beatsync/include"
        });

        // Add library paths and dependencies
        PublicAdditionalLibraries.Add("$(ProjectDir)/ThirdParty/beatsync/lib/beatsync.lib");

        // Copy DLLs to output directory
        RuntimeDependencies.Add("$(ProjectDir)/ThirdParty/beatsync/bin/beatsync.dll");

        // Enable C++17
        CppStandard = CppStandardVersion.Cpp17;
    }
}