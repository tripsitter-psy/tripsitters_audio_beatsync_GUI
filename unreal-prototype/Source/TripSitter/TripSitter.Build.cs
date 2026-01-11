// TripSitter - Beat Sync Editor Module

using UnrealBuildTool;
using System.IO;

public class TripSitter : ModuleRules
{
    public TripSitter(ReadOnlyTargetRules Target) : base(Target)
    {
        // Include paths for this program
        PublicIncludePaths.Add("Programs/TripSitter");
        PrivateIncludePaths.Add("Programs/TripSitter/Private");

        // For RequiredProgramMainCPPInclude.h - only include path, not full dependency
        PublicIncludePathModuleNames.Add("Launch");

        // Core dependencies (matching SlateViewer pattern - NO Launch!)
        PrivateDependencyModuleNames.AddRange(new string[] {
            "AppFramework",
            "Core",
            "ApplicationCore",
            "Projects",
            "Slate",
            "SlateCore",
            "StandaloneRenderer",
            "ImageWrapper"
        });

        // Beatsync backend DLL path - relative to Engine folder
        // The DLL will need to be deployed alongside the executable
        string BeatsyncLib = Path.Combine(EngineDirectory, "Source", "Programs", "TripSitter", "ThirdParty", "beatsync", "lib", "x64");
        if (Directory.Exists(BeatsyncLib))
        {
            PublicIncludePaths.Add(Path.Combine(BeatsyncLib, "..", "..", "include"));
            PublicAdditionalLibraries.Add(Path.Combine(BeatsyncLib, "beatsync_backend_shared.lib"));
            RuntimeDependencies.Add(Path.Combine(BeatsyncLib, "beatsync_backend_shared.dll"));
        }

        // Windows application icon resource
        if (Target.Platform == UnrealTargetPlatform.Win64)
        {
            string ResourcePath = Path.Combine(ModuleDirectory, "Private", "Windows", "Resources", "TripSitter.rc");
            if (File.Exists(ResourcePath))
            {
                // UBT should pick up .rc files automatically, but we can also add via AdditionalCompilerArguments
                // For now, rely on auto-detection since .rc is in the source tree
            }
        }
    }
}
