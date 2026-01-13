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
        string archFolder = Target.Architecture switch
        {
            UnrealArch.X64 => "x64",
            UnrealArch.Arm64 => "arm64",
            _ => "x64" // Default fallback
        };
        string BeatsyncLib = Path.Combine(EngineDirectory, "Source", "Programs", "TripSitter", "ThirdParty", "beatsync", "lib", archFolder);

        if (!Directory.Exists(BeatsyncLib))
        {
            System.Console.WriteLine($"[TripSitter.Build.cs] WARNING: BeatsyncLib directory not found: {BeatsyncLib}");
        }
        else
        {
            PublicIncludePaths.Add(Path.Combine(BeatsyncLib, "..", "..", "include"));

            if (Target.Platform == UnrealTargetPlatform.Win64)
            {
                PublicAdditionalLibraries.Add(Path.Combine(BeatsyncLib, "beatsync_backend_shared.lib"));
                RuntimeDependencies.Add(Path.Combine(BeatsyncLib, "beatsync_backend_shared.dll"));
            }
            else if (Target.Platform == UnrealTargetPlatform.Mac)
            {
                PublicAdditionalLibraries.Add(Path.Combine(BeatsyncLib, "beatsync_backend_shared.dylib"));
                RuntimeDependencies.Add(Path.Combine(BeatsyncLib, "beatsync_backend_shared.dylib"));
            }
            else if (Target.Platform == UnrealTargetPlatform.Linux)
            {
                PublicAdditionalLibraries.Add(Path.Combine(BeatsyncLib, "beatsync_backend_shared.so"));
                RuntimeDependencies.Add(Path.Combine(BeatsyncLib, "beatsync_backend_shared.so"));
            }
        }
    }
}
