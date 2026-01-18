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
            // Robust parent-directory resolution for include path
            var beatsyncLibDir = new System.IO.DirectoryInfo(BeatsyncLib);
            var parentDir = beatsyncLibDir.Parent?.Parent;
            if (parentDir != null)
            {
                var includePath = System.IO.Path.Combine(parentDir.FullName, "include");
                PublicIncludePaths.Add(includePath);
            }
            else
            {
                System.Console.WriteLine($"[TripSitter.Build.cs] ERROR: Could not resolve parent directories for BeatsyncLib: {BeatsyncLib}");
            }

            if (Target.Platform == UnrealTargetPlatform.Win64)
            {
                PublicAdditionalLibraries.Add(Path.Combine(BeatsyncLib, "beatsync_backend_shared.lib"));
                RuntimeDependencies.Add(Path.Combine(BeatsyncLib, "beatsync_backend_shared.dll"));
            }
            else if (Target.Platform == UnrealTargetPlatform.Mac)
            {
                var dylibPath = Path.Combine(BeatsyncLib, "libbeatsync_backend_shared.dylib");
                PublicAdditionalLibraries.Add(dylibPath);
                // Stage the dylib for packaging next to the executable
                RuntimeDependencies.Add(dylibPath);
                // For macOS app bundles, the dylib should be placed in Contents/Frameworks for @rpath resolution
                // Use $(BinaryOutputDir)/../Frameworks/ for proper staging relative to the executable
                RuntimeDependencies.Add("$(BinaryOutputDir)/../Frameworks/libbeatsync_backend_shared.dylib", dylibPath, StagedFileType.NonUFS);
            }
            else if (Target.Platform == UnrealTargetPlatform.Linux)
            {
                PublicAdditionalLibraries.Add(Path.Combine(BeatsyncLib, "beatsync_backend_shared.so"));
                RuntimeDependencies.Add(Path.Combine(BeatsyncLib, "beatsync_backend_shared.so"));
            }
        }
    }
}
