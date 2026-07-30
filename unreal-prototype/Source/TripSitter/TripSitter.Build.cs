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
        // NOTE: TripSitterUE plugin removed - standalone program has its own BeatsyncLoader/ProcessingTask
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

        if (Target.IsInPlatformGroup(UnrealPlatformGroup.Linux))
        {
            PrivateDependencyModuleNames.AddRange(new string[] {
                "UnixCommonStartup",   // CommonUnixMain entry point wrapper
                "DesktopPlatform",     // File dialogs (routes to SlateFileDialogs on Linux)
                "SlateFileDialogs"
            });
        }

        // Beatsync backend DLL path - conventional ThirdParty location under Engine/Binaries
        // This follows UE's standard pattern for third-party binaries and ensures proper
        // packaging/deployment. The DLL will be staged alongside the executable at runtime.
        string archFolder = "x64"; // Default
        if (Target.Architecture == UnrealArch.Arm64)
        {
            archFolder = "arm64";
        }
        string BeatsyncLib = Path.Combine(EngineDirectory, "Binaries", "ThirdParty", "Beatsync", archFolder);

        if (!Directory.Exists(BeatsyncLib))
        {
            throw new BuildException("TripSitter: BeatsyncLib directory not found at: " + BeatsyncLib);
        }

        // Robust parent-directory resolution for include path
        // BeatsyncLib points to .../Beatsync/x64, we need .../Beatsync/include
        // beatsyncLibDir.Parent yields the Beatsync folder, so combine with "include" directly
        var beatsyncLibDir = new DirectoryInfo(BeatsyncLib);
        var beatsyncDir = beatsyncLibDir.Parent;  // This is the Beatsync folder
        if (beatsyncDir == null)
        {
            throw new BuildException("TripSitter: Could not resolve parent directory for BeatsyncLib at: " + BeatsyncLib);
        }

        var includePath = Path.Combine(beatsyncDir.FullName, "include");
        if (!Directory.Exists(includePath))
        {
            throw new BuildException("TripSitter: Beatsync include directory not found at: " + includePath);
        }
        PublicIncludePaths.Add(includePath);

        if (Target.Platform == UnrealTargetPlatform.Win64)
        {
            var libPath = Path.Combine(BeatsyncLib, "beatsync_backend_shared.lib");
            var dllPath = Path.Combine(BeatsyncLib, "beatsync_backend_shared.dll");

            if (!File.Exists(libPath))
            {
                throw new BuildException("TripSitter: Import library not found at: " + libPath);
            }
            if (!File.Exists(dllPath))
            {
                throw new BuildException("TripSitter: DLL not found at: " + dllPath);
            }

            PublicAdditionalLibraries.Add(libPath);
            RuntimeDependencies.Add(dllPath);
        }
        else if (Target.Platform == UnrealTargetPlatform.Mac)
        {
            var dylibPath = Path.Combine(BeatsyncLib, "libbeatsync_backend_shared.dylib");

            if (!File.Exists(dylibPath))
            {
                throw new BuildException("TripSitter: dylib not found at: " + dylibPath);
            }

            PublicAdditionalLibraries.Add(dylibPath);
            // For macOS app bundles, stage the dylib in Contents/Frameworks for @rpath resolution
            RuntimeDependencies.Add("$(BinaryOutputDir)/../Frameworks/libbeatsync_backend_shared.dylib", dylibPath, StagedFileType.NonUFS);
        }
        else if (Target.Platform == UnrealTargetPlatform.Linux)
        {
            var soPath = Path.Combine(BeatsyncLib, "libbeatsync_backend_shared.so");

            if (!File.Exists(soPath))
            {
                throw new BuildException("TripSitter: Shared library not found at: " + soPath);
            }

            PublicAdditionalLibraries.Add(soPath);
            RuntimeDependencies.Add(soPath);
        }
    }
}
