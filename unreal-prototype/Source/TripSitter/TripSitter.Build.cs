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
            "ImageWrapper",
            "TripSitterUE" // Plugin dependency
        });

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
            Log.TraceError("TripSitter: BeatsyncLib directory not found: {0}", BeatsyncLib);
            throw new BuildException("BeatsyncLib is required but not found at: " + BeatsyncLib);
        }

        // Robust parent-directory resolution for include path
        var beatsyncLibDir = new DirectoryInfo(BeatsyncLib);
        var parentDir = beatsyncLibDir.Parent?.Parent;
        if (parentDir == null)
        {
            Log.TraceError("TripSitter: Could not resolve parent directories for BeatsyncLib: {0}", BeatsyncLib);
            throw new BuildException("Could not resolve include path for BeatsyncLib at: " + BeatsyncLib);
        }

        var includePath = Path.Combine(parentDir.FullName, "include");
        if (!Directory.Exists(includePath))
        {
            Log.TraceError("TripSitter: Beatsync include directory not found: {0}", includePath);
            throw new BuildException("Beatsync include directory is required but not found at: " + includePath);
        }
        PublicIncludePaths.Add(includePath);

        if (Target.Platform == UnrealTargetPlatform.Win64)
        {
            var libPath = Path.Combine(BeatsyncLib, "beatsync_backend_shared.lib");
            var dllPath = Path.Combine(BeatsyncLib, "beatsync_backend_shared.dll");

            if (!File.Exists(libPath))
            {
                Log.TraceError("TripSitter: Import library not found: {0}", libPath);
                throw new BuildException("Beatsync import library is required but not found at: " + libPath);
            }
            if (!File.Exists(dllPath))
            {
                Log.TraceError("TripSitter: DLL not found: {0}", dllPath);
                throw new BuildException("Beatsync DLL is required but not found at: " + dllPath);
            }

            PublicAdditionalLibraries.Add(libPath);
            RuntimeDependencies.Add(dllPath);
        }
        else if (Target.Platform == UnrealTargetPlatform.Mac)
        {
            var dylibPath = Path.Combine(BeatsyncLib, "libbeatsync_backend_shared.dylib");

            if (!File.Exists(dylibPath))
            {
                Log.TraceError("TripSitter: dylib not found: {0}", dylibPath);
                throw new BuildException("Beatsync dylib is required but not found at: " + dylibPath);
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
                Log.TraceError("TripSitter: Shared library not found: {0}", soPath);
                throw new BuildException("Beatsync shared library is required but not found at: " + soPath);
            }

            PublicAdditionalLibraries.Add(soPath);
            RuntimeDependencies.Add(soPath);
        }
    }
}
