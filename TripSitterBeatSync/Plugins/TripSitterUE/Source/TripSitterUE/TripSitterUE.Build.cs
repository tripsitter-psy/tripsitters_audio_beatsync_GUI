using UnrealBuildTool;
using System.IO;

public class TripSitterUE : ModuleRules
{
	public TripSitterUE(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicIncludePaths.AddRange(
			new string[] {
				Path.Combine(ModuleDirectory, "Public")
			}
		);

		PrivateIncludePaths.AddRange(
			new string[] {
				Path.Combine(ModuleDirectory, "Private")
			}
		);

		PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				"CoreUObject",
				"Engine",
				"InputCore",
				"Slate",
				"SlateCore",
				"UMG",
				"MediaAssets",
				"Projects",
				"Json",
				"JsonUtilities"
			}
		);

		// DesktopPlatform is only available in Editor builds
		if (Target.bBuildEditor || Target.Type == TargetType.Editor)
		{
			PrivateDependencyModuleNames.Add("DesktopPlatform");
			PrivateDefinitions.Add("WITH_DESKTOP_PLATFORM=1");
		}
		else
		{
			PrivateDefinitions.Add("WITH_DESKTOP_PLATFORM=0");
		}

		// Native C++ beat detection - no external libraries needed
		// All beat detection and waveform extraction is done in BeatsyncSubsystem.cpp
	}
}
