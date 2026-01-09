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

		// ThirdParty beatsync library path
		string ThirdPartyPath = Path.Combine(PluginDirectory, "..", "..", "ThirdParty", "beatsync");

		if (Target.Platform == UnrealTargetPlatform.Win64)
		{
			string LibPath = Path.Combine(ThirdPartyPath, "lib", "x64");
			PublicAdditionalLibraries.Add(Path.Combine(LibPath, "beatsync_backend.lib"));
			RuntimeDependencies.Add(Path.Combine(LibPath, "beatsync_backend.dll"));
		}
		else if (Target.Platform == UnrealTargetPlatform.Mac)
		{
			string LibPath = Path.Combine(ThirdPartyPath, "lib", "Mac");
			PublicAdditionalLibraries.Add(Path.Combine(LibPath, "libbeatsync_backend.dylib"));
			RuntimeDependencies.Add(Path.Combine(LibPath, "libbeatsync_backend.dylib"));
			// Add Python beat detection script
			RuntimeDependencies.Add(Path.Combine(ThirdPartyPath, "beat_detect.py"));
		}
		else if (Target.Platform == UnrealTargetPlatform.Linux)
		{
			string LibPath = Path.Combine(ThirdPartyPath, "lib", "Linux");
			PublicAdditionalLibraries.Add(Path.Combine(LibPath, "libbeatsync_backend.so"));
			RuntimeDependencies.Add(Path.Combine(LibPath, "libbeatsync_backend.so"));
		}
	}
}
