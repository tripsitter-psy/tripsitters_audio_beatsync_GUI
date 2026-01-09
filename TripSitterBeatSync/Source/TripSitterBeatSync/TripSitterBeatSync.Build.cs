using UnrealBuildTool;

public class TripSitterBeatSync : ModuleRules
{
	public TripSitterBeatSync(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[] {
			"Core",
			"CoreUObject",
			"Engine",
			"InputCore",
			"EnhancedInput",
			"UMG",
			"Slate",
			"SlateCore",
			"TripSitterUE"
		});

		PrivateDependencyModuleNames.AddRange(new string[] {
			"MediaAssets"
		});

		// DesktopPlatform is only available in Editor builds (precompiled UE limitation)
		if (Target.bBuildEditor || Target.Type == TargetType.Editor)
		{
			PrivateDependencyModuleNames.Add("DesktopPlatform");
			PrivateDefinitions.Add("WITH_DESKTOP_PLATFORM=1");
		}
		else
		{
			PrivateDefinitions.Add("WITH_DESKTOP_PLATFORM=0");
		}

		// For native file dialogs on Mac in packaged builds
		if (Target.Platform == UnrealTargetPlatform.Mac)
		{
			PublicFrameworks.Add("AppKit");
			PublicFrameworks.Add("CoreFoundation");
		}
	}
}
