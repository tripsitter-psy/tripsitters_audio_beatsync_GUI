using UnrealBuildTool;
using System.Collections.Generic;

public class TripSitterBeatSyncTarget : TargetRules
{
	public TripSitterBeatSyncTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Game;
		DefaultBuildSettings = BuildSettingsVersion.V5;
		IncludeOrderVersion = EngineIncludeOrderVersion.Unreal5_7;
		ExtraModuleNames.Add("TripSitterBeatSync");
		bOverrideBuildEnvironment = true;
	}
}
