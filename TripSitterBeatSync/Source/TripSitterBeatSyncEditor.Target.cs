using UnrealBuildTool;
using System.Collections.Generic;

public class TripSitterBeatSyncEditorTarget : TargetRules
{
	public TripSitterBeatSyncEditorTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Editor;
		DefaultBuildSettings = BuildSettingsVersion.V5;
		IncludeOrderVersion = EngineIncludeOrderVersion.Unreal5_7;
		ExtraModuleNames.Add("TripSitterBeatSync");
		bOverrideBuildEnvironment = true;
	}
}
