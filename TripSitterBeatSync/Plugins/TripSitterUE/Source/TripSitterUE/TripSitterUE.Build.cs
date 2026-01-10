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

		// NNE (Neural Network Engine) for ONNX inference - BeatNet and Demucs
		// NNE is available in UE 5.3+
		PublicDependencyModuleNames.Add("NNE");

		// Try to add NNERuntimeORT if available (ONNX Runtime backend)
		// This provides the actual ONNX inference capability
		bool bHasNNERuntimeORT = false;
		string NNERuntimeORTPath = Path.Combine(EngineDirectory, "Plugins", "NNE", "NNERuntimeORT");
		if (Directory.Exists(NNERuntimeORTPath))
		{
			PrivateDependencyModuleNames.Add("NNERuntimeORT");
			bHasNNERuntimeORT = true;
		}

		// Define WITH_NNE to enable ONNX inference code paths
		if (bHasNNERuntimeORT)
		{
			PrivateDefinitions.Add("WITH_NNE=1");
		}
		else
		{
			PrivateDefinitions.Add("WITH_NNE=0");
		}

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

		// AI-powered beat detection using ONNX models:
		// - BeatNet: Neural network beat/downbeat detection
		// - Demucs: Audio source separation (drums, bass, vocals, other)
		// Falls back to native C++ onset detection if ONNX models unavailable
	}
}
