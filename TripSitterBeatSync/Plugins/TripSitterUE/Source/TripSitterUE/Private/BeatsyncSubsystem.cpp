#include "BeatsyncSubsystem.h"
#include "BeatsyncLoader.h"
#include "ONNXInference.h"
#include "Async/Async.h"
#include "Misc/FileHelper.h"
#include "HAL/PlatformFileManager.h"
#include "Misc/Paths.h"
#include "HAL/PlatformProcess.h"
#include "Interfaces/IPluginManager.h"
#include <cstdlib>  // for system()

// Native FFmpeg command-line helpers
namespace NativeFFmpeg
{
	static FString LastError;

	static FString FindFFmpegPath()
	{
		// Common FFmpeg locations - check these directly
		TArray<FString> SearchPaths = {
			TEXT("/opt/homebrew/bin/ffmpeg"),  // Mac Apple Silicon (Homebrew)
			TEXT("/usr/local/bin/ffmpeg"),      // Mac Intel (Homebrew) / Linux
			TEXT("/usr/bin/ffmpeg"),            // Linux system
			TEXT("C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe"), // Windows
			TEXT("C:\\ffmpeg\\bin\\ffmpeg.exe"), // Windows alternate
		};

		for (const FString& Path : SearchPaths)
		{
			if (FPaths::FileExists(Path))
			{
				UE_LOG(LogTemp, Log, TEXT("Found FFmpeg at: %s"), *Path);
				return Path;
			}
		}

		LastError = TEXT("FFmpeg not found. Install with: brew install ffmpeg");
		UE_LOG(LogTemp, Error, TEXT("%s"), *LastError);
		return FString();
	}

	static FString GetLastError() { return LastError; }

	static bool CutVideoSegment(const FString& FFmpegPath, const FString& InputVideo,
		double StartTime, double Duration, const FString& OutputVideo)
	{
		if (FFmpegPath.IsEmpty())
		{
			LastError = TEXT("FFmpeg path is empty");
			return false;
		}

		// Build full command line for system() call
		// Use -an to strip audio (we'll add the original audio track later)
		FString FullCommand = FString::Printf(
			TEXT("\"%s\" -y -ss %.3f -i \"%s\" -t %.3f -an -c:v copy -avoid_negative_ts make_zero \"%s\" 2>/dev/null"),
			*FFmpegPath, StartTime, *InputVideo, Duration, *OutputVideo
		);

		UE_LOG(LogTemp, Log, TEXT("FFmpeg cut command: %s"), *FullCommand);

		// Use system() for maximum compatibility
		int32 ReturnCode = system(TCHAR_TO_UTF8(*FullCommand));

		if (ReturnCode != 0)
		{
			LastError = FString::Printf(TEXT("FFmpeg cut failed with code %d"), ReturnCode);
			UE_LOG(LogTemp, Error, TEXT("%s"), *LastError);
			return false;
		}

		// Verify output exists
		if (!FPaths::FileExists(OutputVideo))
		{
			LastError = FString::Printf(TEXT("FFmpeg ran but output not found: %s"), *OutputVideo);
			UE_LOG(LogTemp, Error, TEXT("%s"), *LastError);
			return false;
		}

		UE_LOG(LogTemp, Log, TEXT("FFmpeg cut success: %s"), *OutputVideo);
		return true;
	}

	// Merge silent video with audio track
	static bool MergeVideoWithAudio(const FString& FFmpegPath, const FString& VideoFile,
		const FString& AudioFile, const FString& OutputFile)
	{
		if (FFmpegPath.IsEmpty()) return false;

		// Combine video (no audio) with audio file, trim audio to video length
		FString FullCommand = FString::Printf(
			TEXT("\"%s\" -y -i \"%s\" -i \"%s\" -c:v copy -c:a aac -shortest \"%s\" 2>/dev/null"),
			*FFmpegPath, *VideoFile, *AudioFile, *OutputFile
		);

		UE_LOG(LogTemp, Log, TEXT("FFmpeg merge command: %s"), *FullCommand);

		int32 ReturnCode = system(TCHAR_TO_UTF8(*FullCommand));

		if (ReturnCode != 0)
		{
			LastError = FString::Printf(TEXT("FFmpeg merge failed with code %d"), ReturnCode);
			UE_LOG(LogTemp, Error, TEXT("%s"), *LastError);
			return false;
		}

		if (!FPaths::FileExists(OutputFile))
		{
			LastError = TEXT("FFmpeg merge ran but output not found");
			return false;
		}

		UE_LOG(LogTemp, Log, TEXT("FFmpeg merge success: %s"), *OutputFile);
		return true;
	}

	static bool ConcatenateVideos(const FString& FFmpegPath, const TArray<FString>& InputVideos,
		const FString& OutputVideo, const FString& TempDir)
	{
		if (FFmpegPath.IsEmpty() || InputVideos.Num() == 0) return false;

		// Create concat file list
		FString ConcatFile = FPaths::Combine(TempDir, TEXT("concat_list.txt"));
		FString ConcatContent;
		for (const FString& Video : InputVideos)
		{
			ConcatContent += FString::Printf(TEXT("file '%s'\n"), *Video);
		}

		if (!FFileHelper::SaveStringToFile(ConcatContent, *ConcatFile))
		{
			LastError = FString::Printf(TEXT("Failed to write concat list: %s"), *ConcatFile);
			return false;
		}

		// Re-encode during concatenation to avoid keyframe glitches at cut points
		// Use libx264 with fast preset for reasonable speed, crf 18 for good quality
		FString FullCommand = FString::Printf(
			TEXT("\"%s\" -y -f concat -safe 0 -i \"%s\" -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p \"%s\" 2>/dev/null"),
			*FFmpegPath, *ConcatFile, *OutputVideo
		);

		UE_LOG(LogTemp, Log, TEXT("FFmpeg concat command: %s"), *FullCommand);

		int32 ReturnCode = system(TCHAR_TO_UTF8(*FullCommand));

		// Clean up temp file
		IFileManager::Get().Delete(*ConcatFile);

		if (ReturnCode != 0)
		{
			LastError = FString::Printf(TEXT("FFmpeg concat failed with code %d"), ReturnCode);
			UE_LOG(LogTemp, Error, TEXT("%s"), *LastError);
			return false;
		}

		if (!FPaths::FileExists(OutputVideo))
		{
			LastError = FString::Printf(TEXT("FFmpeg concat ran but output not found: %s"), *OutputVideo);
			return false;
		}

		UE_LOG(LogTemp, Log, TEXT("FFmpeg concat success: %s"), *OutputVideo);
		return true;
	}
}

void UBeatsyncSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);

	if (FBeatsyncLoader::IsInitialized())
	{
		AnalyzerHandle = FBeatsyncLoader::CreateAnalyzer();
		WriterHandle = FBeatsyncLoader::CreateVideoWriter();
	}

	// Initialize AI models for beat detection
	InitializeAIModels();
}

void UBeatsyncSubsystem::InitializeAIModels()
{
	AIAnalyzer = MakeUnique<FAIAudioAnalyzer>();

	// Find ONNX model paths
	// First check the plugin's ThirdParty directory
	FString PluginDir = FPaths::ProjectPluginsDir() / TEXT("TripSitterUE");
	FString ProjectDir = FPaths::ProjectDir();

	// Possible model locations
	TArray<FString> SearchPaths = {
		ProjectDir / TEXT("ThirdParty/onnx_models"),
		PluginDir / TEXT("ThirdParty/onnx_models"),
		ProjectDir / TEXT("Content/Models"),
		FPaths::ProjectContentDir() / TEXT("Models"),
	};

	FString BeatNetPath;
	FString DemucsPath;

	for (const FString& SearchPath : SearchPaths)
	{
		FString TestBeatNet = SearchPath / TEXT("beatnet.onnx");
		FString TestDemucs = SearchPath / TEXT("demucs.onnx");

		if (BeatNetPath.IsEmpty() && FPaths::FileExists(TestBeatNet))
		{
			BeatNetPath = TestBeatNet;
			UE_LOG(LogTemp, Log, TEXT("Found BeatNet model: %s"), *BeatNetPath);
		}

		if (DemucsPath.IsEmpty() && FPaths::FileExists(TestDemucs))
		{
			DemucsPath = TestDemucs;
			UE_LOG(LogTemp, Log, TEXT("Found Demucs model: %s"), *DemucsPath);
		}
	}

	// Try to initialize AI models
	if (!BeatNetPath.IsEmpty())
	{
		bAIDetectionAvailable = AIAnalyzer->Initialize(BeatNetPath, DemucsPath);
		if (bAIDetectionAvailable)
		{
			DetectionMethod = TEXT("BeatNet AI");
			if (AIAnalyzer->HasDemucs())
			{
				DetectionMethod += TEXT(" + Demucs");
			}
			UE_LOG(LogTemp, Log, TEXT("AI beat detection enabled: %s"), *DetectionMethod);
		}
		else
		{
			UE_LOG(LogTemp, Warning, TEXT("Failed to initialize AI models, falling back to native detection"));
			DetectionMethod = TEXT("Native (AI unavailable)");
		}
	}
	else
	{
		UE_LOG(LogTemp, Log, TEXT("ONNX models not found, using native beat detection"));
		UE_LOG(LogTemp, Log, TEXT("To enable AI detection, export models using scripts/export_beatnet_onnx.py"));
		DetectionMethod = TEXT("Native");
	}
}

void UBeatsyncSubsystem::Deinitialize()
{
	CancelOperation();

	// Clean up AI analyzer
	AIAnalyzer.Reset();

	if (AnalyzerHandle)
	{
		FBeatsyncLoader::DestroyAnalyzer(AnalyzerHandle);
		AnalyzerHandle = nullptr;
	}

	if (WriterHandle)
	{
		FBeatsyncLoader::DestroyVideoWriter(WriterHandle);
		WriterHandle = nullptr;
	}

	Super::Deinitialize();
}

// Forward declaration for WAV reading
static bool ReadWavFileForAnalysis(const FString& FilePath, TArray<float>& OutSamples, int32& OutSampleRate, int32& OutChannels);

// AI-powered beat detection wrapper using BeatNet ONNX model
bool UBeatsyncSubsystem::AIBeatDetection(const TArray<float>& MonoSamples, int32 SampleRate, TArray<double>& OutBeats, double& OutBPM, double& OutDuration)
{
	if (!AIAnalyzer.IsValid() || !AIAnalyzer->HasBeatNet())
	{
		return false;
	}

	OutDuration = static_cast<double>(MonoSamples.Num()) / static_cast<double>(SampleRate);

	bool bSuccess = AIAnalyzer->AnalyzeBeats(MonoSamples, SampleRate, OutBeats, OutBPM);

	if (bSuccess)
	{
		UE_LOG(LogTemp, Log, TEXT("AI beat detection: %d beats at %.1f BPM, duration %.1fs"), OutBeats.Num(), OutBPM, OutDuration);
	}

	return bSuccess;
}

// Native C++ beat detection wrapper (fallback when AI not available)
bool UBeatsyncSubsystem::NativeBeatDetectionMethod(const TArray<float>& MonoSamples, int32 SampleRate, TArray<double>& OutBeats, double& OutBPM, double& OutDuration)
{
	return NativeBeatDetection(MonoSamples, SampleRate, OutBeats, OutBPM, OutDuration);
}

// Native C++ beat detection using onset/energy detection
// FORCENOINLINE prevents linker from stripping this during optimization
FORCENOINLINE static bool NativeBeatDetection(const TArray<float>& MonoSamples, int32 SampleRate, TArray<double>& OutBeats, double& OutBPM, double& OutDuration)
{
	if (MonoSamples.Num() == 0 || SampleRate == 0)
	{
		return false;
	}

	OutDuration = static_cast<double>(MonoSamples.Num()) / static_cast<double>(SampleRate);

	// Parameters for onset detection
	const int32 HopSize = SampleRate / 100;  // 10ms hops
	const int32 WindowSize = SampleRate / 10; // 100ms window
	const int32 NumFrames = MonoSamples.Num() / HopSize;

	if (NumFrames < 10)
	{
		UE_LOG(LogTemp, Warning, TEXT("Audio too short for beat detection"));
		return false;
	}

	// Calculate RMS energy for each frame
	TArray<float> Energy;
	Energy.Reserve(NumFrames);

	for (int32 Frame = 0; Frame < NumFrames; Frame++)
	{
		int32 StartSample = Frame * HopSize;
		int32 EndSample = FMath::Min(StartSample + WindowSize, MonoSamples.Num());

		float FrameEnergy = 0.0f;
		for (int32 i = StartSample; i < EndSample; i++)
		{
			FrameEnergy += MonoSamples[i] * MonoSamples[i];
		}
		FrameEnergy = FMath::Sqrt(FrameEnergy / FMath::Max(1, EndSample - StartSample));
		Energy.Add(FrameEnergy);
	}

	// Calculate onset strength (energy difference, half-wave rectified)
	TArray<float> OnsetStrength;
	OnsetStrength.Reserve(NumFrames);
	OnsetStrength.Add(0.0f);

	for (int32 i = 1; i < Energy.Num(); i++)
	{
		float Diff = Energy[i] - Energy[i - 1];
		OnsetStrength.Add(FMath::Max(0.0f, Diff));
	}

	// Adaptive threshold using local mean
	const int32 MedianWindow = SampleRate / HopSize; // ~1 second window
	TArray<float> Threshold;
	Threshold.Reserve(OnsetStrength.Num());

	for (int32 i = 0; i < OnsetStrength.Num(); i++)
	{
		int32 WinStart = FMath::Max(0, i - MedianWindow / 2);
		int32 WinEnd = FMath::Min(OnsetStrength.Num(), i + MedianWindow / 2);

		float Sum = 0.0f;
		float MaxVal = 0.0f;
		for (int32 j = WinStart; j < WinEnd; j++)
		{
			Sum += OnsetStrength[j];
			MaxVal = FMath::Max(MaxVal, OnsetStrength[j]);
		}
		float LocalMean = Sum / FMath::Max(1, WinEnd - WinStart);
		Threshold.Add(LocalMean * 1.3f + MaxVal * 0.1f);
	}

	// Pick peaks above threshold with minimum distance
	TArray<int32> OnsetFrames;
	const int32 MinPeakDistance = SampleRate / HopSize / 5; // Minimum ~200ms between beats

	for (int32 i = 2; i < OnsetStrength.Num() - 2; i++)
	{
		bool bIsPeak = OnsetStrength[i] > Threshold[i] &&
			OnsetStrength[i] > OnsetStrength[i - 1] &&
			OnsetStrength[i] > OnsetStrength[i + 1] &&
			OnsetStrength[i] > OnsetStrength[i - 2] &&
			OnsetStrength[i] > OnsetStrength[i + 2];

		if (bIsPeak)
		{
			if (OnsetFrames.Num() == 0 || (i - OnsetFrames.Last()) >= MinPeakDistance)
			{
				OnsetFrames.Add(i);
			}
		}
	}

	// Convert frames to timestamps
	OutBeats.Empty(OnsetFrames.Num());
	for (int32 Frame : OnsetFrames)
	{
		double TimeStamp = static_cast<double>(Frame * HopSize) / static_cast<double>(SampleRate);
		OutBeats.Add(TimeStamp);
	}

	// Calculate BPM from inter-beat intervals with tempo histogram
	if (OutBeats.Num() >= 4)
	{
		// Collect all valid intervals
		TArray<double> Intervals;
		for (int32 i = 1; i < OutBeats.Num(); i++)
		{
			double Interval = OutBeats[i] - OutBeats[i - 1];
			if (Interval > 0.15 && Interval < 2.0)
			{
				Intervals.Add(Interval);
			}
		}

		if (Intervals.Num() > 0)
		{
			// Create tempo histogram - vote for tempo bins
			TMap<int32, int32> TempoVotes;
			for (double Interval : Intervals)
			{
				double RawBPM = 60.0 / Interval;

				// Normalize to 70-180 range
				while (RawBPM > 180.0) RawBPM /= 2.0;
				while (RawBPM < 70.0) RawBPM *= 2.0;

				// Round to nearest integer BPM and vote
				int32 BPMBin = FMath::RoundToInt(RawBPM);
				TempoVotes.FindOrAdd(BPMBin)++;
			}

			// Find the BPM with most votes
			int32 BestBPM = 120;
			int32 MaxVotes = 0;
			for (auto& Pair : TempoVotes)
			{
				if (Pair.Value > MaxVotes)
				{
					MaxVotes = Pair.Value;
					BestBPM = Pair.Key;
				}
			}

			// Check neighboring bins and find weighted average for more precision
			int32 VotesM1 = TempoVotes.FindRef(BestBPM - 1);
			int32 VotesP1 = TempoVotes.FindRef(BestBPM + 1);
			int32 VotesCenter = TempoVotes.FindRef(BestBPM);

			// Weighted average of the peak and neighbors for sub-BPM precision
			double WeightedBPM = static_cast<double>(BestBPM);
			int32 TotalVotes = VotesM1 + VotesCenter + VotesP1;
			if (TotalVotes > 0)
			{
				WeightedBPM = (static_cast<double>(BestBPM - 1) * VotesM1 +
							   static_cast<double>(BestBPM) * VotesCenter +
							   static_cast<double>(BestBPM + 1) * VotesP1) / TotalVotes;
			}

			// Round to 1 decimal place
			OutBPM = FMath::RoundToDouble(WeightedBPM * 10.0) / 10.0;

			UE_LOG(LogTemp, Log, TEXT("Tempo histogram: peak=%d BPM (votes=%d), weighted=%.1f BPM"),
				BestBPM, MaxVotes, OutBPM);
		}
		else
		{
			OutBPM = 120.0;
		}
	}
	else
	{
		OutBPM = 120.0;
	}

	UE_LOG(LogTemp, Log, TEXT("Native beat detection: %d beats at %.1f BPM, duration %.1fs"), OutBeats.Num(), OutBPM, OutDuration);
	return OutBeats.Num() > 0;
}

// Read WAV file for analysis (simplified version)
static bool ReadWavFileForAnalysis(const FString& FilePath, TArray<float>& OutSamples, int32& OutSampleRate, int32& OutChannels)
{
	TArray<uint8> FileData;
	if (!FFileHelper::LoadFileToArray(FileData, *FilePath))
	{
		UE_LOG(LogTemp, Error, TEXT("Failed to load audio file: %s"), *FilePath);
		return false;
	}

	if (FileData.Num() < 44)
	{
		UE_LOG(LogTemp, Error, TEXT("File too small to be valid WAV"));
		return false;
	}

	// Parse WAV header
	const uint8* Data = FileData.GetData();

	// Check RIFF header
	if (Data[0] != 'R' || Data[1] != 'I' || Data[2] != 'F' || Data[3] != 'F')
	{
		UE_LOG(LogTemp, Error, TEXT("Not a valid WAV file (no RIFF header)"));
		return false;
	}

	// Check WAVE format
	if (Data[8] != 'W' || Data[9] != 'A' || Data[10] != 'V' || Data[11] != 'E')
	{
		UE_LOG(LogTemp, Error, TEXT("Not a valid WAV file (no WAVE format)"));
		return false;
	}

	// Find fmt chunk
	int32 Offset = 12;
	int32 FmtOffset = -1;
	int32 DataOffset = -1;
	int32 DataSize = 0;

	while (Offset < FileData.Num() - 8)
	{
		FString ChunkID = FString::Printf(TEXT("%c%c%c%c"), Data[Offset], Data[Offset+1], Data[Offset+2], Data[Offset+3]);
		int32 ChunkSize = *reinterpret_cast<const int32*>(&Data[Offset + 4]);

		if (ChunkID == TEXT("fmt "))
		{
			FmtOffset = Offset + 8;
		}
		else if (ChunkID == TEXT("data"))
		{
			DataOffset = Offset + 8;
			DataSize = ChunkSize;
			break;
		}

		Offset += 8 + ChunkSize;
		if (ChunkSize % 2 == 1) Offset++; // Padding
	}

	if (FmtOffset < 0 || DataOffset < 0)
	{
		UE_LOG(LogTemp, Error, TEXT("Invalid WAV structure"));
		return false;
	}

	// Read format info
	int16 AudioFormat = *reinterpret_cast<const int16*>(&Data[FmtOffset]);
	OutChannels = *reinterpret_cast<const int16*>(&Data[FmtOffset + 2]);
	OutSampleRate = *reinterpret_cast<const int32*>(&Data[FmtOffset + 4]);
	int16 BitsPerSample = *reinterpret_cast<const int16*>(&Data[FmtOffset + 14]);

	if (AudioFormat != 1) // PCM
	{
		UE_LOG(LogTemp, Error, TEXT("Only PCM WAV supported, got format %d"), AudioFormat);
		return false;
	}

	// Read samples
	int32 BytesPerSample = BitsPerSample / 8;
	int32 NumSamples = DataSize / BytesPerSample;

	OutSamples.Empty(NumSamples);

	for (int32 i = 0; i < NumSamples && (DataOffset + i * BytesPerSample) < FileData.Num(); i++)
	{
		float Sample = 0.0f;
		if (BitsPerSample == 16)
		{
			int16 RawSample = *reinterpret_cast<const int16*>(&Data[DataOffset + i * 2]);
			Sample = static_cast<float>(RawSample) / 32768.0f;
		}
		else if (BitsPerSample == 24)
		{
			int32 RawSample = (Data[DataOffset + i * 3] | (Data[DataOffset + i * 3 + 1] << 8) | (Data[DataOffset + i * 3 + 2] << 16));
			if (RawSample & 0x800000) RawSample |= 0xFF000000; // Sign extend
			Sample = static_cast<float>(RawSample) / 8388608.0f;
		}
		else if (BitsPerSample == 32)
		{
			int32 RawSample = *reinterpret_cast<const int32*>(&Data[DataOffset + i * 4]);
			Sample = static_cast<float>(RawSample) / 2147483648.0f;
		}
		OutSamples.Add(Sample);
	}

	UE_LOG(LogTemp, Log, TEXT("Loaded WAV: %d samples, %d Hz, %d channels, %d bits"), OutSamples.Num(), OutSampleRate, OutChannels, BitsPerSample);
	return OutSamples.Num() > 0;
}

bool UBeatsyncSubsystem::AnalyzeAudioFile(const FString& FilePath)
{
	if (bIsAnalyzing || bIsProcessing)
	{
		OnError.Broadcast(TEXT("Operation already in progress"));
		return false;
	}

	// Store audio file path for later use in video muxing
	CurrentAudioFilePath = FilePath;

	bIsAnalyzing = true;
	bCancelRequested = false;

	// Run analysis on background thread
	AsyncTask(ENamedThreads::AnyBackgroundThreadNormalTask, [this, FilePath]()
	{
		TArray<double> Beats;
		double BPM = 0.0;
		double Duration = 0.0;
		bool bSuccess = false;

		// Load audio file
		TArray<float> RawSamples;
		int32 SampleRate = 0;
		int32 Channels = 0;

		if (ReadWavFileForAnalysis(FilePath, RawSamples, SampleRate, Channels))
		{
			// Convert stereo to mono if needed
			TArray<float> MonoSamples;
			if (Channels == 2)
			{
				MonoSamples.Reserve(RawSamples.Num() / 2);
				for (int32 i = 0; i < RawSamples.Num() - 1; i += 2)
				{
					MonoSamples.Add((RawSamples[i] + RawSamples[i + 1]) * 0.5f);
				}
			}
			else
			{
				MonoSamples = MoveTemp(RawSamples);
			}

			// Try AI beat detection first (BeatNet), fall back to native if unavailable
			if (bAIDetectionAvailable && AIAnalyzer.IsValid())
			{
				bSuccess = AIAnalyzer->AnalyzeBeats(MonoSamples, SampleRate, Beats, BPM);
				Duration = static_cast<double>(MonoSamples.Num()) / static_cast<double>(SampleRate);
				if (bSuccess)
				{
					UE_LOG(LogTemp, Log, TEXT("AI beat detection succeeded: %d beats at %.1f BPM"), Beats.Num(), BPM);
				}
				else
				{
					UE_LOG(LogTemp, Warning, TEXT("AI beat detection failed, falling back to native"));
				}
			}

			// Fall back to native detection if AI failed or unavailable
			if (!bSuccess)
			{
				bSuccess = NativeBeatDetection(MonoSamples, SampleRate, Beats, BPM, Duration);
			}
		}

		int32 LoadedSamples = RawSamples.Num();
		int32 LoadedSR = SampleRate;

		// Return to game thread to update state and broadcast events
		AsyncTask(ENamedThreads::GameThread, [this, bSuccess, Beats = MoveTemp(Beats), BPM, Duration, FilePath, LoadedSamples, LoadedSR]()
		{
			bIsAnalyzing = false;

			if (bCancelRequested)
			{
				return;
			}

			if (bSuccess && BPM > 0 && Beats.Num() > 0)
			{
				CurrentBeatData.BeatTimestamps.Empty(Beats.Num());
				for (double Beat : Beats)
				{
					CurrentBeatData.BeatTimestamps.Add(static_cast<float>(Beat));
				}
				CurrentBeatData.BPM = static_cast<float>(BPM);
				CurrentBeatData.Duration = static_cast<float>(Duration);
				CurrentBeatData.BeatCount = Beats.Num();

				// Store debug info in status (include detection method)
				CurrentBeatData.DebugInfo = FString::Printf(TEXT("[%s] %d samples @ %dHz = %.1fs"),
					*DetectionMethod, LoadedSamples, LoadedSR, Duration);

				// Also extract waveform for visualization
				ExtractWaveform(FilePath, 2048);

				OnAnalysisComplete.Broadcast();
			}
			else
			{
				FString ErrorMsg = FString::Printf(TEXT("[%s] Failed: %d samples @ %dHz from %s"),
					*DetectionMethod, LoadedSamples, LoadedSR, *FPaths::GetCleanFilename(FilePath));
				OnError.Broadcast(ErrorMsg);
			}
		});
	});

	return true;
}

bool UBeatsyncSubsystem::ProcessVideos(const TArray<FString>& VideoFiles, const FString& OutputPath, float ClipDuration)
{
	// ALWAYS prefer native FFmpeg if available (more reliable than external library)
	FString FFmpegPath = NativeFFmpeg::FindFFmpegPath();

	if (FFmpegPath.IsEmpty())
	{
		OnError.Broadcast(TEXT("FFmpeg not found - install with: brew install ffmpeg"));
		return false;
	}

	if (bIsAnalyzing || bIsProcessing)
	{
		OnError.Broadcast(TEXT("Operation already in progress"));
		return false;
	}

	if (CurrentBeatData.BeatCount == 0)
	{
		OnError.Broadcast(TEXT("No beat data available - analyze audio first"));
		return false;
	}

	if (VideoFiles.Num() == 0)
	{
		OnError.Broadcast(TEXT("No video files provided"));
		return false;
	}

	if (CurrentAudioFilePath.IsEmpty())
	{
		OnError.Broadcast(TEXT("No audio file path stored - analyze audio first"));
		return false;
	}

	bIsProcessing = true;
	bCancelRequested = false;

	// Calculate how many beats we need based on desired output duration
	int32 TotalBeats = CurrentBeatData.BeatCount;
	float AudioDuration = CurrentBeatData.Duration;

	UE_LOG(LogTemp, Log, TEXT("ProcessVideos: %d videos, %d beats, clip duration %.2fs, audio: %s"),
		VideoFiles.Num(), TotalBeats, ClipDuration, *CurrentAudioFilePath);

	// Store audio path for lambda capture
	FString AudioPath = CurrentAudioFilePath;

	AsyncTask(ENamedThreads::AnyBackgroundThreadNormalTask, [this, VideoFiles, OutputPath, ClipDuration, TotalBeats, AudioDuration, FFmpegPath, AudioPath]()
	{
		bool bSuccess = true;
		TArray<FString> ProcessedVideos;
		FString ErrorMsg;

		// Ensure output directory exists
		IFileManager::Get().MakeDirectory(*OutputPath, true);

		int32 NumVideos = VideoFiles.Num();

		// Simple random seed based on time
		uint32 RandomSeed = static_cast<uint32>(FPlatformTime::Cycles());

		// For each beat, cut a segment from a cycling video
		// The segment is cut from a random position in the source video (NOT at beat timestamps)
		for (int32 BeatIdx = 0; BeatIdx < TotalBeats && bSuccess && !bCancelRequested; ++BeatIdx)
		{
			// Cycle through videos: beat 0 -> video 0, beat 1 -> video 1, etc.
			int32 VideoIdx = BeatIdx % NumVideos;
			const FString& InputVideo = VideoFiles[VideoIdx];

			// Pick a random start time within the source video
			// Assume source videos are at least 10 seconds long, start from random point
			// Use a simple LCG random for reproducibility
			RandomSeed = RandomSeed * 1103515245 + 12345;
			float RandomFactor = static_cast<float>((RandomSeed >> 16) & 0x7FFF) / 32767.0f;
			double MaxStartTime = 5.0;  // Assume videos are at least 5+clipDuration seconds
			double StartTime = RandomFactor * MaxStartTime;

			FString OutputFile = FPaths::Combine(OutputPath, FString::Printf(TEXT("clip_%03d.mp4"), BeatIdx));

			bSuccess = NativeFFmpeg::CutVideoSegment(FFmpegPath, InputVideo, StartTime, ClipDuration, OutputFile);

			if (bSuccess && FPaths::FileExists(OutputFile))
			{
				ProcessedVideos.Add(OutputFile);
			}
			else if (!bSuccess)
			{
				ErrorMsg = FString::Printf(TEXT("Failed to cut video %d at %.2fs"), VideoIdx, StartTime);
				UE_LOG(LogTemp, Error, TEXT("%s"), *ErrorMsg);
			}

			// Report progress (cutting is 60% of work)
			if (BeatIdx % 10 == 0)
			{
				float Progress = static_cast<float>(BeatIdx) / static_cast<float>(TotalBeats) * 0.6f;
				AsyncTask(ENamedThreads::GameThread, [this, Progress]()
				{
					OnProcessingProgress.Broadcast(Progress);
				});
			}
		}

		// Concatenate all silent video clips
		FString SilentVideo;
		if (bSuccess && !bCancelRequested && ProcessedVideos.Num() > 0)
		{
			SilentVideo = FPaths::Combine(OutputPath, TEXT("silent_concat.mp4"));

			bSuccess = NativeFFmpeg::ConcatenateVideos(FFmpegPath, ProcessedVideos, SilentVideo, OutputPath);

			if (!bSuccess)
			{
				ErrorMsg = TEXT("Failed to concatenate video segments");
			}

			// Report progress (concat is 20% of work)
			AsyncTask(ENamedThreads::GameThread, [this]()
			{
				OnProcessingProgress.Broadcast(0.8f);
			});
		}

		// Merge with original audio
		FString FinalOutput;
		if (bSuccess && !bCancelRequested && FPaths::FileExists(SilentVideo))
		{
			FinalOutput = FPaths::Combine(OutputPath, TEXT("final_beatsync.mp4"));

			bSuccess = NativeFFmpeg::MergeVideoWithAudio(FFmpegPath, SilentVideo, AudioPath, FinalOutput);

			if (!bSuccess)
			{
				ErrorMsg = TEXT("Failed to merge video with audio");
			}

			// Clean up silent video
			IFileManager::Get().Delete(*SilentVideo);
		}

		// Clean up temporary clip files
		for (const FString& TempFile : ProcessedVideos)
		{
			IFileManager::Get().Delete(*TempFile);
		}

		// Also clean up concat list if it exists
		FString ConcatList = FPaths::Combine(OutputPath, TEXT("concat_list.txt"));
		IFileManager::Get().Delete(*ConcatList);

		// Return to game thread
		AsyncTask(ENamedThreads::GameThread, [this, bSuccess, FinalOutput, ErrorMsg]()
		{
			bIsProcessing = false;

			if (bCancelRequested)
			{
				return;
			}

			if (bSuccess && FPaths::FileExists(FinalOutput))
			{
				OnProcessingProgress.Broadcast(1.0f);
				OnProcessingComplete.Broadcast();
				UE_LOG(LogTemp, Log, TEXT("Video processing complete: %s"), *FinalOutput);
			}
			else
			{
				FString Error = !ErrorMsg.IsEmpty() ? ErrorMsg : NativeFFmpeg::GetLastError();
				if (Error.IsEmpty())
				{
					Error = TEXT("Video processing failed - no output created");
				}
				OnError.Broadcast(Error);
				UE_LOG(LogTemp, Error, TEXT("Video processing failed: %s"), *Error);
			}
		});
	});

	return true;
}

void UBeatsyncSubsystem::CancelOperation()
{
	bCancelRequested = true;
}

FWaveformSample UBeatsyncSubsystem::GetWaveformAt(float NormalizedPosition) const
{
	FWaveformSample Result;
	if (CurrentBeatData.WaveformData.Num() == 0)
	{
		return Result;
	}

	// Clamp and get index
	NormalizedPosition = FMath::Clamp(NormalizedPosition, 0.0f, 1.0f);
	int32 Index = FMath::FloorToInt(NormalizedPosition * (CurrentBeatData.WaveformData.Num() - 1));
	Index = FMath::Clamp(Index, 0, CurrentBeatData.WaveformData.Num() - 1);

	return CurrentBeatData.WaveformData[Index];
}

bool UBeatsyncSubsystem::ReadWavFile(const FString& FilePath, TArray<float>& OutSamples, int32& OutSampleRate, int32& OutChannels)
{
	TArray<uint8> FileData;
	if (!FFileHelper::LoadFileToArray(FileData, *FilePath))
	{
		UE_LOG(LogTemp, Error, TEXT("Failed to load WAV file: %s"), *FilePath);
		return false;
	}

	if (FileData.Num() < 44)
	{
		UE_LOG(LogTemp, Error, TEXT("WAV file too small: %s"), *FilePath);
		return false;
	}

	// Parse WAV header
	const uint8* Data = FileData.GetData();

	// Check RIFF header
	if (Data[0] != 'R' || Data[1] != 'I' || Data[2] != 'F' || Data[3] != 'F')
	{
		UE_LOG(LogTemp, Error, TEXT("Not a valid WAV file (missing RIFF): %s"), *FilePath);
		return false;
	}

	// Check WAVE format
	if (Data[8] != 'W' || Data[9] != 'A' || Data[10] != 'V' || Data[11] != 'E')
	{
		UE_LOG(LogTemp, Error, TEXT("Not a valid WAV file (missing WAVE): %s"), *FilePath);
		return false;
	}

	// Find fmt chunk
	int32 Pos = 12;
	int32 FmtPos = -1;
	int32 DataPos = -1;
	int32 DataSize = 0;

	while (Pos < FileData.Num() - 8)
	{
		FString ChunkID = FString::Printf(TEXT("%c%c%c%c"), Data[Pos], Data[Pos+1], Data[Pos+2], Data[Pos+3]);
		int32 ChunkSize = *reinterpret_cast<const int32*>(&Data[Pos + 4]);

		if (ChunkID == TEXT("fmt "))
		{
			FmtPos = Pos + 8;
		}
		else if (ChunkID == TEXT("data"))
		{
			DataPos = Pos + 8;
			DataSize = ChunkSize;
			break;
		}

		Pos += 8 + ChunkSize;
		if (ChunkSize % 2 == 1) Pos++; // Padding
	}

	if (FmtPos < 0 || DataPos < 0)
	{
		UE_LOG(LogTemp, Error, TEXT("Invalid WAV structure: %s"), *FilePath);
		return false;
	}

	// Parse format
	int16 AudioFormat = *reinterpret_cast<const int16*>(&Data[FmtPos]);
	OutChannels = *reinterpret_cast<const int16*>(&Data[FmtPos + 2]);
	OutSampleRate = *reinterpret_cast<const int32*>(&Data[FmtPos + 4]);
	int16 BitsPerSample = *reinterpret_cast<const int16*>(&Data[FmtPos + 14]);

	if (AudioFormat != 1) // PCM
	{
		UE_LOG(LogTemp, Error, TEXT("Unsupported WAV format (only PCM supported): %s"), *FilePath);
		return false;
	}

	// Read samples
	int32 BytesPerSample = BitsPerSample / 8;
	int32 NumSamples = DataSize / BytesPerSample;
	OutSamples.Reserve(NumSamples);

	const uint8* SampleData = &Data[DataPos];

	for (int32 i = 0; i < NumSamples && (DataPos + i * BytesPerSample) < FileData.Num(); i++)
	{
		float Sample = 0.0f;

		if (BitsPerSample == 16)
		{
			int16 RawSample = *reinterpret_cast<const int16*>(&SampleData[i * BytesPerSample]);
			Sample = static_cast<float>(RawSample) / 32768.0f;
		}
		else if (BitsPerSample == 24)
		{
			int32 RawSample = (SampleData[i * 3] | (SampleData[i * 3 + 1] << 8) | (SampleData[i * 3 + 2] << 16));
			if (RawSample & 0x800000) RawSample |= 0xFF000000; // Sign extend
			Sample = static_cast<float>(RawSample) / 8388608.0f;
		}
		else if (BitsPerSample == 32)
		{
			int32 RawSample = *reinterpret_cast<const int32*>(&SampleData[i * BytesPerSample]);
			Sample = static_cast<float>(RawSample) / 2147483648.0f;
		}
		else if (BitsPerSample == 8)
		{
			Sample = (static_cast<float>(SampleData[i]) - 128.0f) / 128.0f;
		}

		OutSamples.Add(Sample);
	}

	UE_LOG(LogTemp, Log, TEXT("Loaded WAV: %d samples, %d Hz, %d channels, %d bits"), OutSamples.Num(), OutSampleRate, OutChannels, BitsPerSample);
	return true;
}

bool UBeatsyncSubsystem::ExtractWaveform(const FString& FilePath, int32 NumSamples)
{
	TArray<float> RawSamples;
	int32 SampleRate, Channels;

	// Only WAV supported for now - other formats would need a decoder
	FString Extension = FPaths::GetExtension(FilePath).ToLower();
	if (Extension != TEXT("wav"))
	{
		UE_LOG(LogTemp, Warning, TEXT("Waveform extraction only supports WAV files. Using beat-based simulation for: %s"), *FilePath);
		return false;
	}

	if (!ReadWavFile(FilePath, RawSamples, SampleRate, Channels))
	{
		return false;
	}

	CurrentBeatData.SampleRate = SampleRate;

	// Convert stereo to mono if needed
	TArray<float> MonoSamples;
	if (Channels == 2)
	{
		MonoSamples.Reserve(RawSamples.Num() / 2);
		for (int32 i = 0; i < RawSamples.Num() - 1; i += 2)
		{
			MonoSamples.Add((RawSamples[i] + RawSamples[i + 1]) * 0.5f);
		}
	}
	else
	{
		MonoSamples = MoveTemp(RawSamples);
	}

	// Downsample for display
	int32 SamplesPerBucket = MonoSamples.Num() / NumSamples;
	if (SamplesPerBucket < 1) SamplesPerBucket = 1;

	CurrentBeatData.WaveformData.Empty(NumSamples);

	// Simple frequency band estimation using zero-crossing rate and amplitude
	for (int32 i = 0; i < NumSamples; i++)
	{
		int32 StartIdx = i * SamplesPerBucket;
		int32 EndIdx = FMath::Min(StartIdx + SamplesPerBucket, MonoSamples.Num());

		float MaxAmp = 0.0f;
		float SumAmp = 0.0f;
		int32 ZeroCrossings = 0;
		float PrevSample = 0.0f;

		// Simple RMS-like energy calculation with zero-crossing for frequency estimation
		for (int32 j = StartIdx; j < EndIdx; j++)
		{
			float Sample = MonoSamples[j];
			float AbsSample = FMath::Abs(Sample);
			MaxAmp = FMath::Max(MaxAmp, AbsSample);
			SumAmp += AbsSample * AbsSample;

			// Count zero crossings (rough frequency indicator)
			if (j > StartIdx && ((Sample > 0 && PrevSample <= 0) || (Sample <= 0 && PrevSample > 0)))
			{
				ZeroCrossings++;
			}
			PrevSample = Sample;
		}

		float RMSAmp = FMath::Sqrt(SumAmp / FMath::Max(1, EndIdx - StartIdx));
		float NormalizedCrossings = static_cast<float>(ZeroCrossings) / FMath::Max(1, EndIdx - StartIdx);

		FWaveformSample WaveSample;
		WaveSample.Amplitude = FMath::Clamp(RMSAmp * 3.0f, 0.0f, 1.0f); // Boost for visibility

		// Estimate frequency bands based on zero-crossing rate
		// Low ZCR = more bass, High ZCR = more treble
		float FreqFactor = FMath::Clamp(NormalizedCrossings * 10.0f, 0.0f, 1.0f);

		// Bass is high when amplitude is high but zero crossings are low
		WaveSample.LowFreq = WaveSample.Amplitude * (1.0f - FreqFactor * 0.7f);
		// Mids are always present
		WaveSample.MidFreq = WaveSample.Amplitude * 0.5f;
		// Highs correlate with zero crossings
		WaveSample.HighFreq = WaveSample.Amplitude * FreqFactor;

		CurrentBeatData.WaveformData.Add(WaveSample);
	}

	UE_LOG(LogTemp, Log, TEXT("Extracted waveform: %d samples for display"), CurrentBeatData.WaveformData.Num());
	return true;
}
