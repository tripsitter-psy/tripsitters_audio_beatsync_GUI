#include "BeatsyncSubsystem.h"
#include "BeatsyncLoader.h"
#include "Async/Async.h"
#include "Misc/FileHelper.h"
#include "HAL/PlatformFileManager.h"
#include "Misc/Paths.h"

void UBeatsyncSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);

	if (FBeatsyncLoader::IsInitialized())
	{
		AnalyzerHandle = FBeatsyncLoader::CreateAnalyzer();
		WriterHandle = FBeatsyncLoader::CreateVideoWriter();
	}
}

void UBeatsyncSubsystem::Deinitialize()
{
	CancelOperation();

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

// Native C++ beat detection using onset/energy detection
// FORCENOINLINE prevents linker from stripping this during optimization
FORCENOINLINE bool NativeBeatDetection(const TArray<float>& MonoSamples, int32 SampleRate, TArray<double>& OutBeats, double& OutBPM, double& OutDuration)
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

	// Calculate BPM from inter-beat intervals
	if (OutBeats.Num() >= 4)
	{
		TArray<double> Intervals;
		for (int32 i = 1; i < OutBeats.Num(); i++)
		{
			double Interval = OutBeats[i] - OutBeats[i - 1];
			if (Interval > 0.2 && Interval < 2.0) // Filter unreasonable intervals
			{
				Intervals.Add(Interval);
			}
		}

		if (Intervals.Num() > 0)
		{
			// Sort and take median
			Intervals.Sort();
			double MedianInterval = Intervals[Intervals.Num() / 2];

			OutBPM = 60.0 / MedianInterval;

			// Adjust to common tempo range (70-180 BPM)
			while (OutBPM > 180.0) OutBPM /= 2.0;
			while (OutBPM < 70.0) OutBPM *= 2.0;
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

			// Run native beat detection
			bSuccess = NativeBeatDetection(MonoSamples, SampleRate, Beats, BPM, Duration);
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

				// Also extract waveform for visualization
				ExtractWaveform(FilePath, 2048);

				OnAnalysisComplete.Broadcast();
			}
			else
			{
				FString ErrorMsg = FString::Printf(TEXT("Failed: Loaded %d samples at %dHz from %s"),
					LoadedSamples, LoadedSR, *FPaths::GetCleanFilename(FilePath));
				OnError.Broadcast(ErrorMsg);
			}
		});
	});

	return true;
}

bool UBeatsyncSubsystem::ProcessVideos(const TArray<FString>& VideoFiles, const FString& OutputPath, float ClipDuration)
{
	if (!WriterHandle)
	{
		OnError.Broadcast(TEXT("BeatSync backend not initialized"));
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

	bIsProcessing = true;
	bCancelRequested = false;

	// Convert beat timestamps to double array
	TArray<double> BeatTimes;
	for (float Beat : CurrentBeatData.BeatTimestamps)
	{
		BeatTimes.Add(static_cast<double>(Beat));
	}

	AsyncTask(ENamedThreads::AnyBackgroundThreadNormalTask, [this, VideoFiles, OutputPath, ClipDuration, BeatTimes = MoveTemp(BeatTimes)]()
	{
		bool bSuccess = true;
		TArray<FString> ProcessedVideos;

		// Process each video
		int32 TotalVideos = VideoFiles.Num();
		for (int32 i = 0; i < TotalVideos && bSuccess && !bCancelRequested; ++i)
		{
			FString OutputFile = FPaths::Combine(OutputPath, FString::Printf(TEXT("segment_%03d.mp4"), i));

			bSuccess = FBeatsyncLoader::CutVideoAtBeats(WriterHandle, VideoFiles[i], BeatTimes, OutputFile, ClipDuration);

			if (bSuccess)
			{
				ProcessedVideos.Add(OutputFile);
			}

			// Report progress
			float Progress = static_cast<float>(i + 1) / static_cast<float>(TotalVideos);
			AsyncTask(ENamedThreads::GameThread, [this, Progress]()
			{
				OnProcessingProgress.Broadcast(Progress * 0.8f); // Reserve 20% for concatenation
			});
		}

		// Concatenate if successful
		if (bSuccess && !bCancelRequested && ProcessedVideos.Num() > 0)
		{
			FString FinalOutput = FPaths::Combine(OutputPath, TEXT("final_output.mp4"));
			bSuccess = FBeatsyncLoader::ConcatenateVideos(ProcessedVideos, FinalOutput);
		}

		// Return to game thread
		AsyncTask(ENamedThreads::GameThread, [this, bSuccess]()
		{
			bIsProcessing = false;

			if (bCancelRequested)
			{
				return;
			}

			if (bSuccess)
			{
				OnProcessingProgress.Broadcast(1.0f);
				OnProcessingComplete.Broadcast();
			}
			else
			{
				FString Error = WriterHandle ? FBeatsyncLoader::GetVideoWriterLastError(WriterHandle) : TEXT("Unknown error");
				OnError.Broadcast(Error);
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
