#pragma once

#include "CoreMinimal.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "BeatsyncSubsystem.generated.h"

// Forward declaration for AI analyzer
class FAIAudioAnalyzer;

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnAnalysisProgress, float, Progress);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnProcessingProgress, float, Progress);
DECLARE_DYNAMIC_MULTICAST_DELEGATE(FOnAnalysisComplete);
DECLARE_DYNAMIC_MULTICAST_DELEGATE(FOnProcessingComplete);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnError, const FString&, ErrorMessage);

// Waveform sample for visualization (downsampled)
USTRUCT(BlueprintType)
struct FWaveformSample
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadOnly, Category = "BeatSync")
	float LowFreq = 0.0f;    // Bass energy (red in Rekordbox)

	UPROPERTY(BlueprintReadOnly, Category = "BeatSync")
	float MidFreq = 0.0f;    // Mid energy (white/green)

	UPROPERTY(BlueprintReadOnly, Category = "BeatSync")
	float HighFreq = 0.0f;   // High energy (blue/cyan)

	UPROPERTY(BlueprintReadOnly, Category = "BeatSync")
	float Amplitude = 0.0f;  // Overall amplitude
};

USTRUCT(BlueprintType)
struct FBeatData
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadOnly, Category = "BeatSync")
	TArray<float> BeatTimestamps;

	UPROPERTY(BlueprintReadOnly, Category = "BeatSync")
	float BPM = 0.0f;

	UPROPERTY(BlueprintReadOnly, Category = "BeatSync")
	float Duration = 0.0f;

	UPROPERTY(BlueprintReadOnly, Category = "BeatSync")
	int32 BeatCount = 0;

	UPROPERTY(BlueprintReadOnly, Category = "BeatSync")
	TArray<FWaveformSample> WaveformData;  // Downsampled waveform for display

	UPROPERTY(BlueprintReadOnly, Category = "BeatSync")
	int32 SampleRate = 0;

	UPROPERTY(BlueprintReadOnly, Category = "BeatSync")
	FString DebugInfo;  // Debug info for troubleshooting
};

/**
 * Subsystem managing BeatSync audio analysis and video processing
 */
UCLASS()
class TRIPSITTERUE_API UBeatsyncSubsystem : public UGameInstanceSubsystem
{
	GENERATED_BODY()

public:
	virtual void Initialize(FSubsystemCollectionBase& Collection) override;
	virtual void Deinitialize() override;

	/** Analyze an audio file for beats */
	UFUNCTION(BlueprintCallable, Category = "BeatSync")
	bool AnalyzeAudioFile(const FString& FilePath);

	/** Get the current beat data after analysis */
	UFUNCTION(BlueprintCallable, Category = "BeatSync")
	FBeatData GetBeatData() const { return CurrentBeatData; }

	/** Check if analysis is in progress */
	UFUNCTION(BlueprintCallable, Category = "BeatSync")
	bool IsAnalyzing() const { return bIsAnalyzing; }

	/** Check if processing is in progress */
	UFUNCTION(BlueprintCallable, Category = "BeatSync")
	bool IsProcessing() const { return bIsProcessing; }

	/** Process videos with current beat data */
	UFUNCTION(BlueprintCallable, Category = "BeatSync")
	bool ProcessVideos(const TArray<FString>& VideoFiles, const FString& OutputPath, float ClipDuration);

	/** Cancel current operation */
	UFUNCTION(BlueprintCallable, Category = "BeatSync")
	void CancelOperation();

	/** Events */
	UPROPERTY(BlueprintAssignable, Category = "BeatSync|Events")
	FOnAnalysisProgress OnAnalysisProgress;

	UPROPERTY(BlueprintAssignable, Category = "BeatSync|Events")
	FOnProcessingProgress OnProcessingProgress;

	UPROPERTY(BlueprintAssignable, Category = "BeatSync|Events")
	FOnAnalysisComplete OnAnalysisComplete;

	UPROPERTY(BlueprintAssignable, Category = "BeatSync|Events")
	FOnProcessingComplete OnProcessingComplete;

	UPROPERTY(BlueprintAssignable, Category = "BeatSync|Events")
	FOnError OnError;

	/** Get waveform sample at normalized position (0-1) */
	UFUNCTION(BlueprintCallable, Category = "BeatSync")
	FWaveformSample GetWaveformAt(float NormalizedPosition) const;

	/** Check if waveform data is available */
	UFUNCTION(BlueprintCallable, Category = "BeatSync")
	bool HasWaveformData() const { return CurrentBeatData.WaveformData.Num() > 0; }

	/** Check if AI-based beat detection is available */
	UFUNCTION(BlueprintCallable, Category = "BeatSync")
	bool IsAIDetectionAvailable() const { return bAIDetectionAvailable; }

	/** Get the detection method used (for UI display) */
	UFUNCTION(BlueprintCallable, Category = "BeatSync")
	FString GetDetectionMethod() const { return DetectionMethod; }

private:
	FBeatData CurrentBeatData;
	FString CurrentAudioFilePath;  // Store audio path for final mux
	void* AnalyzerHandle = nullptr;
	void* WriterHandle = nullptr;
	bool bIsAnalyzing = false;
	bool bIsProcessing = false;
	bool bCancelRequested = false;

	// AI-based audio analysis (BeatNet + Demucs)
	TUniquePtr<FAIAudioAnalyzer> AIAnalyzer;
	bool bAIDetectionAvailable = false;
	FString DetectionMethod = TEXT("Native");

	/** Initialize AI models (BeatNet, Demucs) */
	void InitializeAIModels();

	/** AI-powered beat detection using BeatNet */
	bool AIBeatDetection(const TArray<float>& MonoSamples, int32 SampleRate, TArray<double>& OutBeats, double& OutBPM, double& OutDuration);

	/** Native C++ beat detection (fallback) */
	bool NativeBeatDetectionMethod(const TArray<float>& MonoSamples, int32 SampleRate, TArray<double>& OutBeats, double& OutBPM, double& OutDuration);

	/** Extract waveform data from audio file */
	bool ExtractWaveform(const FString& FilePath, int32 NumSamples = 2048);

	/** Simple WAV file reader for waveform extraction */
	bool ReadWavFile(const FString& FilePath, TArray<float>& OutSamples, int32& OutSampleRate, int32& OutChannels);
};
