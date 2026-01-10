#pragma once

#include "CoreMinimal.h"

// Forward declarations for NNE types
namespace UE::NNE
{
	class IModelCPU;
	class IModelInstanceCPU;
	struct FTensorBindingCPU;
	class FTensorShape;
}

class UNNEModelData;

/**
 * Audio preprocessing utilities for neural network inference
 * Implements STFT and mel spectrogram computation for BeatNet/Demucs
 */
namespace AudioPreprocessing
{
	/**
	 * Compute Short-Time Fourier Transform (STFT)
	 * @param AudioSamples Input audio samples (mono)
	 * @param FFTSize Size of FFT window (typically 2048)
	 * @param HopSize Hop between frames (typically FFTSize/4)
	 * @param OutMagnitude Output magnitude spectrogram (frames x bins)
	 * @param OutPhase Output phase spectrogram (frames x bins)
	 */
	void ComputeSTFT(
		const TArray<float>& AudioSamples,
		int32 FFTSize,
		int32 HopSize,
		TArray<TArray<float>>& OutMagnitude,
		TArray<TArray<float>>& OutPhase
	);

	/**
	 * Compute Mel Spectrogram from audio samples
	 * @param AudioSamples Input audio samples (mono)
	 * @param SampleRate Audio sample rate
	 * @param NumMelBins Number of mel frequency bins (typically 80)
	 * @param FFTSize Size of FFT window (typically 2048)
	 * @param HopSize Hop between frames (typically 512)
	 * @param OutMelSpectrogram Output mel spectrogram (frames x mels), log-scaled
	 */
	void ComputeMelSpectrogram(
		const TArray<float>& AudioSamples,
		int32 SampleRate,
		int32 NumMelBins,
		int32 FFTSize,
		int32 HopSize,
		TArray<TArray<float>>& OutMelSpectrogram
	);

	/**
	 * Create mel filterbank matrix
	 * @param SampleRate Audio sample rate
	 * @param FFTSize Size of FFT
	 * @param NumMelBins Number of mel bins
	 * @param FMin Minimum frequency (Hz)
	 * @param FMax Maximum frequency (Hz), 0 = Nyquist
	 * @return Mel filterbank matrix (NumMelBins x FFTSize/2+1)
	 */
	TArray<TArray<float>> CreateMelFilterbank(
		int32 SampleRate,
		int32 FFTSize,
		int32 NumMelBins,
		float FMin = 0.0f,
		float FMax = 0.0f
	);

	/**
	 * Convert frequency to mel scale
	 */
	float HzToMel(float Hz);

	/**
	 * Convert mel scale to frequency
	 */
	float MelToHz(float Mel);
}

/**
 * ONNX model wrapper for BeatNet beat detection
 */
class TRIPSITTERUE_API FBeatNetInference
{
public:
	FBeatNetInference();
	~FBeatNetInference();

	/**
	 * Load BeatNet ONNX model
	 * @param ModelPath Path to beatnet.onnx file
	 * @return true if model loaded successfully
	 */
	bool LoadModel(const FString& ModelPath);

	/**
	 * Check if model is loaded and ready
	 */
	bool IsReady() const { return bIsReady; }

	/**
	 * Run beat detection inference
	 * @param MelSpectrogram Input mel spectrogram (frames x mels)
	 * @param OutBeatActivations Output beat activations per frame
	 * @param OutDownbeatActivations Output downbeat activations per frame
	 * @return true if inference successful
	 */
	bool RunInference(
		const TArray<TArray<float>>& MelSpectrogram,
		TArray<float>& OutBeatActivations,
		TArray<float>& OutDownbeatActivations
	);

	/**
	 * Post-process activations to get beat timestamps
	 * @param BeatActivations Raw beat activations from model
	 * @param FrameRate Frames per second (SampleRate / HopSize)
	 * @param Threshold Detection threshold (0-1)
	 * @param OutBeatTimestamps Output beat timestamps in seconds
	 */
	static void ActivationsToBeats(
		const TArray<float>& BeatActivations,
		float FrameRate,
		float Threshold,
		TArray<double>& OutBeatTimestamps
	);

	/**
	 * Estimate BPM from beat timestamps
	 * @param BeatTimestamps Beat timestamps in seconds
	 * @return Estimated BPM (normalized to 70-180 range)
	 */
	static double EstimateBPM(const TArray<double>& BeatTimestamps);

private:
	bool bIsReady = false;
	int32 NumMelBins = 80;
	int32 MaxSequenceLength = 256;

	// NNE model handles
	TSharedPtr<UE::NNE::IModelCPU> Model;
	TSharedPtr<UE::NNE::IModelInstanceCPU> ModelInstance;
};

/**
 * ONNX model wrapper for Demucs stem separation
 */
class TRIPSITTERUE_API FDemucsInference
{
public:
	// Stem indices in output
	enum EStem : int32
	{
		Drums = 0,
		Bass = 1,
		Other = 2,
		Vocals = 3,
		NumStems = 4
	};

	FDemucsInference();
	~FDemucsInference();

	/**
	 * Load Demucs ONNX model
	 * @param ModelPath Path to demucs.onnx file
	 * @return true if model loaded successfully
	 */
	bool LoadModel(const FString& ModelPath);

	/**
	 * Check if model is loaded and ready
	 */
	bool IsReady() const { return bIsReady; }

	/**
	 * Run stem separation inference
	 * @param AudioLeft Left channel samples
	 * @param AudioRight Right channel samples
	 * @param OutStems Output separated stems [4][2][samples] (stems x channels x samples)
	 * @return true if inference successful
	 */
	bool RunInference(
		const TArray<float>& AudioLeft,
		const TArray<float>& AudioRight,
		TArray<TArray<TArray<float>>>& OutStems
	);

	/**
	 * Get a specific stem from the last separation
	 * @param Stem Which stem to get
	 * @param OutLeft Output left channel
	 * @param OutRight Output right channel
	 */
	void GetStem(EStem Stem, TArray<float>& OutLeft, TArray<float>& OutRight) const;

private:
	bool bIsReady = false;
	int32 ChunkSize = 262144;  // ~6 seconds at 44.1kHz

	// NNE model handles
	TSharedPtr<UE::NNE::IModelCPU> Model;
	TSharedPtr<UE::NNE::IModelInstanceCPU> ModelInstance;

	// Cached output from last inference
	TArray<TArray<TArray<float>>> CachedStems;
};

/**
 * Combined AI audio analyzer using BeatNet + Demucs
 */
class TRIPSITTERUE_API FAIAudioAnalyzer
{
public:
	FAIAudioAnalyzer();
	~FAIAudioAnalyzer();

	/**
	 * Initialize the analyzer by loading models
	 * @param BeatNetPath Path to beatnet.onnx
	 * @param DemucsPath Path to demucs.onnx (optional, for stem separation)
	 * @return true if at least BeatNet loaded successfully
	 */
	bool Initialize(const FString& BeatNetPath, const FString& DemucsPath = TEXT(""));

	/**
	 * Analyze audio for beats using BeatNet
	 * @param AudioSamples Mono audio samples
	 * @param SampleRate Audio sample rate
	 * @param OutBeatTimestamps Output beat timestamps
	 * @param OutBPM Output estimated BPM
	 * @return true if analysis successful
	 */
	bool AnalyzeBeats(
		const TArray<float>& AudioSamples,
		int32 SampleRate,
		TArray<double>& OutBeatTimestamps,
		double& OutBPM
	);

	/**
	 * Separate audio into stems using Demucs
	 * @param AudioLeft Left channel
	 * @param AudioRight Right channel
	 * @return true if separation successful
	 */
	bool SeparateStems(
		const TArray<float>& AudioLeft,
		const TArray<float>& AudioRight
	);

	/**
	 * Get drums stem after separation (for beat detection enhancement)
	 */
	void GetDrumsStem(TArray<float>& OutMono) const;

	/**
	 * Check if BeatNet is available
	 */
	bool HasBeatNet() const { return BeatNet.IsReady(); }

	/**
	 * Check if Demucs is available
	 */
	bool HasDemucs() const { return Demucs.IsReady(); }

private:
	FBeatNetInference BeatNet;
	FDemucsInference Demucs;
};
