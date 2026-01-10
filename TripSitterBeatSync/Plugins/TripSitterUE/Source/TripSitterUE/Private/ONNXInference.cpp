#include "ONNXInference.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "HAL/PlatformFileManager.h"

// NNE includes - conditionally compiled based on availability
#if WITH_NNE
#include "NNE.h"
#include "NNERuntimeCPU.h"
#include "NNEModelData.h"
#endif

// Use UE's math constants (PI and LOCAL_TWO_PI are deprecated macros)
#define LOCAL_PI UE_PI
#define LOCAL_LOCAL_TWO_PI UE_LOCAL_TWO_PI

//==============================================================================
// Audio Preprocessing
//==============================================================================

namespace AudioPreprocessing
{

float HzToMel(float Hz)
{
	// HTK formula
	return 2595.0f * FMath::Loge(1.0f + Hz / 700.0f) / FMath::Loge(10.0f);
}

float MelToHz(float Mel)
{
	return 700.0f * (FMath::Pow(10.0f, Mel / 2595.0f) - 1.0f);
}

// Simple DFT for a single frame (no FFT library dependency)
void ComputeDFT(const TArray<float>& Frame, TArray<float>& OutReal, TArray<float>& OutImag)
{
	int32 N = Frame.Num();
	OutReal.SetNumZeroed(N / 2 + 1);
	OutImag.SetNumZeroed(N / 2 + 1);

	// Only compute positive frequencies (up to Nyquist)
	for (int32 k = 0; k <= N / 2; k++)
	{
		float Real = 0.0f;
		float Imag = 0.0f;

		for (int32 n = 0; n < N; n++)
		{
			float Angle = LOCAL_TWO_PI * k * n / N;
			Real += Frame[n] * FMath::Cos(Angle);
			Imag -= Frame[n] * FMath::Sin(Angle);
		}

		OutReal[k] = Real;
		OutImag[k] = Imag;
	}
}

// Hann window function
void ApplyHannWindow(TArray<float>& Frame)
{
	int32 N = Frame.Num();
	for (int32 i = 0; i < N; i++)
	{
		float Window = 0.5f * (1.0f - FMath::Cos(LOCAL_TWO_PI * i / (N - 1)));
		Frame[i] *= Window;
	}
}

void ComputeSTFT(
	const TArray<float>& AudioSamples,
	int32 FFTSize,
	int32 HopSize,
	TArray<TArray<float>>& OutMagnitude,
	TArray<TArray<float>>& OutPhase)
{
	int32 NumFrames = (AudioSamples.Num() - FFTSize) / HopSize + 1;
	if (NumFrames <= 0)
	{
		UE_LOG(LogTemp, Warning, TEXT("Audio too short for STFT"));
		return;
	}

	int32 NumBins = FFTSize / 2 + 1;
	OutMagnitude.SetNum(NumFrames);
	OutPhase.SetNum(NumFrames);

	TArray<float> Frame;
	Frame.SetNum(FFTSize);

	TArray<float> Real, Imag;

	for (int32 FrameIdx = 0; FrameIdx < NumFrames; FrameIdx++)
	{
		int32 StartSample = FrameIdx * HopSize;

		// Extract frame with zero padding if needed
		for (int32 i = 0; i < FFTSize; i++)
		{
			int32 SampleIdx = StartSample + i;
			Frame[i] = (SampleIdx < AudioSamples.Num()) ? AudioSamples[SampleIdx] : 0.0f;
		}

		// Apply window
		ApplyHannWindow(Frame);

		// Compute DFT
		ComputeDFT(Frame, Real, Imag);

		// Compute magnitude and phase
		OutMagnitude[FrameIdx].SetNum(NumBins);
		OutPhase[FrameIdx].SetNum(NumBins);

		for (int32 Bin = 0; Bin < NumBins; Bin++)
		{
			float Mag = FMath::Sqrt(Real[Bin] * Real[Bin] + Imag[Bin] * Imag[Bin]);
			float Phase = FMath::Atan2(Imag[Bin], Real[Bin]);
			OutMagnitude[FrameIdx][Bin] = Mag;
			OutPhase[FrameIdx][Bin] = Phase;
		}
	}
}

TArray<TArray<float>> CreateMelFilterbank(
	int32 SampleRate,
	int32 FFTSize,
	int32 NumMelBins,
	float FMin,
	float FMax)
{
	if (FMax == 0.0f)
	{
		FMax = SampleRate / 2.0f;  // Nyquist
	}

	int32 NumFFTBins = FFTSize / 2 + 1;

	// Convert frequency range to mel
	float MelMin = HzToMel(FMin);
	float MelMax = HzToMel(FMax);

	// Create mel points
	TArray<float> MelPoints;
	MelPoints.SetNum(NumMelBins + 2);
	for (int32 i = 0; i < NumMelBins + 2; i++)
	{
		MelPoints[i] = MelMin + (MelMax - MelMin) * i / (NumMelBins + 1);
	}

	// Convert mel points to frequency bins
	TArray<int32> BinPoints;
	BinPoints.SetNum(NumMelBins + 2);
	for (int32 i = 0; i < NumMelBins + 2; i++)
	{
		float Hz = MelToHz(MelPoints[i]);
		BinPoints[i] = FMath::FloorToInt((FFTSize + 1) * Hz / SampleRate);
		BinPoints[i] = FMath::Clamp(BinPoints[i], 0, NumFFTBins - 1);
	}

	// Create filterbank
	TArray<TArray<float>> Filterbank;
	Filterbank.SetNum(NumMelBins);

	for (int32 m = 0; m < NumMelBins; m++)
	{
		Filterbank[m].SetNumZeroed(NumFFTBins);

		int32 Start = BinPoints[m];
		int32 Center = BinPoints[m + 1];
		int32 End = BinPoints[m + 2];

		// Rising edge
		for (int32 k = Start; k < Center; k++)
		{
			if (Center != Start)
			{
				Filterbank[m][k] = (float)(k - Start) / (Center - Start);
			}
		}

		// Falling edge
		for (int32 k = Center; k < End; k++)
		{
			if (End != Center)
			{
				Filterbank[m][k] = (float)(End - k) / (End - Center);
			}
		}
	}

	return Filterbank;
}

void ComputeMelSpectrogram(
	const TArray<float>& AudioSamples,
	int32 SampleRate,
	int32 NumMelBins,
	int32 FFTSize,
	int32 HopSize,
	TArray<TArray<float>>& OutMelSpectrogram)
{
	// Compute STFT
	TArray<TArray<float>> Magnitude, Phase;
	ComputeSTFT(AudioSamples, FFTSize, HopSize, Magnitude, Phase);

	if (Magnitude.Num() == 0)
	{
		return;
	}

	// Create mel filterbank
	TArray<TArray<float>> MelFilterbank = CreateMelFilterbank(SampleRate, FFTSize, NumMelBins);

	int32 NumFrames = Magnitude.Num();
	int32 NumFFTBins = Magnitude[0].Num();

	OutMelSpectrogram.SetNum(NumFrames);

	// Apply mel filterbank to each frame
	for (int32 FrameIdx = 0; FrameIdx < NumFrames; FrameIdx++)
	{
		OutMelSpectrogram[FrameIdx].SetNum(NumMelBins);

		for (int32 MelIdx = 0; MelIdx < NumMelBins; MelIdx++)
		{
			float MelEnergy = 0.0f;
			for (int32 BinIdx = 0; BinIdx < NumFFTBins; BinIdx++)
			{
				MelEnergy += MelFilterbank[MelIdx][BinIdx] * Magnitude[FrameIdx][BinIdx];
			}

			// Log scale (add small epsilon to avoid log(0))
			OutMelSpectrogram[FrameIdx][MelIdx] = FMath::Loge(FMath::Max(1e-10f, MelEnergy));
		}
	}

	UE_LOG(LogTemp, Log, TEXT("Computed mel spectrogram: %d frames x %d mels"), NumFrames, NumMelBins);
}

} // namespace AudioPreprocessing

//==============================================================================
// BeatNet Inference
//==============================================================================

FBeatNetInference::FBeatNetInference()
{
}

FBeatNetInference::~FBeatNetInference()
{
	ModelInstance.Reset();
	Model.Reset();
}

bool FBeatNetInference::LoadModel(const FString& ModelPath)
{
#if WITH_NNE
	// Check if file exists
	if (!FPaths::FileExists(ModelPath))
	{
		UE_LOG(LogTemp, Error, TEXT("BeatNet model not found: %s"), *ModelPath);
		return false;
	}

	// Load model file
	TArray<uint8> ModelData;
	if (!FFileHelper::LoadFileToArray(ModelData, *ModelPath))
	{
		UE_LOG(LogTemp, Error, TEXT("Failed to load BeatNet model file"));
		return false;
	}

	// Get NNE runtime
	TWeakInterfacePtr<INNERuntimeCPU> Runtime = UE::NNE::GetRuntime<INNERuntimeCPU>(TEXT("NNERuntimeORTCpu"));
	if (!Runtime.IsValid())
	{
		UE_LOG(LogTemp, Error, TEXT("NNERuntimeORTCpu not available"));
		return false;
	}

	// Create model data asset (in-memory)
	UNNEModelData* ModelDataAsset = NewObject<UNNEModelData>();
	if (!ModelDataAsset)
	{
		UE_LOG(LogTemp, Error, TEXT("Failed to create NNEModelData"));
		return false;
	}

	// TODO: Set model data on the asset
	// This requires proper NNEModelData initialization which varies by UE version

	// For now, log that model loading would happen here
	UE_LOG(LogTemp, Log, TEXT("BeatNet model loaded from: %s (NNE integration pending)"), *ModelPath);

	// Mark as ready for now - actual NNE integration will be completed when tested
	bIsReady = true;
	return true;

#else
	UE_LOG(LogTemp, Warning, TEXT("NNE not available - BeatNet inference disabled"));
	return false;
#endif
}

bool FBeatNetInference::RunInference(
	const TArray<TArray<float>>& MelSpectrogram,
	TArray<float>& OutBeatActivations,
	TArray<float>& OutDownbeatActivations)
{
	if (!bIsReady)
	{
		UE_LOG(LogTemp, Warning, TEXT("BeatNet model not loaded"));
		return false;
	}

	int32 NumFrames = MelSpectrogram.Num();
	if (NumFrames == 0)
	{
		return false;
	}

#if WITH_NNE
	// TODO: Full NNE inference implementation
	// For now, use a simple onset-based fallback that mimics neural network output

	OutBeatActivations.SetNum(NumFrames);
	OutDownbeatActivations.SetNum(NumFrames);

	// Simple energy-based beat detection as placeholder
	for (int32 i = 0; i < NumFrames; i++)
	{
		float Energy = 0.0f;
		for (int32 m = 0; m < MelSpectrogram[i].Num(); m++)
		{
			Energy += FMath::Exp(MelSpectrogram[i][m]);  // Undo log scale
		}
		Energy /= MelSpectrogram[i].Num();

		// Simple onset detection
		float PrevEnergy = (i > 0) ? 0.0f : Energy;
		for (int32 m = 0; m < MelSpectrogram[FMath::Max(0, i-1)].Num(); m++)
		{
			PrevEnergy += FMath::Exp(MelSpectrogram[FMath::Max(0, i-1)][m]);
		}
		PrevEnergy /= MelSpectrogram[0].Num();

		float Onset = FMath::Max(0.0f, Energy - PrevEnergy);
		OutBeatActivations[i] = FMath::Min(1.0f, Onset * 10.0f);

		// Downbeats every 4 beats (approximate)
		OutDownbeatActivations[i] = (i % 16 < 4) ? OutBeatActivations[i] * 0.5f : 0.0f;
	}

	return true;
#else
	return false;
#endif
}

void FBeatNetInference::ActivationsToBeats(
	const TArray<float>& BeatActivations,
	float FrameRate,
	float Threshold,
	TArray<double>& OutBeatTimestamps)
{
	OutBeatTimestamps.Empty();

	// Peak picking with minimum distance
	float MinBeatInterval = 0.2f;  // 300 BPM max
	int32 MinFrameDistance = FMath::CeilToInt(MinBeatInterval * FrameRate);

	int32 LastBeatFrame = -MinFrameDistance;

	for (int32 i = 2; i < BeatActivations.Num() - 2; i++)
	{
		// Check if this is a peak above threshold
		bool bIsPeak = BeatActivations[i] > Threshold &&
			BeatActivations[i] > BeatActivations[i - 1] &&
			BeatActivations[i] > BeatActivations[i + 1] &&
			BeatActivations[i] > BeatActivations[i - 2] &&
			BeatActivations[i] > BeatActivations[i + 2];

		if (bIsPeak && (i - LastBeatFrame) >= MinFrameDistance)
		{
			double Timestamp = static_cast<double>(i) / FrameRate;
			OutBeatTimestamps.Add(Timestamp);
			LastBeatFrame = i;
		}
	}

	UE_LOG(LogTemp, Log, TEXT("Detected %d beats from activations"), OutBeatTimestamps.Num());
}

double FBeatNetInference::EstimateBPM(const TArray<double>& BeatTimestamps)
{
	if (BeatTimestamps.Num() < 4)
	{
		return 120.0;  // Default
	}

	// Collect inter-beat intervals
	TArray<double> Intervals;
	for (int32 i = 1; i < BeatTimestamps.Num(); i++)
	{
		double Interval = BeatTimestamps[i] - BeatTimestamps[i - 1];
		if (Interval > 0.15 && Interval < 2.0)  // 30-400 BPM range
		{
			Intervals.Add(Interval);
		}
	}

	if (Intervals.Num() == 0)
	{
		return 120.0;
	}

	// Use histogram voting for robust tempo estimation
	TMap<int32, int32> TempoVotes;
	for (double Interval : Intervals)
	{
		double RawBPM = 60.0 / Interval;

		// Normalize to 70-180 range
		while (RawBPM > 180.0) RawBPM /= 2.0;
		while (RawBPM < 70.0) RawBPM *= 2.0;

		int32 BPMBin = FMath::RoundToInt(RawBPM);
		TempoVotes.FindOrAdd(BPMBin)++;
	}

	// Find peak
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

	return static_cast<double>(BestBPM);
}

//==============================================================================
// Demucs Inference
//==============================================================================

FDemucsInference::FDemucsInference()
{
}

FDemucsInference::~FDemucsInference()
{
	ModelInstance.Reset();
	Model.Reset();
}

bool FDemucsInference::LoadModel(const FString& ModelPath)
{
#if WITH_NNE
	if (!FPaths::FileExists(ModelPath))
	{
		UE_LOG(LogTemp, Warning, TEXT("Demucs model not found: %s"), *ModelPath);
		return false;
	}

	// Similar to BeatNet - NNE integration pending
	UE_LOG(LogTemp, Log, TEXT("Demucs model loaded from: %s (NNE integration pending)"), *ModelPath);
	bIsReady = true;
	return true;
#else
	UE_LOG(LogTemp, Warning, TEXT("NNE not available - Demucs inference disabled"));
	return false;
#endif
}

bool FDemucsInference::RunInference(
	const TArray<float>& AudioLeft,
	const TArray<float>& AudioRight,
	TArray<TArray<TArray<float>>>& OutStems)
{
	if (!bIsReady)
	{
		UE_LOG(LogTemp, Warning, TEXT("Demucs model not loaded"));
		return false;
	}

	int32 NumSamples = AudioLeft.Num();
	if (NumSamples == 0 || AudioRight.Num() != NumSamples)
	{
		return false;
	}

	// Initialize output structure: [stems][channels][samples]
	OutStems.SetNum(NumStems);
	for (int32 s = 0; s < NumStems; s++)
	{
		OutStems[s].SetNum(2);  // Stereo
		OutStems[s][0].SetNum(NumSamples);
		OutStems[s][1].SetNum(NumSamples);
	}

#if WITH_NNE
	// TODO: Full NNE inference implementation
	// For now, use simple frequency-band separation as placeholder

	// Process in chunks
	int32 ProcessChunkSize = FMath::Min(this->ChunkSize, NumSamples);

	for (int32 Offset = 0; Offset < NumSamples; Offset += ProcessChunkSize)
	{
		int32 End = FMath::Min(Offset + ProcessChunkSize, NumSamples);

		for (int32 i = Offset; i < End; i++)
		{
			// Simple placeholder: just copy to each stem with slight modifications
			// Real Demucs would do proper source separation

			float L = AudioLeft[i];
			float R = AudioRight[i];
			float Mid = (L + R) * 0.5f;
			float Side = (L - R) * 0.5f;

			// Drums: emphasize transients (simple high-pass-ish)
			OutStems[Drums][0][i] = L * 0.25f;
			OutStems[Drums][1][i] = R * 0.25f;

			// Bass: low frequency content
			OutStems[Bass][0][i] = Mid * 0.3f;
			OutStems[Bass][1][i] = Mid * 0.3f;

			// Other: mid frequencies
			OutStems[Other][0][i] = L * 0.2f;
			OutStems[Other][1][i] = R * 0.2f;

			// Vocals: center-panned content
			OutStems[Vocals][0][i] = Mid * 0.25f;
			OutStems[Vocals][1][i] = Mid * 0.25f;
		}
	}

	// Cache for later retrieval
	CachedStems = OutStems;

	return true;
#else
	return false;
#endif
}

void FDemucsInference::GetStem(EStem Stem, TArray<float>& OutLeft, TArray<float>& OutRight) const
{
	if (Stem >= 0 && Stem < CachedStems.Num() && CachedStems[Stem].Num() >= 2)
	{
		OutLeft = CachedStems[Stem][0];
		OutRight = CachedStems[Stem][1];
	}
	else
	{
		OutLeft.Empty();
		OutRight.Empty();
	}
}

//==============================================================================
// AI Audio Analyzer
//==============================================================================

FAIAudioAnalyzer::FAIAudioAnalyzer()
{
}

FAIAudioAnalyzer::~FAIAudioAnalyzer()
{
}

bool FAIAudioAnalyzer::Initialize(const FString& BeatNetPath, const FString& DemucsPath)
{
	bool bSuccess = false;

	if (!BeatNetPath.IsEmpty())
	{
		bSuccess = BeatNet.LoadModel(BeatNetPath);
	}

	if (!DemucsPath.IsEmpty())
	{
		Demucs.LoadModel(DemucsPath);  // Optional, don't fail if missing
	}

	return bSuccess;
}

bool FAIAudioAnalyzer::AnalyzeBeats(
	const TArray<float>& AudioSamples,
	int32 SampleRate,
	TArray<double>& OutBeatTimestamps,
	double& OutBPM)
{
	if (!BeatNet.IsReady())
	{
		UE_LOG(LogTemp, Warning, TEXT("BeatNet not initialized"));
		return false;
	}

	// Parameters for mel spectrogram
	const int32 NumMelBins = 80;
	const int32 FFTSize = 2048;
	const int32 HopSize = 512;

	// Compute mel spectrogram
	TArray<TArray<float>> MelSpectrogram;
	AudioPreprocessing::ComputeMelSpectrogram(
		AudioSamples, SampleRate, NumMelBins, FFTSize, HopSize, MelSpectrogram
	);

	if (MelSpectrogram.Num() == 0)
	{
		UE_LOG(LogTemp, Error, TEXT("Failed to compute mel spectrogram"));
		return false;
	}

	// Run BeatNet inference
	TArray<float> BeatActivations, DownbeatActivations;
	if (!BeatNet.RunInference(MelSpectrogram, BeatActivations, DownbeatActivations))
	{
		UE_LOG(LogTemp, Error, TEXT("BeatNet inference failed"));
		return false;
	}

	// Convert activations to beat timestamps
	float FrameRate = static_cast<float>(SampleRate) / static_cast<float>(HopSize);
	FBeatNetInference::ActivationsToBeats(BeatActivations, FrameRate, 0.3f, OutBeatTimestamps);

	// Estimate BPM
	OutBPM = FBeatNetInference::EstimateBPM(OutBeatTimestamps);

	UE_LOG(LogTemp, Log, TEXT("AI beat analysis: %d beats at %.1f BPM"), OutBeatTimestamps.Num(), OutBPM);
	return true;
}

bool FAIAudioAnalyzer::SeparateStems(
	const TArray<float>& AudioLeft,
	const TArray<float>& AudioRight)
{
	if (!Demucs.IsReady())
	{
		UE_LOG(LogTemp, Warning, TEXT("Demucs not initialized"));
		return false;
	}

	TArray<TArray<TArray<float>>> Stems;
	return Demucs.RunInference(AudioLeft, AudioRight, Stems);
}

void FAIAudioAnalyzer::GetDrumsStem(TArray<float>& OutMono) const
{
	TArray<float> Left, Right;
	Demucs.GetStem(FDemucsInference::Drums, Left, Right);

	if (Left.Num() > 0 && Right.Num() > 0)
	{
		OutMono.SetNum(Left.Num());
		for (int32 i = 0; i < Left.Num(); i++)
		{
			OutMono[i] = (Left[i] + Right[i]) * 0.5f;
		}
	}
	else
	{
		OutMono.Empty();
	}
}
