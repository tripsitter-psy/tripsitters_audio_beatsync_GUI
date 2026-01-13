#include "BeatsyncProcessingTask.h"
#include "Async/Async.h"
#include "HAL/FileManager.h"
#include "HAL/PlatformProcess.h"
#include "Misc/Paths.h"

FBeatsyncProcessingTask::FBeatsyncProcessingTask(const FBeatsyncProcessingParams& InParams,
                                                  FOnBeatsyncProcessingProgress InProgressDelegate,
                                                  FOnBeatsyncProcessingComplete InCompleteDelegate)
    : Params(InParams)
    , OnProgress(InProgressDelegate)
    , OnComplete(InCompleteDelegate)
{
}

void FBeatsyncProcessingTask::ReportProgress(float Progress, const FString& Status)
{
    if (OnProgress.IsBound())
    {
        // Marshal to game thread for UI updates
        auto LocalOnProgress = OnProgress;
        AsyncTask(ENamedThreads::GameThread, [LocalOnProgress, Progress, Status]() {
            LocalOnProgress.ExecuteIfBound(Progress, Status);
        });
    }
}

bool FBeatsyncProcessingTask::HasAnyEffectsEnabled() const
{
    return Params.EffectsConfig.bEnableVignette ||
           Params.EffectsConfig.bEnableBeatFlash ||
           Params.EffectsConfig.bEnableBeatZoom ||
           Params.EffectsConfig.bEnableColorGrade ||
           Params.EffectsConfig.bEnableTransitions;
}

void FBeatsyncProcessingTask::DoWork()
{
    FBeatsyncProcessingResult Result;

    // Check if backend is initialized
    if (!FBeatsyncLoader::IsInitialized())
    {
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("Backend not loaded");
        auto LocalOnComplete = OnComplete;
        AsyncTask(ENamedThreads::GameThread, [LocalOnComplete, Result]() {
            LocalOnComplete.ExecuteIfBound(Result);
        });
        return;
    }

    // Step 1: Analyze audio
    ReportProgress(0.05f, TEXT("Analyzing audio..."));

    if (bCancelRequested)
    {
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("Cancelled");
        auto LocalOnComplete = OnComplete;
        AsyncTask(ENamedThreads::GameThread, [LocalOnComplete, Result]() {
            LocalOnComplete.ExecuteIfBound(Result);
        });
        return;
    }

    FBeatGrid BeatGrid;
    bool bSuccess = false;

    // Try AI analyzer first (GPU accelerated ONNX)
    if (FBeatsyncLoader::IsAIAvailable())
    {
        ReportProgress(0.05f, TEXT("Analyzing audio with AI (GPU)..."));
        UE_LOG(LogTemp, Log, TEXT("TripSitter: Using AI analyzer (providers: %s)"), *FBeatsyncLoader::GetAIProviders());

        // Get path to beatnet model - look relative to executable or in ThirdParty
        FString ExeDir = FPaths::GetPath(FPlatformProcess::ExecutablePath());
        FString ModelPath = FPaths::Combine(ExeDir, TEXT("models"), TEXT("beatnet.onnx"));
        if (!FPaths::FileExists(ModelPath))
        {
            // Try ThirdParty location
            ModelPath = FPaths::Combine(ExeDir, TEXT(".."), TEXT(".."), TEXT("Source"), TEXT("Programs"),
                                         TEXT("TripSitter"), TEXT("ThirdParty"), TEXT("beatsync"), TEXT("models"), TEXT("beatnet.onnx"));
            ModelPath = FPaths::ConvertRelativePathToFull(ModelPath);
        }

        if (FPaths::FileExists(ModelPath))
        {
            FAIConfig AIConfig;
            AIConfig.BeatModelPath = ModelPath;
            AIConfig.bUseGPU = true;  // Enable CUDA
            AIConfig.bUseStemSeparation = false;  // Quick mode
            AIConfig.bUseDrumsForBeats = true;
            AIConfig.BeatThreshold = 0.5f;
            AIConfig.DownbeatThreshold = 0.5f;

            void* AIAnalyzer = FBeatsyncLoader::CreateAIAnalyzer(AIConfig);
            if (AIAnalyzer)
            {
                FAIResult AIResult;
                bSuccess = FBeatsyncLoader::AIAnalyzeQuick(AIAnalyzer, Params.AudioPath, AIResult);

                if (bSuccess && AIResult.Beats.Num() > 0)
                {
                    BeatGrid.Beats = AIResult.Beats;
                    BeatGrid.BPM = AIResult.BPM;
                    BeatGrid.Duration = AIResult.Duration;
                    UE_LOG(LogTemp, Log, TEXT("TripSitter: AI analysis found %d beats at %.1f BPM"), AIResult.Beats.Num(), AIResult.BPM);
                }
                else
                {
                    FString AIError = FBeatsyncLoader::GetAILastError(AIAnalyzer);
                    UE_LOG(LogTemp, Warning, TEXT("TripSitter: AI analysis failed: %s, falling back to spectral flux"), *AIError);
                    bSuccess = false;
                }

                FBeatsyncLoader::DestroyAIAnalyzer(AIAnalyzer);
            }
            else
            {
                UE_LOG(LogTemp, Warning, TEXT("TripSitter: Failed to create AI analyzer, falling back to spectral flux"));
            }
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("TripSitter: Beat model not found at %s, falling back to spectral flux"), *ModelPath);
        }
    }

    // Fall back to CPU-based spectral flux if AI not available or failed
    if (!bSuccess)
    {
        ReportProgress(0.05f, TEXT("Analyzing audio (CPU)..."));
        UE_LOG(LogTemp, Log, TEXT("TripSitter: Using spectral flux analyzer (CPU)"));

        void* Analyzer = FBeatsyncLoader::CreateAnalyzer();
        if (!Analyzer)
        {
            Result.bSuccess = false;
            Result.ErrorMessage = TEXT("Failed to create analyzer");
            auto LocalOnComplete = OnComplete;
            AsyncTask(ENamedThreads::GameThread, [LocalOnComplete, Result]() {
                LocalOnComplete.ExecuteIfBound(Result);
            });
            return;
        }

        bSuccess = FBeatsyncLoader::AnalyzeAudio(Analyzer, Params.AudioPath, BeatGrid);
        FBeatsyncLoader::DestroyAnalyzer(Analyzer);
    }

    if (!bSuccess || BeatGrid.Beats.Num() == 0)
    {
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("Failed to analyze audio or no beats found");
        auto LocalOnComplete = OnComplete;
        AsyncTask(ENamedThreads::GameThread, [LocalOnComplete, Result]() {
            LocalOnComplete.ExecuteIfBound(Result);
        });
        return;
    }

    Result.BPM = BeatGrid.BPM;
    Result.BeatTimes = BeatGrid.Beats;

    ReportProgress(0.2f, FString::Printf(TEXT("Found %d beats at %.1f BPM"), BeatGrid.Beats.Num(), BeatGrid.BPM));

    if (bCancelRequested)
    {
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("Cancelled");
        auto LocalOnComplete = OnComplete;
        AsyncTask(ENamedThreads::GameThread, [LocalOnComplete, Result]() {
            LocalOnComplete.ExecuteIfBound(Result);
        });
        return;
    }

    // Step 2: Apply beat rate filter
    TArray<double> FilteredBeats;
    int32 ClampedBeatRate = FMath::Clamp(Params.BeatRate, 0, 3); // Clamp to safe range to prevent overflow
    int32 BeatDivisor = 1 << ClampedBeatRate; // 1, 2, 4, 8
    for (int32 i = 0; i < BeatGrid.Beats.Num(); i += BeatDivisor)
    {
        FilteredBeats.Add(BeatGrid.Beats[i]);
    }
    Result.BeatCount = FilteredBeats.Num();

    // Step 3: Create video writer
    void* Writer = FBeatsyncLoader::CreateVideoWriter();
    if (!Writer)
    {
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("Failed to create video writer");
        auto LocalOnComplete = OnComplete;
        AsyncTask(ENamedThreads::GameThread, [LocalOnComplete, Result]() {
            LocalOnComplete.ExecuteIfBound(Result);
        });
        return;
    }

    // Set up progress callback for video processing
    // Note: bCancelRequested is a member of this task object which outlives the callback
    // since we destroy the Writer (and its callback) before task destruction
    auto LocalOnProgress = OnProgress;
    const FThreadSafeBool* CancelFlag = &bCancelRequested;
    FBeatsyncLoader::SetProgressCallback(Writer, [LocalOnProgress, CancelFlag](double Prog) {
        if (!(*CancelFlag))
        {
            LocalOnProgress.ExecuteIfBound(0.2f + 0.5f * static_cast<float>(Prog), TEXT("Processing video..."));
        }
    });

    // Step 4: Cut video at beats
    ReportProgress(0.25f, TEXT("Cutting video at beats..."));

    double ClipDuration = FilteredBeats.Num() > 1 ? (FilteredBeats[1] - FilteredBeats[0]) : 1.0;

    // Create temp files in system temp directory (not next to output)
    FString TempDir = FPaths::ConvertRelativePathToFull(FPaths::ProjectIntermediateDir());
    // For Program target, use system temp if ProjectIntermediateDir doesn't exist
    if (!FPaths::DirectoryExists(TempDir))
    {
        TempDir = FPlatformProcess::UserTempDir();
    }
    FString TempBaseName = FString::Printf(TEXT("TripSitter_%s"), *FGuid::NewGuid().ToString(EGuidFormats::Digits));
    FString TempVideoPath = FPaths::Combine(TempDir, TempBaseName + TEXT("_video.mp4"));
    FString TempEffectsPath = FPaths::Combine(TempDir, TempBaseName + TEXT("_effects.mp4"));

    UE_LOG(LogTemp, Log, TEXT("TripSitter: Using temp directory: %s"), *TempDir);

    // Cut video
    if (Params.bIsMultiClip && Params.VideoPaths.Num() > 1)
    {
        bSuccess = FBeatsyncLoader::CutVideoAtBeatsMulti(Writer, Params.VideoPaths, FilteredBeats, TempVideoPath, ClipDuration);
    }
    else
    {
        FString SingleVideo = Params.VideoPaths.Num() > 0 ? Params.VideoPaths[0] : Params.VideoPath;
        bSuccess = FBeatsyncLoader::CutVideoAtBeats(Writer, SingleVideo, FilteredBeats, TempVideoPath, ClipDuration);
    }

    if (!bSuccess)
    {
        FString ErrorMsg = FBeatsyncLoader::GetVideoLastError(Writer);
        Result.bSuccess = false;
        Result.ErrorMessage = ErrorMsg.IsEmpty() ? TEXT("Failed to cut video") : ErrorMsg;
        FBeatsyncLoader::DestroyVideoWriter(Writer);
        IFileManager::Get().Delete(*TempVideoPath, false, true, true);
        auto LocalOnComplete = OnComplete;
        AsyncTask(ENamedThreads::GameThread, [LocalOnComplete, Result]() {
            LocalOnComplete.ExecuteIfBound(Result);
        });
        return;
    }

    if (bCancelRequested)
    {
        FBeatsyncLoader::DestroyVideoWriter(Writer);
        IFileManager::Get().Delete(*TempVideoPath, false, true, true);
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("Cancelled");
        auto LocalOnComplete = OnComplete;
        AsyncTask(ENamedThreads::GameThread, [LocalOnComplete, Result]() {
            LocalOnComplete.ExecuteIfBound(Result);
        });
        return;
    }

    // Step 5: Apply effects if enabled
    FString CurrentVideoPath = TempVideoPath;
    if (HasAnyEffectsEnabled())
    {
        ReportProgress(0.75f, TEXT("Applying effects..."));

        // Set effects config
        FBeatsyncLoader::SetEffectsConfig(Writer, Params.EffectsConfig);

        // Apply effects
        bSuccess = FBeatsyncLoader::ApplyEffects(Writer, CurrentVideoPath, TempEffectsPath, FilteredBeats);

        if (bSuccess)
        {
            IFileManager::Get().Delete(*CurrentVideoPath, false, true, true);
            CurrentVideoPath = TempEffectsPath;
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("TripSitter: Effects application failed, continuing without effects"));
        }
    }

    if (bCancelRequested)
    {
        FBeatsyncLoader::DestroyVideoWriter(Writer);
        IFileManager::Get().Delete(*CurrentVideoPath, false, true, true);
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("Cancelled");
        auto LocalOnComplete = OnComplete;
        AsyncTask(ENamedThreads::GameThread, [LocalOnComplete, Result]() {
            LocalOnComplete.ExecuteIfBound(Result);
        });
        return;
    }

    // Step 6: Mux audio
    ReportProgress(0.9f, TEXT("Adding audio track..."));

    bSuccess = FBeatsyncLoader::AddAudioTrack(Writer, CurrentVideoPath, Params.AudioPath, Params.OutputPath,
                                               true, Params.AudioStart, Params.AudioEnd);

    if (!bSuccess)
    {
        UE_LOG(LogTemp, Warning, TEXT("TripSitter: Audio muxing failed, using video-only output"));
        // Fall back to video-only output
        IFileManager::Get().Move(*Params.OutputPath, *CurrentVideoPath, true, true);
        bSuccess = true; // Consider it a partial success
    }

    // Clean up temp files
    IFileManager::Get().Delete(*TempVideoPath, false, true, true);
    IFileManager::Get().Delete(*TempEffectsPath, false, true, true);

    FBeatsyncLoader::DestroyVideoWriter(Writer);

    // Report completion
    Result.bSuccess = bSuccess;
    if (!bSuccess)
    {
        Result.ErrorMessage = TEXT("Processing failed");
    }

    ReportProgress(1.0f, bSuccess ? TEXT("Complete!") : TEXT("Failed"));

    auto LocalOnComplete = OnComplete;
    AsyncTask(ENamedThreads::GameThread, [LocalOnComplete, Result]() {
        LocalOnComplete.ExecuteIfBound(Result);
    });
}
