#include "BeatsyncProcessingTask.h"
#include "Async/Async.h"
#include "HAL/FileManager.h"
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
        AsyncTask(ENamedThreads::GameThread, [this, Progress, Status]() {
            OnProgress.ExecuteIfBound(Progress, Status);
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
        AsyncTask(ENamedThreads::GameThread, [this, Result]() {
            OnComplete.ExecuteIfBound(Result);
        });
        return;
    }

    // Step 1: Analyze audio
    ReportProgress(0.05f, TEXT("Analyzing audio..."));

    if (bCancelRequested)
    {
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("Cancelled");
        AsyncTask(ENamedThreads::GameThread, [this, Result]() {
            OnComplete.ExecuteIfBound(Result);
        });
        return;
    }

    void* Analyzer = FBeatsyncLoader::CreateAnalyzer();
    if (!Analyzer)
    {
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("Failed to create analyzer");
        AsyncTask(ENamedThreads::GameThread, [this, Result]() {
            OnComplete.ExecuteIfBound(Result);
        });
        return;
    }

    FBeatGrid BeatGrid;
    bool bSuccess = FBeatsyncLoader::AnalyzeAudio(Analyzer, Params.AudioPath, BeatGrid);
    FBeatsyncLoader::DestroyAnalyzer(Analyzer);

    if (!bSuccess || BeatGrid.Beats.Num() == 0)
    {
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("Failed to analyze audio or no beats found");
        AsyncTask(ENamedThreads::GameThread, [this, Result]() {
            OnComplete.ExecuteIfBound(Result);
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
        AsyncTask(ENamedThreads::GameThread, [this, Result]() {
            OnComplete.ExecuteIfBound(Result);
        });
        return;
    }

    // Step 2: Apply beat rate filter
    TArray<double> FilteredBeats;
    int32 BeatDivisor = 1 << Params.BeatRate; // 1, 2, 4, 8
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
        AsyncTask(ENamedThreads::GameThread, [this, Result]() {
            OnComplete.ExecuteIfBound(Result);
        });
        return;
    }

    // Set up progress callback for video processing
    FBeatsyncLoader::SetProgressCallback(Writer, [this](double Prog) {
        if (!bCancelRequested)
        {
            ReportProgress(0.2f + 0.5f * static_cast<float>(Prog), TEXT("Processing video..."));
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
    FString TempBaseName = FString::Printf(TEXT("TripSitter_%d"), FMath::Rand());
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
        AsyncTask(ENamedThreads::GameThread, [this, Result]() {
            OnComplete.ExecuteIfBound(Result);
        });
        return;
    }

    if (bCancelRequested)
    {
        FBeatsyncLoader::DestroyVideoWriter(Writer);
        IFileManager::Get().Delete(*TempVideoPath, false, true, true);
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("Cancelled");
        AsyncTask(ENamedThreads::GameThread, [this, Result]() {
            OnComplete.ExecuteIfBound(Result);
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
        AsyncTask(ENamedThreads::GameThread, [this, Result]() {
            OnComplete.ExecuteIfBound(Result);
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

    AsyncTask(ENamedThreads::GameThread, [this, Result]() {
        OnComplete.ExecuteIfBound(Result);
    });
}
