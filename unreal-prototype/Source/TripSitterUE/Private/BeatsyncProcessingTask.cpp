#include "BeatsyncProcessingTask.h"
#include "Async/Async.h"
#include "HAL/FileManager.h"
#include "HAL/PlatformProcess.h"
#include "Misc/Paths.h"
#include <memory>


FBeatsyncProcessingTask::FBeatsyncProcessingTask(const FBeatsyncProcessingParams& InParams,
                                                  FOnBeatsyncProcessingProgress InProgressDelegate,
                                                  FOnBeatsyncProcessingComplete InCompleteDelegate)
    : Params(InParams)
    , OnProgress(InProgressDelegate)
    , OnComplete(InCompleteDelegate)
    , SharedCancelFlag(MakeShared<FThreadSafeBool>(false))
{
}

FBeatsyncProcessingTask::~FBeatsyncProcessingTask()
{
    // Signal cancellation to stop any in-progress work
    if (SharedCancelFlag.IsValid())
    {
        SharedCancelFlag->AtomicSet(true);
    }

    // Wait for DoWork to complete (with timeout to prevent indefinite hangs)
    constexpr double TimeoutSeconds = 10.0;
    double StartTime = FPlatformTime::Seconds();
    while (!bWorkCompleted && (FPlatformTime::Seconds() - StartTime) < TimeoutSeconds)
    {
        FPlatformProcess::Sleep(0.01f);
    }

    if (!bWorkCompleted)
    {
        UE_LOG(LogTemp, Warning, TEXT("FBeatsyncProcessingTask: Destructor timed out waiting for DoWork to complete"));
    }

    // Clean up Writer if DoWork didn't complete normally (e.g., early exit or exception)
    if (Writer)
    {
        FBeatsyncLoader::SetProgressCallback(Writer, nullptr);
        FBeatsyncLoader::DestroyVideoWriter(Writer);
        Writer = nullptr;
    }
}

void FBeatsyncProcessingTask::ReportProgress(float Progress, const FString& Status)
{
    // Optionally add an event to the current backend span if available
    // Note: This assumes StartSpan was called outside and a span handle is available via FBeatsyncLoader

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
        bWorkCompleted.AtomicSet(true);
        return;
    }

    // Create top-level span for the processing job only if backend is initialized
    auto Span = FBeatsyncLoader::StartSpan(TEXT("BeatsyncProcessingTask"));

    // Step 1: Analyze audio
    ReportProgress(0.05f, TEXT("Analyzing audio..."));

    if (SharedCancelFlag.IsValid() && *SharedCancelFlag)
    {
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("Cancelled");
        if (Span) {
            FBeatsyncLoader::SpanAddEvent(Span, TEXT("cancelled-before-analyze"));
            FBeatsyncLoader::SpanSetError(Span, Result.ErrorMessage);
            FBeatsyncLoader::EndSpan(Span);
        }
        auto LocalOnComplete = OnComplete;
        AsyncTask(ENamedThreads::GameThread, [LocalOnComplete, Result]() {
            LocalOnComplete.ExecuteIfBound(Result);
        });
        bWorkCompleted.AtomicSet(true);
        return;
    }

    auto Analyzer = FBeatsyncLoader::CreateAnalyzer();
    if (!Analyzer)
    {
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("Failed to create analyzer");
        if (Span) FBeatsyncLoader::SpanSetError(Span, Result.ErrorMessage);
        if (Span) FBeatsyncLoader::EndSpan(Span);
        auto LocalOnComplete = OnComplete;
        AsyncTask(ENamedThreads::GameThread, [LocalOnComplete, Result]() {
            LocalOnComplete.ExecuteIfBound(Result);
        });
        bWorkCompleted.AtomicSet(true);
        return;
    }

    FBeatGrid BeatGrid;
    bool bSuccess = FBeatsyncLoader::AnalyzeAudio(Analyzer, Params.AudioPath, BeatGrid);
    FBeatsyncLoader::DestroyAnalyzer(Analyzer);

    if (!bSuccess || BeatGrid.Beats.Num() == 0)
    {
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("Failed to analyze audio or no beats found");
        if (Span) FBeatsyncLoader::SpanSetError(Span, Result.ErrorMessage);
        if (Span) FBeatsyncLoader::EndSpan(Span);
        auto LocalOnComplete = OnComplete;
        AsyncTask(ENamedThreads::GameThread, [LocalOnComplete, Result]() {
            LocalOnComplete.ExecuteIfBound(Result);
        });
        bWorkCompleted.AtomicSet(true);
        return;
    }

    Result.BPM = BeatGrid.BPM;
    Result.BeatTimes = BeatGrid.Beats;

    ReportProgress(0.2f, FString::Printf(TEXT("Found %d beats at %.1f BPM"), BeatGrid.Beats.Num(), BeatGrid.BPM));

    if (SharedCancelFlag.IsValid() && *SharedCancelFlag)
    {
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("Cancelled");
        if (Span) { FBeatsyncLoader::SpanSetError(Span, Result.ErrorMessage); FBeatsyncLoader::EndSpan(Span); }
        auto LocalOnComplete = OnComplete;
        AsyncTask(ENamedThreads::GameThread, [LocalOnComplete, Result]() {
            LocalOnComplete.ExecuteIfBound(Result);
        });
        bWorkCompleted.AtomicSet(true);
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
    
    // Validate FilteredBeats - if empty, try to recover or fail gracefully
    if (FilteredBeats.Num() == 0)
    {
        // Try reducing beat divisor to get at least one beat
        if (ClampedBeatRate > 0)
        {
            ClampedBeatRate = 0; // Reset to no filtering
            BeatDivisor = 1;
            FilteredBeats.Reset();
            for (int32 i = 0; i < BeatGrid.Beats.Num(); i += BeatDivisor)
            {
                FilteredBeats.Add(BeatGrid.Beats[i]);
            }
        }
        
        // If still empty, fallback to first beat or fail
        if (FilteredBeats.Num() == 0)
        {
            if (BeatGrid.Beats.Num() > 0)
            {
                FilteredBeats.Add(BeatGrid.Beats[0]); // Use first beat as fallback
            }
            else
            {
                Result.bSuccess = false;
                Result.ErrorMessage = TEXT("No beats detected in audio");
                if (Span) FBeatsyncLoader::SpanSetError(Span, Result.ErrorMessage);
                if (Span) FBeatsyncLoader::EndSpan(Span);

                // Marshal completion to game thread
                auto LocalOnComplete = OnComplete;
                AsyncTask(ENamedThreads::GameThread, [LocalOnComplete, Result]() {
                    LocalOnComplete.ExecuteIfBound(Result);
                });
                bWorkCompleted.AtomicSet(true);
                return;
            }
        }
    }
    
    Result.BeatCount = FilteredBeats.Num();

    // Step 3: Create video writer (assign to member for destructor cleanup)
    Writer = FBeatsyncLoader::CreateVideoWriter();
    if (!Writer)
    {
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("Failed to create video writer");
        if (Span) FBeatsyncLoader::SpanSetError(Span, Result.ErrorMessage);
        if (Span) FBeatsyncLoader::EndSpan(Span);
        auto LocalOnComplete = OnComplete;
        AsyncTask(ENamedThreads::GameThread, [LocalOnComplete, Result]() {
            LocalOnComplete.ExecuteIfBound(Result);
        });
        bWorkCompleted.AtomicSet(true);
        return;
    }

    // Set up progress callback for video processing
    // Note: SharedCancelFlag is a shared member that reflects runtime cancellation state
    auto LocalOnProgress = OnProgress;
    FBeatsyncLoader::SetProgressCallback(Writer, [LocalOnProgress, SharedCancelFlag = this->SharedCancelFlag](double Prog) {
        if (!(*SharedCancelFlag)) {
            float Progress = 0.2f + 0.5f * static_cast<float>(Prog);
            FString Status = TEXT("Processing video...");
            // Marshal to game thread for UI updates - Slate cannot be accessed from worker threads
            AsyncTask(ENamedThreads::GameThread, [LocalOnProgress, Progress, Status]() {
                LocalOnProgress.ExecuteIfBound(Progress, Status);
            });
        }
    });

    // Step 4: Cut video at beats
    ReportProgress(0.25f, TEXT("Cutting video at beats..."));

    double ClipDuration;
    if (FilteredBeats.Num() > 1) {
        ClipDuration = FilteredBeats[1] - FilteredBeats[0];
    } else if (BeatGrid.BPM > 0.0) {
        // Use one beat duration when BPM is available
        ClipDuration = 60.0 / BeatGrid.BPM;
    } else {
        // Fallback to default duration when no BPM info is available
        ClipDuration = 1.0;
    }

    // Create temp files in system temp directory (not next to output)
    FString TempDir = FPaths::Combine(FPlatformProcess::UserTempDir(), TEXT("TripSitter"));
    
    // Ensure the TripSitter temp directory exists
    IFileManager::Get().MakeDirectory(*TempDir, true);
    
    // Generate unique temp filenames using GUID
    FString TempBaseName = FString::Printf(TEXT("temp_%s"), *FGuid::NewGuid().ToString(EGuidFormats::Digits));
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
        Result.bSuccess = false;
        Result.ErrorMessage = ErrorMsg.IsEmpty() ? TEXT("Failed to cut video") : ErrorMsg;
        if (Span) FBeatsyncLoader::SpanSetError(Span, Result.ErrorMessage);
        FBeatsyncLoader::SetProgressCallback(Writer, nullptr);
        FBeatsyncLoader::DestroyVideoWriter(Writer);
        Writer = nullptr;
        IFileManager::Get().Delete(*TempVideoPath, false, true, true);
        TempVideoPath.Empty();
        if (Span) FBeatsyncLoader::EndSpan(Span);
        auto LocalOnComplete = OnComplete;
        AsyncTask(ENamedThreads::GameThread, [LocalOnComplete, Result]() {
            LocalOnComplete.ExecuteIfBound(Result);
        });
        bWorkCompleted.AtomicSet(true);
        return;
    }

    if (SharedCancelFlag.IsValid() && *SharedCancelFlag)
    {
        if (Span) FBeatsyncLoader::SpanAddEvent(Span, TEXT("cancelled-after-cut"));
        FBeatsyncLoader::DestroyVideoWriter(Writer);
        Writer = nullptr;
        IFileManager::Get().Delete(*TempVideoPath, false, true, true);
        TempVideoPath.Empty();
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("Cancelled");
        if (Span) { FBeatsyncLoader::SpanSetError(Span, Result.ErrorMessage); FBeatsyncLoader::EndSpan(Span); }
        auto LocalOnComplete = OnComplete;
        AsyncTask(ENamedThreads::GameThread, [LocalOnComplete, Result]() {
            LocalOnComplete.ExecuteIfBound(Result);
        });
        bWorkCompleted.AtomicSet(true);
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
            TempEffectsPath.Empty();
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("TripSitter: Effects application failed, continuing without effects"));
            if (Span) FBeatsyncLoader::SpanAddEvent(Span, TEXT("effects-failed"));
        }
    }

    if (SharedCancelFlag.IsValid() && *SharedCancelFlag)
    {
        FBeatsyncLoader::DestroyVideoWriter(Writer);
        Writer = nullptr;
        IFileManager::Get().Delete(*CurrentVideoPath, false, true, true);
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("Cancelled");
        if (Span) { FBeatsyncLoader::SpanSetError(Span, Result.ErrorMessage); FBeatsyncLoader::EndSpan(Span); }
        auto LocalOnComplete = OnComplete;
        AsyncTask(ENamedThreads::GameThread, [LocalOnComplete, Result]() {
            LocalOnComplete.ExecuteIfBound(Result);
        });
        bWorkCompleted.AtomicSet(true);
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
        // Invalidate temp paths so later cleanup does not attempt to delete files that were moved
        TempVideoPath.Empty();
        CurrentVideoPath.Empty();
        // Set result flag for mux failure so caller can detect partial success
        Result.bAudioMuxFailed = true;
        bSuccess = true; // Consider it a partial success
        if (Span) FBeatsyncLoader::SpanAddEvent(Span, TEXT("audio-mux-failed"));
    }

    // Clean up temp files
    if (!TempVideoPath.IsEmpty())
    {
        IFileManager::Get().Delete(*TempVideoPath, false, true, true);
    }
    if (!TempEffectsPath.IsEmpty())
    {
        IFileManager::Get().Delete(*TempEffectsPath, false, true, true);
        TempEffectsPath.Empty();
    }

    FBeatsyncLoader::DestroyVideoWriter(Writer);
    Writer = nullptr;  // Prevent double-destroy in destructor

    // Report completion
    Result.bSuccess = bSuccess;
    if (!bSuccess)
    {
        Result.ErrorMessage = TEXT("Processing failed");
        if (Span) FBeatsyncLoader::SpanSetError(Span, Result.ErrorMessage);
    }

    ReportProgress(1.0f, bSuccess ? TEXT("Complete!") : TEXT("Failed"));

    if (Span) FBeatsyncLoader::EndSpan(Span);

    auto LocalOnComplete = OnComplete;
    AsyncTask(ENamedThreads::GameThread, [LocalOnComplete, Result]() {
        LocalOnComplete.ExecuteIfBound(Result);
    });

    // Signal completion for destructor synchronization
    bWorkCompleted.AtomicSet(true);
}
