#include "BeatsyncProcessingTask.h"
#include "Async/Async.h"
#include "HAL/FileManager.h"
#include "HAL/ThreadSafeBool.h"
#include "HAL/PlatformProcess.h"
#include "Misc/Paths.h"
#include <memory>

FBeatsyncProcessingTask::FBeatsyncProcessingTask(const FBeatsyncProcessingParams& InParams,
                                                  FOnBeatsyncProcessingProgress InProgressDelegate,
                                                  FOnBeatsyncProcessingComplete InCompleteDelegate)
    : Params(InParams)
    , OnProgress(InProgressDelegate)
    , OnComplete(InCompleteDelegate)
{
    ProgressGuard = MakeShared<FThreadSafeBool>(false);
    // Create event for synchronization - DoWork() will trigger this when complete
    WorkCompletedEvent = FPlatformProcess::GetSynchEventFromPool(true);
}

FBeatsyncProcessingTask::~FBeatsyncProcessingTask()
{
    // Invalidate progress guard so any in-flight callbacks won't attempt to call back into this object
    if (ProgressGuard) {
        ProgressGuard->AtomicSet(true);
    }

    // Request cancellation to unblock waiting threads
    bCancelRequested = true;

    // Wait for work completion with timeout to avoid indefinite hangs
    bool bCompleted = false;
    constexpr uint32 TimeoutMs = 10000; // 10 second timeout
    if (WorkCompletedEvent)
    {
        if (!WorkCompletedEvent->Wait(TimeoutMs))
        {
            UE_LOG(LogTemp, Warning, TEXT("FBeatsyncProcessingTask: Timeout waiting for DoWork to complete"));
        }
        else
        {
            bCompleted = true;
        }
        FPlatformProcess::ReturnSynchEventToPool(WorkCompletedEvent);
        WorkCompletedEvent = nullptr;
    }

    // Only destroy writer if work completed, otherwise we risk use-after-free race
    if (bCompleted && Writer)
    {
        FBeatsyncLoader::SetProgressCallback(Writer, nullptr);
        FBeatsyncLoader::DestroyVideoWriter(Writer);
        Writer = nullptr;
    }
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

    // Check if backend is initialized FIRST (before creating span)
    if (!FBeatsyncLoader::IsInitialized())
    {
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("Backend not loaded");
        SignalWorkComplete();
        auto LocalOnComplete = OnComplete;
        AsyncTask(ENamedThreads::GameThread, [LocalOnComplete, Result]() {
            LocalOnComplete.ExecuteIfBound(Result);
        });
        return;
    }

    // Create top-level span for the processing job (only after init check)
    auto Span = FBeatsyncLoader::StartSpan(TEXT("BeatsyncProcessingTask"));

    // Step 1: Analyze audio
    ReportProgress(0.05f, TEXT("Analyzing audio..."));

    if (bCancelRequested)
    {
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("Cancelled");
        if (Span) {
            FBeatsyncLoader::SpanAddEvent(Span, TEXT("cancelled-before-analyze"));
            FBeatsyncLoader::SpanSetError(Span, Result.ErrorMessage);
            FBeatsyncLoader::EndSpan(Span);
        }
        SignalWorkComplete();
        auto LocalOnComplete = OnComplete;
        AsyncTask(ENamedThreads::GameThread, [LocalOnComplete, Result]() {
            LocalOnComplete.ExecuteIfBound(Result);
        });
        return;
    }

    auto Analyzer = FBeatsyncLoader::CreateAnalyzer();
    if (!Analyzer)
    {
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("Failed to create analyzer");
        if (Span) {
            FBeatsyncLoader::SpanSetError(Span, Result.ErrorMessage);
            FBeatsyncLoader::EndSpan(Span);
        }
        SignalWorkComplete();
        auto LocalOnComplete = OnComplete;
        AsyncTask(ENamedThreads::GameThread, [LocalOnComplete, Result]() {
            LocalOnComplete.ExecuteIfBound(Result);
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
        if (Span) FBeatsyncLoader::SpanSetError(Span, Result.ErrorMessage);
        if (Span) FBeatsyncLoader::EndSpan(Span);
        SignalWorkComplete();
        auto LocalOnComplete = OnComplete;
        AsyncTask(ENamedThreads::GameThread, [LocalOnComplete, Result]() {
            LocalOnComplete.ExecuteIfBound(Result);
        });
        return;
    }

    Result.BPM = BeatGrid.BPM;
    Result.BeatTimes = BeatGrid.Beats;

    ReportProgress(0.2f, FString::Printf(TEXT("Found %d beats at %.1f BPM"), BeatGrid.Beats.Num(), BeatGrid.BPM));

    // Check for cancellation before video processing starts
    // Note: No temp files exist yet at this point, so no cleanup needed
    if (bCancelRequested)
    {
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("Cancelled");
        if (Span) {
            FBeatsyncLoader::SpanAddEvent(Span, TEXT("cancelled-before-video"));
            FBeatsyncLoader::SpanSetError(Span, Result.ErrorMessage);
            FBeatsyncLoader::EndSpan(Span);
        }
        SignalWorkComplete();
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

    // Validate FilteredBeats is not empty
    if (FilteredBeats.Num() == 0)
    {
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("No beats after filtering");
        if (Span) FBeatsyncLoader::SpanSetError(Span, Result.ErrorMessage);
        if (Span) FBeatsyncLoader::EndSpan(Span);
        SignalWorkComplete();
        auto LocalOnComplete = OnComplete;
        AsyncTask(ENamedThreads::GameThread, [LocalOnComplete, Result]() {
            LocalOnComplete.ExecuteIfBound(Result);
        });
        return;
    }

    // Step 3: Create video writer
    Writer = FBeatsyncLoader::CreateVideoWriter();
    if (!Writer)
    {
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("Failed to create video writer");
        if (Span) FBeatsyncLoader::SpanSetError(Span, Result.ErrorMessage);
        if (Span) FBeatsyncLoader::EndSpan(Span);
        SignalWorkComplete();
        auto LocalOnComplete = OnComplete;
        AsyncTask(ENamedThreads::GameThread, [LocalOnComplete, Result]() {
            LocalOnComplete.ExecuteIfBound(Result);
        });
        return;
    }

    // Set up progress callback for video processing using a weak guard to avoid calling into destroyed task
    // Capture delegate by value to avoid dangling pointer to 'this'
    TWeakPtr<FThreadSafeBool> weakGuard = ProgressGuard;
    FOnBeatsyncProcessingProgress LocalOnProgress = OnProgress;
    FBeatsyncLoader::SetProgressCallback(Writer, [weakGuard, LocalOnProgress](double Prog) {
        TSharedPtr<FThreadSafeBool> guard = weakGuard.Pin();
        if (!guard) return; // Guard is gone (task destroyed)
        if (*guard) return; // invalidated or cancelled
        if (LocalOnProgress.IsBound()) {
            float Progress = 0.2f + 0.5f * static_cast<float>(Prog);
            FString Status = TEXT("Processing video...");
            AsyncTask(ENamedThreads::GameThread, [LocalOnProgress, Progress, Status]() {
                LocalOnProgress.ExecuteIfBound(Progress, Status);
            });
        }
    });

    // Step 4: Cut video at beats
    ReportProgress(0.25f, TEXT("Cutting video at beats..."));

    double ClipDuration = FilteredBeats.Num() > 1 ? (FilteredBeats[1] - FilteredBeats[0]) : 1.0;

    // Create temp file for video-only output
    TempVideoPath = Params.OutputPath + TEXT(".temp_video.mp4");
    TempEffectsPath = Params.OutputPath + TEXT(".temp_effects.mp4");

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
        if (Span) FBeatsyncLoader::SpanSetError(Span, Result.ErrorMessage);
        FBeatsyncLoader::SetProgressCallback(Writer, nullptr);
        FBeatsyncLoader::DestroyVideoWriter(Writer);
        Writer = nullptr;
        IFileManager::Get().Delete(*TempVideoPath, false, true, true);
        TempVideoPath.Empty();
        if (Span) FBeatsyncLoader::EndSpan(Span);
        SignalWorkComplete();
        auto LocalOnComplete = OnComplete;
        AsyncTask(ENamedThreads::GameThread, [LocalOnComplete, Result]() {
            LocalOnComplete.ExecuteIfBound(Result);
        });
        return;
    }

    if (bCancelRequested)
    {
        if (Span) FBeatsyncLoader::SpanAddEvent(Span, TEXT("cancelled-after-cut"));
        FBeatsyncLoader::SetProgressCallback(Writer, nullptr);
        FBeatsyncLoader::DestroyVideoWriter(Writer);
        Writer = nullptr;
        IFileManager::Get().Delete(*TempVideoPath, false, true, true);
        TempVideoPath.Empty();
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("Cancelled");
        if (Span) { FBeatsyncLoader::SpanSetError(Span, Result.ErrorMessage); FBeatsyncLoader::EndSpan(Span); }
        SignalWorkComplete();
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
            TempEffectsPath.Empty();  // Clear to prevent double deletion
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("TripSitter: Effects application failed, continuing without effects"));
            if (Span) FBeatsyncLoader::SpanAddEvent(Span, TEXT("effects-failed"));
        }
    }

    if (bCancelRequested)
    {
        FBeatsyncLoader::SetProgressCallback(Writer, nullptr);
        FBeatsyncLoader::DestroyVideoWriter(Writer);
        Writer = nullptr;
        IFileManager::Get().Delete(*CurrentVideoPath, false, true, true);
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("Cancelled");
        if (Span) { FBeatsyncLoader::SpanSetError(Span, Result.ErrorMessage); FBeatsyncLoader::EndSpan(Span); }
        SignalWorkComplete();
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
        // Invalidate temp paths so later cleanup does not attempt to delete files that were moved
        // If effects were applied, CurrentVideoPath == TempEffectsPath, so empty both
        if (CurrentVideoPath == TempEffectsPath)
        {
            TempEffectsPath.Empty();
        }
        TempVideoPath.Empty();
        CurrentVideoPath.Empty();
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
    }

    FBeatsyncLoader::SetProgressCallback(Writer, nullptr);
    FBeatsyncLoader::DestroyVideoWriter(Writer);
    Writer = nullptr;

    // Report completion
    Result.bSuccess = bSuccess;
    if (!bSuccess)
    {
        Result.ErrorMessage = TEXT("Processing failed");
        if (Span) FBeatsyncLoader::SpanSetError(Span, Result.ErrorMessage);
    }

    ReportProgress(1.0f, bSuccess ? TEXT("Complete!") : TEXT("Failed"));

    if (Span) FBeatsyncLoader::EndSpan(Span);

    SignalWorkComplete();
    auto LocalOnComplete = OnComplete;
    AsyncTask(ENamedThreads::GameThread, [LocalOnComplete, Result]() {
        LocalOnComplete.ExecuteIfBound(Result);
    });
}
