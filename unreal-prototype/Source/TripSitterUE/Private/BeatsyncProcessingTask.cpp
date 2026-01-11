#include "BeatsyncProcessingTask.h"
#include "Async/Async.h"
#include "HAL/FileManager.h"
#include "Misc/Paths.h"
#include <memory>

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

    // Create top-level span for the processing job
    auto Span = FBeatsyncLoader::StartSpan(TEXT("BeatsyncProcessingTask"));

    // Check if backend is initialized
    if (!FBeatsyncLoader::IsInitialized())
    {
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("Backend not loaded");
        if (Span) FBeatsyncLoader::SpanSetError(Span, Result.ErrorMessage);
        if (Span) FBeatsyncLoader::EndSpan(Span);
        auto LocalOnComplete = OnComplete;
        AsyncTask(ENamedThreads::GameThread, [LocalOnComplete, Result]() {
            LocalOnComplete.ExecuteIfBound(Result);
        });
        return;
    }

    // Step 1: Analyze audio
    ReportProgress(0.05f, TEXT("Analyzing audio..."));

    if (bCancelRequested.load())
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
        return;
    }

    Result.BPM = BeatGrid.BPM;
    Result.BeatTimes = BeatGrid.Beats;

    ReportProgress(0.2f, FString::Printf(TEXT("Found %d beats at %.1f BPM"), BeatGrid.Beats.Num(), BeatGrid.BPM));

    if (bCancelRequested.load())
    {
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("Cancelled");
        if (Span) { FBeatsyncLoader::SpanSetError(Span, Result.ErrorMessage); FBeatsyncLoader::EndSpan(Span); }
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
    auto Writer = FBeatsyncLoader::CreateVideoWriter();
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
        return;
    }

    // Set up progress callback for video processing
    auto SharedThis = std::shared_ptr<FBeatsyncProcessingTask>(this, [](FBeatsyncProcessingTask*){}); // Dummy deleter to prevent deletion
    std::weak_ptr<FBeatsyncProcessingTask> WeakThis = SharedThis;
    FBeatsyncLoader::SetProgressCallback(Writer, [WeakThis](double Prog) {
        if (auto Task = WeakThis.lock()) {
            if (!Task->bCancelRequested.load()) {
                Task->ReportProgress(0.2f + 0.5f * static_cast<float>(Prog), TEXT("Processing video..."));
            }
        }
    });

    // Step 4: Cut video at beats
    ReportProgress(0.25f, TEXT("Cutting video at beats..."));

    double ClipDuration = FilteredBeats.Num() > 1 ? (FilteredBeats[1] - FilteredBeats[0]) : 1.0;

    // Create temp file for video-only output
    FString TempVideoPath = Params.OutputPath + TEXT(".temp_video.mp4");
    FString TempEffectsPath = Params.OutputPath + TEXT(".temp_effects.mp4");

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
        FBeatsyncLoader::DestroyVideoWriter(Writer);
        IFileManager::Get().Delete(*TempVideoPath, false, true, true);
        TempVideoPath.Empty();
        if (Span) FBeatsyncLoader::EndSpan(Span);
        auto LocalOnComplete = OnComplete;
        AsyncTask(ENamedThreads::GameThread, [LocalOnComplete, Result]() {
            LocalOnComplete.ExecuteIfBound(Result);
        });
        return;
    }

    if (bCancelRequested.load())
    {
        if (Span) FBeatsyncLoader::SpanAddEvent(Span, TEXT("cancelled-after-cut"));
        FBeatsyncLoader::DestroyVideoWriter(Writer);
        IFileManager::Get().Delete(*TempVideoPath, false, true, true);
        TempVideoPath.Empty();
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("Cancelled");
        if (Span) { FBeatsyncLoader::SpanSetError(Span, Result.ErrorMessage); FBeatsyncLoader::EndSpan(Span); }
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
            if (Span) FBeatsyncLoader::SpanAddEvent(Span, TEXT("effects-failed"));
        }
    }

    if (bCancelRequested.load())
    {
        FBeatsyncLoader::DestroyVideoWriter(Writer);
        IFileManager::Get().Delete(*CurrentVideoPath, false, true, true);
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("Cancelled");
        if (Span) { FBeatsyncLoader::SpanSetError(Span, Result.ErrorMessage); FBeatsyncLoader::EndSpan(Span); }
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
        if (Span) FBeatsyncLoader::SpanAddEvent(Span, TEXT("audio-mux-failed"));
    }

    // Clean up temp files
    if (!TempVideoPath.IsEmpty())
    {
        IFileManager::Get().Delete(*TempVideoPath, false, true, true);
    }
    IFileManager::Get().Delete(*TempEffectsPath, false, true, true);

    FBeatsyncLoader::DestroyVideoWriter(Writer);

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
}
