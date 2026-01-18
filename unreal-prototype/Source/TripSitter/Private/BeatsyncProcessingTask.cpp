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
    , SharedCancelFlag(MakeShared<FThreadSafeBool>(false))
{
}

FBeatsyncProcessingTask::~FBeatsyncProcessingTask()
{
    // Clean up the video writer if it was created
    if (Writer)
    {
        // Clear callback first to prevent use-after-free - callback captures &bCancelRequested
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

    // Step 1: Get beat times (either from pre-analyzed UI markers or by analyzing audio)
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

    // Check if we have pre-analyzed beat times from the UI (user-edited markers)
    if (Params.PreAnalyzedBeatTimes.Num() > 0)
    {
        // Use the user-edited beat markers instead of re-analyzing
        ReportProgress(0.05f, TEXT("Using edited beat markers..."));
        BeatGrid.Beats = Params.PreAnalyzedBeatTimes;
        BeatGrid.BPM = Params.PreAnalyzedBPM > 0 ? Params.PreAnalyzedBPM : 120.0;
        BeatGrid.Duration = Params.AudioEnd > 0 ? (Params.AudioEnd - Params.AudioStart) : 0.0;
        bSuccess = true;
        UE_LOG(LogTemp, Log, TEXT("TripSitter: Using %d user-edited beat markers at %.1f BPM"),
            BeatGrid.Beats.Num(), BeatGrid.BPM);
    }
    else
    {
        // No pre-analyzed beats - analyze the audio
        ReportProgress(0.05f, TEXT("Analyzing audio..."));
    }

    // Only analyze audio if we don't have pre-analyzed beats from the UI
    if (!bSuccess)
    {
        // Use analysis mode from UI params
        bool bUseAI = (Params.AnalysisMode == EAnalysisModeParam::AIBeat || Params.AnalysisMode == EAnalysisModeParam::AIStems);
        bool bUseStemSeparation = (Params.AnalysisMode == EAnalysisModeParam::AIStems);

        // Try AI analyzer if requested and available
        if (bUseAI && FBeatsyncLoader::IsAIAvailable())
    {
        FString ModeStr = bUseStemSeparation ? TEXT("AI + Stems") : TEXT("AI Beat");
        ReportProgress(0.05f, FString::Printf(TEXT("Analyzing audio with %s (GPU)..."), *ModeStr));
        UE_LOG(LogTemp, Log, TEXT("TripSitter: Using %s analyzer (providers: %s)"), *ModeStr, *FBeatsyncLoader::GetAIProviders());

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

        // Get path to stem separation model (demucs) if using AI+Stems mode
        FString StemModelPath;
        if (bUseStemSeparation)
        {
            StemModelPath = FPaths::Combine(ExeDir, TEXT("models"), TEXT("demucs.onnx"));
            if (!FPaths::FileExists(StemModelPath))
            {
                StemModelPath = FPaths::Combine(ExeDir, TEXT(".."), TEXT(".."), TEXT("Source"), TEXT("Programs"),
                                                 TEXT("TripSitter"), TEXT("ThirdParty"), TEXT("beatsync"), TEXT("models"), TEXT("demucs.onnx"));
                StemModelPath = FPaths::ConvertRelativePathToFull(StemModelPath);
            }
        }

        if (FPaths::FileExists(ModelPath))
        {
            FAIConfig AIConfig;
            AIConfig.BeatModelPath = ModelPath;
            AIConfig.bUseGPU = true;  // Enable CUDA
            AIConfig.bUseStemSeparation = bUseStemSeparation;
            AIConfig.bUseDrumsForBeats = true;
            AIConfig.BeatThreshold = 0.66f;
            AIConfig.DownbeatThreshold = 0.66f;

            // Set stem model path if using stem separation
            if (bUseStemSeparation && FPaths::FileExists(StemModelPath))
            {
                AIConfig.StemModelPath = StemModelPath;
                UE_LOG(LogTemp, Log, TEXT("TripSitter: Using stem separation model: %s"), *StemModelPath);
            }
            else if (bUseStemSeparation)
            {
                UE_LOG(LogTemp, Warning, TEXT("TripSitter: Stem model not found at %s, stem separation disabled"), *StemModelPath);
                AIConfig.bUseStemSeparation = false;
            }

            void* AIAnalyzer = FBeatsyncLoader::CreateAIAnalyzer(AIConfig);
            if (AIAnalyzer)
            {
                FAIResult AIResult;
                // Use full analysis with stem separation, or quick mode without
                if (AIConfig.bUseStemSeparation)
                {
                    bSuccess = FBeatsyncLoader::AIAnalyzeFile(AIAnalyzer, Params.AudioPath, AIResult);
                }
                else
                {
                    bSuccess = FBeatsyncLoader::AIAnalyzeQuick(AIAnalyzer, Params.AudioPath, AIResult);
                }

                if (bSuccess && AIResult.Beats.Num() > 0)
                {
                    BeatGrid.Beats = AIResult.Beats;
                    BeatGrid.BPM = AIResult.BPM;
                    BeatGrid.Duration = AIResult.Duration;
                    UE_LOG(LogTemp, Log, TEXT("TripSitter: %s analysis found %d beats at %.1f BPM"), *ModeStr, AIResult.Beats.Num(), AIResult.BPM);
                }
                else
                {
                    FString AIError = FBeatsyncLoader::GetAILastError(AIAnalyzer);
                    UE_LOG(LogTemp, Warning, TEXT("TripSitter: %s analysis failed: %s, falling back to spectral flux"), *ModeStr, *AIError);
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

        // Fall back to CPU-based spectral flux if AI not available, not requested, or failed
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
    } // End of if (!bSuccess) - skip audio analysis if we have pre-analyzed beats

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

    // Step 2: Apply beat rate filter AND selection range filter
    // This is critical for performance - only process beats within the selected range
    TArray<double> FilteredBeats;
    int32 ClampedBeatRate = FMath::Clamp(Params.BeatRate, 0, 3); // Clamp to safe range to prevent overflow
    int32 BeatDivisor = 1 << ClampedBeatRate; // 1, 2, 4, 8

    double SelectionStart = Params.AudioStart;
    double SelectionEnd = Params.AudioEnd > 0 ? Params.AudioEnd : BeatGrid.Duration;

    int32 BeatIndex = 0;
    for (int32 i = 0; i < BeatGrid.Beats.Num(); ++i)
    {
        double BeatTime = BeatGrid.Beats[i];

        // Skip beats outside selection range
        if (BeatTime < SelectionStart)
        {
            continue;
        }
        if (BeatTime > SelectionEnd)
        {
            break; // All remaining beats are past the selection
        }

        // Apply beat divisor (every beat, every 2nd, etc.)
        if ((BeatIndex % BeatDivisor) == 0)
        {
            // Offset beat time relative to selection start for the output video
            FilteredBeats.Add(BeatTime - SelectionStart);
        }
        BeatIndex++;
    }

    UE_LOG(LogTemp, Log, TEXT("TripSitter: Selection range %.2f - %.2f, filtered %d beats to %d"),
        SelectionStart, SelectionEnd, BeatGrid.Beats.Num(), FilteredBeats.Num());

    Result.BeatCount = FilteredBeats.Num();

    // Step 3: Create video writer
    Writer = FBeatsyncLoader::CreateVideoWriter();
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
    // Note: SharedCancelFlag is a shared member that reflects runtime cancellation state
    // CRITICAL: This callback is called from a worker thread in the backend DLL,
    // but Slate UI can ONLY be updated from the GameThread. Must marshal!
    auto LocalOnProgress = OnProgress;
    FBeatsyncLoader::SetProgressCallback(Writer, [LocalOnProgress, SharedCancelFlag = this->SharedCancelFlag](double Prog) {
        if (!(*SharedCancelFlag))
        {
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

    // Step 3.5: Pre-normalize videos to common format (reduces CUDA memory issues)
    TArray<FString> VideosToProcess;
    TArray<FString> NormalizedVideos;
    bool bUsingNormalized = false;

    if (Params.bIsMultiClip && Params.VideoPaths.Num() > 1)
    {
        ReportProgress(0.22f, TEXT("Normalizing videos..."));
        UE_LOG(LogTemp, Log, TEXT("TripSitter: Normalizing %d source videos"), Params.VideoPaths.Num());

        if (FBeatsyncLoader::NormalizeVideos(Writer, Params.VideoPaths, NormalizedVideos))
        {
            if (NormalizedVideos.Num() == Params.VideoPaths.Num())
            {
                VideosToProcess = NormalizedVideos;
                bUsingNormalized = true;
                UE_LOG(LogTemp, Log, TEXT("TripSitter: Using %d normalized videos"), NormalizedVideos.Num());
            }
            else
            {
                UE_LOG(LogTemp, Warning, TEXT("TripSitter: Normalization returned %d paths but expected %d, using original videos"),
                    NormalizedVideos.Num(), Params.VideoPaths.Num());
                VideosToProcess = Params.VideoPaths;
            }
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("TripSitter: Video normalization failed, using original videos"));
            VideosToProcess = Params.VideoPaths;
        }
    }
    else
    {
        // Single video - no normalization needed
        VideosToProcess.Add(Params.VideoPaths.Num() > 0 ? Params.VideoPaths[0] : Params.VideoPath);
    }

    // Cut video at beats
    ReportProgress(0.25f, TEXT("Cutting video at beats..."));

    if (Params.bIsMultiClip && VideosToProcess.Num() > 1)
    {
        UE_LOG(LogTemp, Log, TEXT("TripSitter: Processing %d videos"), VideosToProcess.Num());
        bSuccess = FBeatsyncLoader::CutVideoAtBeatsMulti(Writer, VideosToProcess, FilteredBeats, TempVideoPath, ClipDuration);
    }
    else
    {
        FString SingleVideo = VideosToProcess.Num() > 0 ? VideosToProcess[0] : Params.VideoPath;
        bSuccess = FBeatsyncLoader::CutVideoAtBeats(Writer, SingleVideo, FilteredBeats, TempVideoPath, ClipDuration);
    }

    if (!bSuccess)
    {
        FString ErrorMsg = FBeatsyncLoader::GetVideoLastError(Writer);
        Result.bSuccess = false;
        Result.ErrorMessage = ErrorMsg.IsEmpty() ? TEXT("Failed to cut video") : ErrorMsg;
        FBeatsyncLoader::SetProgressCallback(Writer, nullptr);  // Clear callback before destroy to prevent UAF
        FBeatsyncLoader::DestroyVideoWriter(Writer);
        Writer = nullptr;  // Prevent double-free in destructor
        IFileManager::Get().Delete(*TempVideoPath, false, true, true);
        if (bUsingNormalized) FBeatsyncLoader::CleanupNormalizedVideos(NormalizedVideos);
        auto LocalOnComplete = OnComplete;
        AsyncTask(ENamedThreads::GameThread, [LocalOnComplete, Result]() {
            LocalOnComplete.ExecuteIfBound(Result);
        });
        return;
    }

    if (bCancelRequested)
    {
        FBeatsyncLoader::SetProgressCallback(Writer, nullptr);  // Clear callback before destroy to prevent UAF
        FBeatsyncLoader::DestroyVideoWriter(Writer);
        Writer = nullptr;  // Prevent double-free in destructor
        IFileManager::Get().Delete(*TempVideoPath, false, true, true);
        if (bUsingNormalized) FBeatsyncLoader::CleanupNormalizedVideos(NormalizedVideos);
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
        FBeatsyncLoader::SetProgressCallback(Writer, nullptr);  // Clear callback before destroy to prevent UAF
        FBeatsyncLoader::DestroyVideoWriter(Writer);
        Writer = nullptr;  // Prevent double-free in destructor
        IFileManager::Get().Delete(*CurrentVideoPath, false, true, true);
        if (bUsingNormalized) FBeatsyncLoader::CleanupNormalizedVideos(NormalizedVideos);
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
        Result.bAudioMuxFailed = true;  // Mark partial success so UI can warn user
        bSuccess = true; // Consider it a partial success
    }

    // Clean up temp files
    IFileManager::Get().Delete(*TempVideoPath, false, true, true);
    IFileManager::Get().Delete(*TempEffectsPath, false, true, true);

    // Clean up normalized videos if we created them
    if (bUsingNormalized && NormalizedVideos.Num() > 0)
    {
        UE_LOG(LogTemp, Log, TEXT("TripSitter: Cleaning up %d normalized video files"), NormalizedVideos.Num());
        FBeatsyncLoader::CleanupNormalizedVideos(NormalizedVideos);
    }

    FBeatsyncLoader::SetProgressCallback(Writer, nullptr);  // Clear callback before destroy to prevent UAF
    FBeatsyncLoader::DestroyVideoWriter(Writer);
    Writer = nullptr;  // Prevent double-free in destructor

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
