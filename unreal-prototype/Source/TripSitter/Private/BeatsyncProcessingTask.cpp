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
    if (IsCancelled())
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

        // Compute duration: use AudioEnd if set, otherwise compute from last beat + some buffer
        if (Params.AudioEnd > 0)
        {
            BeatGrid.Duration = Params.AudioEnd - Params.AudioStart;
        }
        else if (Params.PreAnalyzedBeatTimes.Num() > 0)
        {
            // Use last beat time + 1 second as duration estimate
            BeatGrid.Duration = Params.PreAnalyzedBeatTimes.Last() + 1.0;
        }
        else
        {
            BeatGrid.Duration = 0.0;
        }

        bSuccess = true;
        UE_LOG(LogTemp, Log, TEXT("TripSitter: Using %d user-edited beat markers at %.1f BPM, duration=%.2f"),
            BeatGrid.Beats.Num(), BeatGrid.BPM, BeatGrid.Duration);
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
        bool bEnableStemSeparation = (Params.AnalysisMode == EAnalysisModeParam::AIStems);

        // Try AI analyzer if requested and available
        if (bUseAI && FBeatsyncLoader::IsAIAvailable())
        {
            FString ModeStr = bEnableStemSeparation ? TEXT("AI + Stems") : TEXT("AI Beat");
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
            if (bEnableStemSeparation)
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
                AIConfig.bEnableGPU = true;  // Enable CUDA
                AIConfig.bEnableStemSeparation = bEnableStemSeparation;
                AIConfig.bEnableDrumsForBeats = true;
                AIConfig.BeatThreshold = 0.66f;
                AIConfig.DownbeatThreshold = 0.66f;

                // Set stem model path if using stem separation
                if (bEnableStemSeparation && FPaths::FileExists(StemModelPath))
                {
                    AIConfig.StemModelPath = StemModelPath;
                    UE_LOG(LogTemp, Log, TEXT("TripSitter: Using stem separation model: %s"), *StemModelPath);
                }
                else if (bEnableStemSeparation)
                {
                    UE_LOG(LogTemp, Warning, TEXT("TripSitter: Stem model not found at %s, stem separation disabled"), *StemModelPath);
                    AIConfig.bEnableStemSeparation = false;
                }

                void* AIAnalyzer = FBeatsyncLoader::CreateAIAnalyzer(AIConfig);
                if (AIAnalyzer)
                {
                    // RAII guard to ensure AI analyzer is always destroyed
                    ON_SCOPE_EXIT
                    {
                        FBeatsyncLoader::DestroyAIAnalyzer(AIAnalyzer);
                    };

                    FAIResult AIResult;
                    // Use full analysis with stem separation, or quick mode without
                    if (AIConfig.bEnableStemSeparation)
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

    if (IsCancelled())
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

    // Check if we're using pre-analyzed (user-selected) beat markers
    // If so, the beats are already the exact ones the user wants - skip range filtering
    bool bUsingPreAnalyzedBeats = (Params.PreAnalyzedBeatTimes.Num() > 0);

    double SelectionStart = Params.AudioStart;
    double SelectionEnd = Params.AudioEnd > 0 ? Params.AudioEnd : BeatGrid.Duration;

    // DIAGNOSTIC: Log beat filtering params
    {
        FString DiagPath = FPaths::Combine(FPlatformMisc::GetEnvironmentVariable(TEXT("TEMP")), TEXT("beatsync_ue_beatfilter.log"));
        FString DiagContent;
        DiagContent += FString::Printf(TEXT("=== BEAT FILTER DIAGNOSTIC ===\n"));
        DiagContent += FString::Printf(TEXT("bUsingPreAnalyzedBeats: %d\n"), bUsingPreAnalyzedBeats ? 1 : 0);
        DiagContent += FString::Printf(TEXT("Params.PreAnalyzedBeatTimes.Num(): %d\n"), Params.PreAnalyzedBeatTimes.Num());
        DiagContent += FString::Printf(TEXT("Params.AudioStart: %.6f\n"), Params.AudioStart);
        DiagContent += FString::Printf(TEXT("Params.AudioEnd: %.6f\n"), Params.AudioEnd);
        DiagContent += FString::Printf(TEXT("BeatGrid.Duration: %.6f\n"), BeatGrid.Duration);
        DiagContent += FString::Printf(TEXT("SelectionStart: %.6f\n"), SelectionStart);
        DiagContent += FString::Printf(TEXT("SelectionEnd: %.6f\n"), SelectionEnd);
        DiagContent += FString::Printf(TEXT("BeatGrid.Beats.Num(): %d\n"), BeatGrid.Beats.Num());
        DiagContent += FString::Printf(TEXT("BeatDivisor: %d\n"), BeatDivisor);
        if (BeatGrid.Beats.Num() > 0) {
            DiagContent += FString::Printf(TEXT("First beat: %.6f\n"), BeatGrid.Beats[0]);
            DiagContent += FString::Printf(TEXT("Last beat: %.6f\n"), BeatGrid.Beats.Last());
        }
        FFileHelper::SaveStringToFile(DiagContent, *DiagPath);
    }

    if (bUsingPreAnalyzedBeats)
    {
        // User-selected beat markers are already exactly what we want
        // Just apply beat divisor (every beat, every 2nd, etc.) - no range filtering
        // The beat times are already relative to the audio selection start
        for (int32 i = 0; i < BeatGrid.Beats.Num(); ++i)
        {
            if ((i % BeatDivisor) == 0)
            {
                FilteredBeats.Add(BeatGrid.Beats[i]);
            }
        }
        UE_LOG(LogTemp, Log, TEXT("TripSitter: Using %d pre-analyzed beats directly (divisor %d -> %d filtered)"),
            BeatGrid.Beats.Num(), BeatDivisor, FilteredBeats.Num());
    }
    else
    {
        // Standard analysis - apply range filtering
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

        UE_LOG(LogTemp, Warning, TEXT("TripSitter: DIAG - About to call NormalizeVideos with %d videos"), Params.VideoPaths.Num());
        if (FBeatsyncLoader::NormalizeVideos(Writer, Params.VideoPaths, NormalizedVideos))
        {
            UE_LOG(LogTemp, Warning, TEXT("TripSitter: DIAG - NormalizeVideos returned TRUE, NormalizedVideos.Num()=%d"), NormalizedVideos.Num());
            if (NormalizedVideos.Num() == Params.VideoPaths.Num())
            {
                VideosToProcess = NormalizedVideos;
                bUsingNormalized = true;
                UE_LOG(LogTemp, Warning, TEXT("TripSitter: Using %d normalized videos"), NormalizedVideos.Num());
                // Log first few paths for debugging
                for (int32 i = 0; i < FMath::Min(3, NormalizedVideos.Num()); ++i)
                {
                    UE_LOG(LogTemp, Warning, TEXT("TripSitter: DIAG - NormalizedVideos[%d] = %s"), i, *NormalizedVideos[i]);
                }
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
            UE_LOG(LogTemp, Warning, TEXT("TripSitter: DIAG - NormalizeVideos returned FALSE"));
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

    // File-based diagnostic - this MUST run
    {
        FString DiagPath = FPaths::Combine(FPlatformMisc::GetEnvironmentVariable(TEXT("TEMP")), TEXT("beatsync_ue_precut.log"));
        FString DiagContent;
        DiagContent += FString::Printf(TEXT("=== PRE-CUT DIAGNOSTIC ===\n"));
        DiagContent += FString::Printf(TEXT("bIsMultiClip: %d\n"), Params.bIsMultiClip ? 1 : 0);
        DiagContent += FString::Printf(TEXT("VideosToProcess.Num(): %d\n"), VideosToProcess.Num());
        DiagContent += FString::Printf(TEXT("FilteredBeats.Num(): %d\n"), FilteredBeats.Num());
        DiagContent += FString::Printf(TEXT("ClipDuration: %.6f\n"), ClipDuration);
        DiagContent += FString::Printf(TEXT("TempVideoPath: %s\n"), *TempVideoPath);
        DiagContent += FString::Printf(TEXT("Will take multi path: %s\n"), (Params.bIsMultiClip && VideosToProcess.Num() > 1) ? TEXT("YES") : TEXT("NO"));
        if (VideosToProcess.Num() > 0) {
            DiagContent += FString::Printf(TEXT("First video: %s\n"), *VideosToProcess[0]);
        }
        FFileHelper::SaveStringToFile(DiagContent, *DiagPath);
    }

    // DIAGNOSTIC: Log the exact code path being taken
    UE_LOG(LogTemp, Warning, TEXT("TripSitter: CUT DIAGNOSTIC - bIsMultiClip=%d, VideosToProcess.Num()=%d, FilteredBeats.Num()=%d, ClipDuration=%.6f"),
        Params.bIsMultiClip ? 1 : 0, VideosToProcess.Num(), FilteredBeats.Num(), ClipDuration);
    UE_LOG(LogTemp, Warning, TEXT("TripSitter: CUT DIAGNOSTIC - TempVideoPath=%s"), *TempVideoPath);

    if (Params.bIsMultiClip && VideosToProcess.Num() > 1)
    {
        UE_LOG(LogTemp, Warning, TEXT("TripSitter: TAKING MULTI-VIDEO PATH with %d videos"), VideosToProcess.Num());
        // Log first video path to verify it exists
        if (VideosToProcess.Num() > 0)
        {
            UE_LOG(LogTemp, Warning, TEXT("TripSitter: DIAG - First video path: %s"), *VideosToProcess[0]);
            // Check if file exists
            bool bExists = FPaths::FileExists(VideosToProcess[0]);
            UE_LOG(LogTemp, Warning, TEXT("TripSitter: DIAG - First video exists: %s"), bExists ? TEXT("YES") : TEXT("NO"));
        }
        bSuccess = FBeatsyncLoader::CutVideoAtBeatsMulti(Writer, VideosToProcess, FilteredBeats, TempVideoPath, ClipDuration);
        UE_LOG(LogTemp, Warning, TEXT("TripSitter: CutVideoAtBeatsMulti returned %s"), bSuccess ? TEXT("SUCCESS") : TEXT("FAILURE"));
    }
    else
    {
        FString SingleVideo = VideosToProcess.Num() > 0 ? VideosToProcess[0] : Params.VideoPath;
        UE_LOG(LogTemp, Warning, TEXT("TripSitter: TAKING SINGLE-VIDEO PATH with video=%s"), *SingleVideo);
        bSuccess = FBeatsyncLoader::CutVideoAtBeats(Writer, SingleVideo, FilteredBeats, TempVideoPath, ClipDuration);
        UE_LOG(LogTemp, Warning, TEXT("TripSitter: CutVideoAtBeats returned %s"), bSuccess ? TEXT("SUCCESS") : TEXT("FAILURE"));
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
