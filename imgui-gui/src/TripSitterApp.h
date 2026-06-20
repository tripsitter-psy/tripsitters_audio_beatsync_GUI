// TripSitterApp - ImGui port of STripSitterMainWidget.
//
// Holds the full UI/processing state (ported from the Slate widget) and renders
// the interface with Dear ImGui each frame. All audio/video work is delegated to
// the beatsync backend through BackendLoader, exactly like the Slate app did via
// the Unreal BeatsyncLoader.
#pragma once

#include "BackendLoader.h"

#include <string>
#include <vector>
#include <array>
#include <thread>
#include <atomic>
#include <mutex>

// ---- Configuration enums (mirror STripSitterMainWidget.h) -----------------
enum class BeatRate    { Every = 0, Every2nd, Every4th, Every8th };
enum class AnalysisMode{ Energy = 0, AIBeat, AIStems, AudioFlux, StemsFlux };
enum class Resolution  { HD1080 = 0, HD720, UHD4K, QHD2K };
enum class Fps         { FPS24 = 0, FPS30, FPS60 };
enum class StemEffect  { None = 0, Flash, Zoom, Vignette, ColorGrade };
enum class ColorPreset { Warm = 0, Cool, Vintage, Vibrant };
enum class TransitionType { Fade = 0, Wipe, Dissolve, Zoom };

constexpr int STEM_COUNT = 4; // Kick, Snare, HiHat, Synth

struct StemConfig
{
    std::string         filePath;
    StemEffect          effect = StemEffect::None;
    std::vector<double> beatTimes;
    bool                enabled = false;
};

class TripSitterApp
{
public:
    explicit TripSitterApp(BackendLoader& backend);
    ~TripSitterApp();

    // Draw the entire UI for one frame into the current ImGui context.
    void Draw();

private:
    BackendLoader& Backend;

    // ---- File paths ----
    std::string AudioPath;
    std::string VideoPath;
    std::vector<std::string> VideoPaths;
    std::string OutputPath;
    bool bIsMultiClip = false;

    // ---- Processing state ----
    float  Progress = 0.0f;
    std::string StatusText = "Ready";
    std::string ETAText;
    std::atomic<bool> bIsProcessing{false};
    bool   bAudioAnalyzed = false;
    double DetectedBPM = 0.0;
    double OriginalFirstBeatTime = 0.0;
    std::vector<double> AnalyzedBeatTimes;

    // ---- Background job system --------------------------------------------
    // Heavy backend work (analysis, video processing) runs on a worker thread.
    // The worker only writes into Pending/WorkerStatus (guarded by JobMutex) and
    // the atomics below; the UI thread applies results in PumpJobs() each frame.
    enum class JobType { None, Analyze, Sync };
    std::thread        Worker;
    JobType            ActiveJob = JobType::None;
    std::atomic<bool>  bJobDone{false};      // worker finished; results awaiting apply
    std::atomic<float> JobProgress{0.0f};    // 0..1, updated from backend callbacks
    std::atomic<int>   CancelFlag{0};        // set to 1 by UI; read by backend/worker

    std::mutex         JobMutex;
    std::string        WorkerStatus;         // live status line during a job (guarded)
    struct JobResult {
        bool ok = false;
        std::string status;
        // Analyze results:
        std::vector<double> beats;
        double bpm = 0.0;
        double duration = 0.0;
    } Pending;                               // guarded by JobMutex

    // Immutable snapshot of UI inputs captured when a job starts, so the worker
    // never reads fields the user may still be editing.
    struct JobInput {
        std::string audioPath, videoPath, outputPath;
        std::vector<std::string> videoPaths;
        bool multiClip = false;
        AnalysisMode mode = AnalysisMode::Energy;
        std::vector<double> beats;           // effective beats for a sync job
        double selStart = 0.0, selEnd = -1.0, duration = 0.0;
    } Job;

    // ---- Configuration ----
    BeatRate      BeatRateSel    = BeatRate::Every;
    AnalysisMode  AnalysisModeSel= AnalysisMode::AIBeat;
    Resolution    ResolutionSel  = Resolution::HD1080;
    Fps           FpsSel         = Fps::FPS30;

    // ---- Effects config ----
    bool  bEnableVignette = false;
    bool  bEnableBeatFlash = false;
    bool  bEnableBeatZoom = false;
    bool  bEnableColorGrade = false;
    bool  bEnableTransitions = false;
    float FlashIntensity = 0.5f;
    float ZoomIntensity = 0.1f;
    float VignetteStrength = 0.3f;
    float TransitionDuration = 0.5f;
    ColorPreset    ColorPresetSel = ColorPreset::Warm;
    TransitionType TransitionTypeSel = TransitionType::Fade;

    // ---- Stems ----
    std::array<StemConfig, STEM_COUNT> StemConfigs;

    // ---- Waveform / selection ----
    std::vector<float> WaveformPeaks;     // mono peaks for display
    std::vector<float> BandBass, BandMid, BandHigh; // frequency-colored bands
    double AudioDuration = 0.0;
    double SelectionStart = 0.0;
    double SelectionEnd = -1.0;
    float  WaveformScroll = 0.0f;         // horizontal scroll [0,1]
    float  WaveformZoom = 1.0f;

    // ---- Section renderers ----
    void DrawHeader();
    void DrawFileSection();
    void DrawWaveformSection();
    void DrawAnalysisSection();
    void DrawEffectsSection();
    void DrawTransitionsSection();
    void DrawStemsSection();
    void DrawControlSection();
    void DrawStatusBar();

    // ---- Actions (delegate to backend) ----
    void LoadWaveform(const std::string& path);   // synchronous (fast)
    void Cancel();
    void RecalculateBeatsFromBPM(double newBPM);

    // Job lifecycle: Start* (UI thread) spawn a worker that runs Run* and posts
    // results; PumpJobs() applies finished results on the UI thread.
    void StartAnalyzeJob();
    void RunAnalyzeJob();    // worker thread
    void StartSyncJob();
    void RunSyncJob();       // worker thread
    void PumpJobs();
    bool JobBusy() const { return bIsProcessing.load(); }

    // Backend progress callbacks (run on worker thread; only touch atomics/guarded state).
    static void VideoProgressThunk(double progress, void* user_data);
    static int  AiProgressThunk(float progress, const char* stage, const char* message, void* user_data);

    // ---- Helpers ----
    std::vector<double> EffectiveBeatTimes() const; // applies BeatRate divisor + selection
    const char* AnalysisModeAvailability(AnalysisMode m) const; // nullptr if available, else reason
};
