#include "TripSitterApp.h"
#include "Theme.h"
#include "FileDialog.h"

#include "imgui.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>

namespace
{
// Resolve a model file inside the repo's models/ dir, searching upward from the
// current working directory so the app works whether launched from the build
// dir or the repo root. Returns an absolute path, or the bare relative path if
// not found (so the backend's own error reporting kicks in).
std::string ResolveModel(const char* name)
{
    namespace fs = std::filesystem;
    const char* bases[] = { "models/", "../models/", "../../models/", "../../../models/" };
    std::error_code ec;
    for (const char* b : bases)
    {
        fs::path p = fs::path(b) / name;
        if (fs::exists(p, ec))
            return fs::absolute(p, ec).string();
    }
    return std::string("models/") + name;
}
} // namespace

namespace
{
const char* kBeatRateLabels[]   = { "Every beat", "Every 2nd", "Every 4th", "Every 8th" };
const char* kAnalysisLabels[]   = { "Energy (Fast)", "AI Beat Detection", "AI + Stem Separation",
                                    "AudioFlux", "Stems + Flux (Best)" };
const char* kResolutionLabels[] = { "1920x1080 (HD)", "1280x720 (720p)", "3840x2160 (4K)", "2560x1440 (2K)" };
const char* kFpsLabels[]        = { "24 fps", "30 fps", "60 fps" };
const char* kColorLabels[]      = { "Warm", "Cool", "Vintage", "Vibrant" };
const char* kTransitionLabels[] = { "Fade", "Wipe", "Dissolve", "Zoom" };
const char* kStemNames[]        = { "Kick", "Snare", "Hi-Hat", "Synth" };
const char* kStemEffectLabels[] = { "None", "Flash", "Zoom", "Vignette", "Color Grade" };

int BeatDivisor(BeatRate r)
{
    switch (r) { case BeatRate::Every2nd: return 2; case BeatRate::Every4th: return 4;
                 case BeatRate::Every8th: return 8; default: return 1; }
}

const char* ColorPresetStr(ColorPreset p)
{ switch (p){case ColorPreset::Cool:return "cool";case ColorPreset::Vintage:return "vintage";
             case ColorPreset::Vibrant:return "vibrant";default:return "warm";} }

const char* TransitionStr(TransitionType t)
{ switch (t){case TransitionType::Wipe:return "wipe";case TransitionType::Dissolve:return "dissolve";
             case TransitionType::Zoom:return "zoom";default:return "fade";} }

// Generic combo that edits an enum stored as int.
template <typename E>
bool EnumCombo(const char* label, E& value, const char* const* items, int count)
{
    int idx = static_cast<int>(value);
    bool changed = ImGui::Combo(label, &idx, items, count);
    if (changed) value = static_cast<E>(idx);
    return changed;
}
} // namespace

TripSitterApp::TripSitterApp(BackendLoader& backend) : Backend(backend) {}

TripSitterApp::~TripSitterApp()
{
    // Request cancel and join any in-flight worker before teardown.
    CancelFlag.store(1);
    if (Worker.joinable())
        Worker.join();
}

// Apply finished worker results on the UI thread, and mirror live progress.
void TripSitterApp::PumpJobs()
{
    Progress = JobProgress.load();

    if (!bJobDone.load())
        return;

    // Worker has finished; join it and apply results under the lock.
    if (Worker.joinable())
        Worker.join();

    JobResult result;
    {
        std::lock_guard<std::mutex> lk(JobMutex);
        result = std::move(Pending);
        Pending = JobResult{};
    }

    if (ActiveJob == JobType::Analyze && result.ok)
    {
        AnalyzedBeatTimes = std::move(result.beats);
        DetectedBPM = result.bpm;
        if (result.duration > 0) AudioDuration = result.duration;
        if (!AnalyzedBeatTimes.empty())
        {
            OriginalFirstBeatTime = AnalyzedBeatTimes.front();
            bAudioAnalyzed = true;
        }
    }

    StatusText = result.status;
    ActiveJob = JobType::None;
    bJobDone.store(false);
    bIsProcessing.store(false);
    CancelFlag.store(0);
    JobProgress.store(0.0f);
}

// ===========================================================================
// Top-level layout
// ===========================================================================
void TripSitterApp::Draw()
{
    PumpJobs();

    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoSavedSettings;

    ImGui::Begin("TripSitter", nullptr, flags);

    DrawHeader();
    ImGui::Spacing();

    // Two-column working area.
    const float rightW = 360.0f;
    ImGui::BeginChild("##left", ImVec2(ImGui::GetContentRegionAvail().x - rightW - 12.0f,
                                       ImGui::GetContentRegionAvail().y - 36.0f), false);
    DrawFileSection();
    DrawWaveformSection();
    DrawAnalysisSection();
    ImGui::EndChild();

    ImGui::SameLine();

    ImGui::BeginChild("##right", ImVec2(rightW, ImGui::GetContentRegionAvail().y - 36.0f), false);
    DrawEffectsSection();
    DrawTransitionsSection();
    DrawStemsSection();
    DrawControlSection();
    ImGui::EndChild();

    DrawStatusBar();
    ImGui::End();
}

void TripSitterApp::DrawHeader()
{
    ImGui::PushStyleColor(ImGuiCol_Text, Theme::NeonCyan);
    ImGui::SetWindowFontScale(1.6f);
    ImGui::TextUnformatted("TRIP SITTER");
    ImGui::SetWindowFontScale(1.0f);
    ImGui::PopStyleColor();
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Text, Theme::WithAlpha(Theme::TextColor, 0.6f));
    ImGui::TextUnformatted("  audio-reactive beat-sync video editor");
    ImGui::PopStyleColor();
    ImGui::Separator();
}

// ===========================================================================
// File section
// ===========================================================================
static void PathRow(const char* label, std::string& path, const char* btnId,
                    bool& browseClicked)
{
    ImGui::TextUnformatted(label);
    ImGui::SameLine(120);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 90);
    char buf[1024];
    std::snprintf(buf, sizeof(buf), "%s", path.c_str());
    if (ImGui::InputText((std::string("##") + btnId).c_str(), buf, sizeof(buf)))
        path = buf;
    ImGui::SameLine();
    browseClicked = ImGui::Button((std::string("Browse##") + btnId).c_str());
}

void TripSitterApp::DrawFileSection()
{
    ImGui::SeparatorText("Files");

    bool browse = false;
    PathRow("Audio", AudioPath, "audio", browse);
    if (browse)
    {
        std::string p = FileDialog::OpenFile("Select audio file", "Audio",
                                             {"wav","mp3","flac","aiff","m4a","ogg"});
        if (!p.empty()) { AudioPath = p; LoadWaveform(p); }
    }

    ImGui::Checkbox("Multiple video clips", &bIsMultiClip);

    if (!bIsMultiClip)
    {
        PathRow("Video", VideoPath, "video", browse);
        if (browse)
        {
            std::string p = FileDialog::OpenFile("Select video file", "Video",
                                                 {"mp4","mov","mkv","avi","webm"});
            if (!p.empty()) VideoPath = p;
        }
    }
    else
    {
        ImGui::TextUnformatted("Clips");
        ImGui::SameLine(120);
        if (ImGui::Button("Add clips##multi"))
        {
            auto sel = FileDialog::OpenFiles("Select video clips", "Video",
                                             {"mp4","mov","mkv","avi","webm"});
            for (auto& s : sel) VideoPaths.push_back(s);
        }
        ImGui::SameLine();
        if (ImGui::Button("Add folder##multi"))
        {
            std::string folder = FileDialog::PickFolder("Select folder of clips");
            if (!folder.empty()) VideoPaths.push_back(folder + "*"); // TODO: scan folder
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear##multi")) VideoPaths.clear();

        ImGui::Indent(120);
        for (size_t i = 0; i < VideoPaths.size(); ++i)
            ImGui::BulletText("%s", VideoPaths[i].c_str());
        ImGui::Unindent(120);
    }

    PathRow("Output", OutputPath, "output", browse);
    if (browse)
    {
        std::string p = FileDialog::SaveFile("Save output video as", "tripsitter_output.mp4", {"mp4"});
        if (!p.empty()) OutputPath = p;
    }
}

// ===========================================================================
// Waveform section (custom DrawList rendering - ports SWaveformViewer)
// ===========================================================================
void TripSitterApp::DrawWaveformSection()
{
    ImGui::SeparatorText("Waveform");

    const ImVec2 size(ImGui::GetContentRegionAvail().x, 160.0f);
    ImVec2 p0 = ImGui::GetCursorScreenPos();
    ImVec2 p1 = ImVec2(p0.x + size.x, p0.y + size.y);
    ImDrawList* dl = ImGui::GetWindowDrawList();

    // Background.
    dl->AddRectFilled(p0, p1, ImGui::GetColorU32(Theme::WithAlpha(Theme::ControlBg, 0.9f)), 6.0f);
    dl->AddRect(p0, p1, ImGui::GetColorU32(Theme::WithAlpha(Theme::NeonCyan, 0.35f)), 6.0f);

    const float midY = p0.y + size.y * 0.5f;
    const ImU32 axis = ImGui::GetColorU32(Theme::WithAlpha(Theme::TextColor, 0.25f));
    dl->AddLine(ImVec2(p0.x, midY), ImVec2(p1.x, midY), axis);

    const bool haveBands = !BandBass.empty() && BandBass.size() == BandMid.size() &&
                           BandMid.size() == BandHigh.size();
    const std::vector<float>& mono = WaveformPeaks;

    auto sampleAt = [&](const std::vector<float>& v, float t01) -> float {
        if (v.empty()) return 0.0f;
        float fi = t01 * (v.size() - 1);
        int i = (int)fi;
        i = std::clamp(i, 0, (int)v.size() - 1);
        return std::fabs(v[i]);
    };

    const int cols = (int)size.x;
    for (int x = 0; x < cols; ++x)
    {
        float t01 = (cols > 1) ? (float)x / (cols - 1) : 0.0f;
        float colX = p0.x + x;
        if (haveBands)
        {
            float hi = sampleAt(BandHigh, t01) * (size.y * 0.5f);
            float md = sampleAt(BandMid,  t01) * (size.y * 0.5f);
            float bs = sampleAt(BandBass, t01) * (size.y * 0.5f);
            dl->AddLine(ImVec2(colX, midY - hi), ImVec2(colX, midY + hi),
                        ImGui::GetColorU32(Theme::WithAlpha(ImVec4(1,1,0.6f,1), 0.9f)));
            dl->AddLine(ImVec2(colX, midY - md), ImVec2(colX, midY + md),
                        ImGui::GetColorU32(Theme::WithAlpha(Theme::NeonCyan, 0.9f)));
            dl->AddLine(ImVec2(colX, midY - bs), ImVec2(colX, midY + bs),
                        ImGui::GetColorU32(Theme::WithAlpha(Theme::HotPink, 0.9f)));
        }
        else
        {
            float a = sampleAt(mono, t01) * (size.y * 0.5f);
            dl->AddLine(ImVec2(colX, midY - a), ImVec2(colX, midY + a),
                        ImGui::GetColorU32(Theme::WithAlpha(Theme::NeonCyan, 0.85f)));
        }
    }

    // Beat markers.
    if (AudioDuration > 0.0)
    {
        const int divisor = BeatDivisor(BeatRateSel);
        for (size_t i = 0; i < AnalyzedBeatTimes.size(); ++i)
        {
            if ((int)(i % divisor) != 0) continue;
            float t01 = (float)(AnalyzedBeatTimes[i] / AudioDuration);
            if (t01 < 0 || t01 > 1) continue;
            float bx = p0.x + t01 * size.x;
            dl->AddLine(ImVec2(bx, p0.y), ImVec2(bx, p1.y),
                        ImGui::GetColorU32(Theme::WithAlpha(Theme::NeonGreen, 0.7f)), 1.5f);
        }

        // Selection overlay.
        double selEnd = (SelectionEnd < 0) ? AudioDuration : SelectionEnd;
        float sx0 = p0.x + (float)(SelectionStart / AudioDuration) * size.x;
        float sx1 = p0.x + (float)(selEnd / AudioDuration) * size.x;
        dl->AddRectFilled(ImVec2(sx0, p0.y), ImVec2(sx1, p1.y),
                          ImGui::GetColorU32(Theme::WithAlpha(Theme::NeonPurple, 0.18f)));
    }

    // Interaction: click+drag to set selection range.
    ImGui::InvisibleButton("##waveform", size);
    if (ImGui::IsItemActive() && AudioDuration > 0.0)
    {
        float mx = ImGui::GetIO().MousePos.x;
        double t = std::clamp((double)((mx - p0.x) / size.x), 0.0, 1.0) * AudioDuration;
        if (ImGui::IsItemActivated()) { SelectionStart = t; SelectionEnd = t; }
        else { SelectionEnd = std::max(t, SelectionStart); }
    }

    if (WaveformPeaks.empty() && !haveBands)
    {
        const char* msg = "Load an audio file to see its waveform";
        ImVec2 ts = ImGui::CalcTextSize(msg);
        dl->AddText(ImVec2(p0.x + (size.x - ts.x) * 0.5f, midY - ts.y * 0.5f),
                    ImGui::GetColorU32(Theme::WithAlpha(Theme::TextColor, 0.5f)), msg);
    }

    if (AudioDuration > 0.0)
    {
        double selEnd = (SelectionEnd < 0) ? AudioDuration : SelectionEnd;
        ImGui::Text("Duration: %.1fs   Selection: %.2fs - %.2fs   Beats: %zu",
                    AudioDuration, SelectionStart, selEnd, AnalyzedBeatTimes.size());
    }
}

// ===========================================================================
// Analysis section
// ===========================================================================
void TripSitterApp::DrawAnalysisSection()
{
    ImGui::SeparatorText("Beat Analysis");

    ImGui::SetNextItemWidth(220);
    EnumCombo("Mode", AnalysisModeSel, kAnalysisLabels, IM_ARRAYSIZE(kAnalysisLabels));
    if (const char* reason = AnalysisModeAvailability(AnalysisModeSel))
    {
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, Theme::HotPink);
        ImGui::Text("(%s)", reason);
        ImGui::PopStyleColor();
    }

    ImGui::BeginDisabled(AudioPath.empty() || JobBusy() || !Backend.IsLoaded());
    if (ImGui::Button("Analyze Audio", ImVec2(140, 0)))
        StartAnalyzeJob();
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::BeginDisabled(!bAudioAnalyzed);

    double bpm = DetectedBPM;
    ImGui::SetNextItemWidth(110);
    if (ImGui::InputDouble("BPM", &bpm, 0.0, 0.0, "%.2f"))
        RecalculateBeatsFromBPM(bpm);
    ImGui::SameLine();
    if (ImGui::Button("/2")) RecalculateBeatsFromBPM(DetectedBPM * 0.5);
    ImGui::SameLine();
    if (ImGui::Button("x2")) RecalculateBeatsFromBPM(DetectedBPM * 2.0);
    ImGui::EndDisabled();

    ImGui::SetNextItemWidth(160);
    EnumCombo("Beat rate", BeatRateSel, kBeatRateLabels, IM_ARRAYSIZE(kBeatRateLabels));
}

// ===========================================================================
// Effects section
// ===========================================================================
void TripSitterApp::DrawEffectsSection()
{
    ImGui::SeparatorText("Effects");

    ImGui::Checkbox("Beat Flash", &bEnableBeatFlash);
    if (bEnableBeatFlash) { ImGui::SameLine(150); ImGui::SetNextItemWidth(150);
        ImGui::SliderFloat("##flash", &FlashIntensity, 0.0f, 1.0f, "%.2f"); }

    ImGui::Checkbox("Beat Zoom", &bEnableBeatZoom);
    if (bEnableBeatZoom) { ImGui::SameLine(150); ImGui::SetNextItemWidth(150);
        ImGui::SliderFloat("##zoom", &ZoomIntensity, 0.0f, 0.5f, "%.2f"); }

    ImGui::Checkbox("Vignette", &bEnableVignette);
    if (bEnableVignette) { ImGui::SameLine(150); ImGui::SetNextItemWidth(150);
        ImGui::SliderFloat("##vig", &VignetteStrength, 0.0f, 1.0f, "%.2f"); }

    ImGui::Checkbox("Color Grade", &bEnableColorGrade);
    if (bEnableColorGrade) { ImGui::SameLine(150); ImGui::SetNextItemWidth(150);
        EnumCombo("##color", ColorPresetSel, kColorLabels, IM_ARRAYSIZE(kColorLabels)); }
}

void TripSitterApp::DrawTransitionsSection()
{
    ImGui::SeparatorText("Transitions");
    ImGui::Checkbox("Enable transitions", &bEnableTransitions);
    ImGui::BeginDisabled(!bEnableTransitions);
    ImGui::SetNextItemWidth(150);
    EnumCombo("Type", TransitionTypeSel, kTransitionLabels, IM_ARRAYSIZE(kTransitionLabels));
    ImGui::SetNextItemWidth(150);
    ImGui::SliderFloat("Duration", &TransitionDuration, 0.1f, 2.0f, "%.2fs");
    ImGui::EndDisabled();
}

// ===========================================================================
// Stems section
// ===========================================================================
void TripSitterApp::DrawStemsSection()
{
    ImGui::SeparatorText("Stem -> Effect Mapping");
    for (int i = 0; i < STEM_COUNT; ++i)
    {
        StemConfig& sc = StemConfigs[i];
        ImGui::PushID(i);
        ImGui::Checkbox(kStemNames[i], &sc.enabled);
        ImGui::SameLine(90);
        ImGui::SetNextItemWidth(120);
        EnumCombo("##effect", sc.effect, kStemEffectLabels, IM_ARRAYSIZE(kStemEffectLabels));
        ImGui::SameLine();
        if (ImGui::SmallButton("file"))
        {
            std::string p = FileDialog::OpenFile(std::string("Select ") + kStemNames[i] + " stem",
                                                 "Audio", {"wav","mp3","flac"});
            if (!p.empty()) sc.filePath = p;
        }
        if (!sc.filePath.empty())
        {
            ImGui::SameLine();
            const char* base = std::strrchr(sc.filePath.c_str(), '/');
            ImGui::TextDisabled("%s", base ? base + 1 : sc.filePath.c_str());
        }
        ImGui::PopID();
    }
}

// ===========================================================================
// Control section (output settings + run)
// ===========================================================================
void TripSitterApp::DrawControlSection()
{
    ImGui::SeparatorText("Output & Run");

    ImGui::SetNextItemWidth(180);
    EnumCombo("Resolution", ResolutionSel, kResolutionLabels, IM_ARRAYSIZE(kResolutionLabels));
    ImGui::SetNextItemWidth(180);
    EnumCombo("FPS", FpsSel, kFpsLabels, IM_ARRAYSIZE(kFpsLabels));

    ImGui::Spacing();

    const bool canRun = bAudioAnalyzed && !OutputPath.empty() && !JobBusy() &&
                        Backend.IsLoaded() &&
                        (bIsMultiClip ? !VideoPaths.empty() : !VideoPath.empty());

    ImGui::BeginDisabled(!canRun);
    ImGui::PushStyleColor(ImGuiCol_Button, Theme::WithAlpha(Theme::NeonGreen, 0.55f));
    if (ImGui::Button("START SYNC", ImVec2(-FLT_MIN, 38)))
        StartSyncJob();
    ImGui::PopStyleColor();
    ImGui::EndDisabled();

    if (JobBusy())
    {
        if (ImGui::Button("Cancel", ImVec2(-FLT_MIN, 0)))
            Cancel();
        ImGui::ProgressBar(JobProgress.load(), ImVec2(-FLT_MIN, 0));
    }
}

void TripSitterApp::DrawStatusBar()
{
    ImGui::Separator();
    ImU32 dot = Backend.IsLoaded()
        ? ImGui::GetColorU32(Theme::NeonGreen) : ImGui::GetColorU32(Theme::HotPink);
    ImVec2 c = ImGui::GetCursorScreenPos();
    ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(c.x + 6, c.y + 9), 5.0f, dot);
    ImGui::Dummy(ImVec2(16, 0)); ImGui::SameLine();

    std::string status = StatusText;
    if (JobBusy())
    {
        std::lock_guard<std::mutex> lk(JobMutex);
        if (!WorkerStatus.empty()) status = WorkerStatus;
    }

    if (Backend.IsLoaded())
        ImGui::Text("Backend: %s   |   %s", Backend.bs_get_version(), status.c_str());
    else
        ImGui::Text("Backend NOT loaded: %s", Backend.LastError().c_str());

    if (!ETAText.empty()) { ImGui::SameLine(); ImGui::TextDisabled("(%s)", ETAText.c_str()); }
}

// ===========================================================================
// Actions
// ===========================================================================
const char* TripSitterApp::AnalysisModeAvailability(AnalysisMode m) const
{
    switch (m)
    {
        case AnalysisMode::AIBeat:
        case AnalysisMode::AIStems:
            if (!Backend.bs_ai_is_available || !Backend.bs_ai_is_available())
                return "AI unavailable - falls back to Energy";
            return nullptr;
        case AnalysisMode::AudioFlux:
        case AnalysisMode::StemsFlux:
            if (!Backend.bs_audioflux_is_available || !Backend.bs_audioflux_is_available())
                return "AudioFlux unavailable - falls back to Energy";
            return nullptr;
        default:
            return nullptr;
    }
}

void TripSitterApp::LoadWaveform(const std::string& path)
{
    WaveformPeaks.clear();
    BandBass.clear(); BandMid.clear(); BandHigh.clear();
    AudioDuration = 0.0;
    SelectionStart = 0.0; SelectionEnd = -1.0;
    if (!Backend.IsLoaded()) { StatusText = "Backend not loaded"; return; }

    void* analyzer = Backend.bs_create_audio_analyzer();
    if (!analyzer) { StatusText = "Failed to create analyzer"; return; }

    // Prefer frequency-band waveform when available.
    bool gotBands = false;
    if (Backend.bs_get_waveform_bands && Backend.bs_free_waveform_bands)
    {
        bs_waveform_bands_t bands{};
        if (Backend.bs_get_waveform_bands(analyzer, path.c_str(), &bands) == 0 && bands.count > 0)
        {
            BandBass.assign(bands.bass_peaks, bands.bass_peaks + bands.count);
            BandMid.assign(bands.mid_peaks, bands.mid_peaks + bands.count);
            BandHigh.assign(bands.high_peaks, bands.high_peaks + bands.count);
            AudioDuration = bands.duration;
            Backend.bs_free_waveform_bands(&bands);
            gotBands = true;
        }
    }
    if (!gotBands)
    {
        float* peaks = nullptr; size_t count = 0; double dur = 0;
        if (Backend.bs_get_waveform(analyzer, path.c_str(), &peaks, &count, &dur) == 0 && peaks)
        {
            WaveformPeaks.assign(peaks, peaks + count);
            AudioDuration = dur;
            Backend.bs_free_waveform(peaks);
        }
        else StatusText = "Could not read waveform";
    }
    Backend.bs_destroy_audio_analyzer(analyzer);
    if (AudioDuration > 0) StatusText = "Waveform loaded";
}

void TripSitterApp::StartAnalyzeJob()
{
    if (!Backend.IsLoaded() || AudioPath.empty() || JobBusy()) return;
    if (Worker.joinable()) Worker.join();

    // Snapshot inputs for the worker.
    Job = JobInput{};
    Job.audioPath = AudioPath;
    Job.mode = AnalysisModeSel;

    bAudioAnalyzed = false;
    AnalyzedBeatTimes.clear();
    DetectedBPM = 0.0;
    StatusText = "Analyzing...";
    { std::lock_guard<std::mutex> lk(JobMutex); WorkerStatus = "Analyzing audio..."; }

    ActiveJob = JobType::Analyze;
    bJobDone.store(false);
    JobProgress.store(0.0f);
    CancelFlag.store(0);
    bIsProcessing.store(true);
    Worker = std::thread(&TripSitterApp::RunAnalyzeJob, this);
}

void TripSitterApp::RunAnalyzeJob()
{
    JobResult out;

    AnalysisMode mode = Job.mode;
    // Downgrade to Energy if the requested engine isn't present.
    if (AnalysisModeAvailability(mode) != nullptr)
        mode = AnalysisMode::Energy;

    const std::string& AudioPath = Job.audioPath; // shadow: worker uses snapshot
    bool ok = false;
    std::vector<double> beats;
    double bpm = 0.0, duration = 0.0;

    if (mode == AnalysisMode::AudioFlux || mode == AnalysisMode::StemsFlux)
    {
        std::string stemModel = ResolveModel("demucs.onnx");
        bs_ai_result_t res{};
        int rc = (mode == AnalysisMode::StemsFlux && Backend.bs_audioflux_analyze_with_stems)
            ? Backend.bs_audioflux_analyze_with_stems(AudioPath.c_str(), stemModel.c_str(), &res, AiProgressThunk, this)
            : Backend.bs_audioflux_analyze(AudioPath.c_str(), &res, AiProgressThunk, this);
        if (rc == 0 && res.beats)
        {
            beats.assign(res.beats, res.beats + res.beat_count);
            bpm = res.bpm;
            if (res.duration > 0) duration = res.duration;
            ok = true;
        }
        if (Backend.bs_free_ai_result) Backend.bs_free_ai_result(&res);
    }
    else if (mode == AnalysisMode::AIBeat || mode == AnalysisMode::AIStems)
    {
        std::string beatModel = ResolveModel("beatnet.onnx");
        std::string stemModel = ResolveModel("demucs.onnx");
        bs_ai_config_t cfg{};
        cfg.beat_model_path = beatModel.c_str();
        cfg.stem_model_path = (mode == AnalysisMode::AIStems) ? stemModel.c_str() : nullptr;
        cfg.use_stem_separation = (mode == AnalysisMode::AIStems) ? 1 : 0;
        cfg.use_drums_for_beats = 1;
        cfg.use_gpu = 1;
        cfg.beat_threshold = 0.5f;
        cfg.downbeat_threshold = 0.5f;
        cfg.kick_only_mode = 1;
        cfg.kick_freq_cutoff = 200.0f;

        void* ai = Backend.bs_create_ai_analyzer(&cfg);
        if (ai)
        {
            bs_ai_result_t res{};
            int rc = Backend.bs_ai_analyze_file(ai, AudioPath.c_str(), &res, AiProgressThunk, this);
            if (rc == 0 && res.beats)
            {
                beats.assign(res.beats, res.beats + res.beat_count);
                bpm = res.bpm;
                if (res.duration > 0) duration = res.duration;
                ok = true;
            }
            if (Backend.bs_free_ai_result) Backend.bs_free_ai_result(&res);
            Backend.bs_destroy_ai_analyzer(ai);
        }
    }

    if (!ok && CancelFlag.load() == 0) // Energy fallback (always available)
    {
        void* analyzer = Backend.bs_create_audio_analyzer();
        if (analyzer)
        {
            bs_beatgrid_t grid{};
            if (Backend.bs_analyze_audio(analyzer, AudioPath.c_str(), &grid) == 0 && grid.beats)
            {
                beats.assign(grid.beats, grid.beats + grid.count);
                bpm = grid.bpm;
                if (grid.duration > 0) duration = grid.duration;
                ok = true;
            }
            Backend.bs_free_beatgrid(&grid);
            Backend.bs_destroy_audio_analyzer(analyzer);
        }
    }

    out.ok = ok && !beats.empty();
    out.beats = std::move(beats);
    out.bpm = bpm;
    out.duration = duration;
    if (CancelFlag.load() != 0)
        out.status = "Analysis cancelled";
    else if (out.ok)
    {
        char b[128];
        std::snprintf(b, sizeof(b), "Detected %zu beats at %.1f BPM",
                      out.beats.size(), out.bpm);
        out.status = b;
    }
    else
        out.status = "Analysis failed";

    { std::lock_guard<std::mutex> lk(JobMutex); Pending = std::move(out); }
    bJobDone.store(true);
}

void TripSitterApp::RecalculateBeatsFromBPM(double newBPM)
{
    if (newBPM <= 0.0 || AudioDuration <= 0.0) return;
    DetectedBPM = newBPM;
    const double interval = 60.0 / newBPM;
    AnalyzedBeatTimes.clear();
    for (double t = OriginalFirstBeatTime; t <= AudioDuration; t += interval)
        AnalyzedBeatTimes.push_back(t);
}

std::vector<double> TripSitterApp::EffectiveBeatTimes() const
{
    std::vector<double> out;
    const int divisor = BeatDivisor(BeatRateSel);
    const double selEnd = (SelectionEnd < 0) ? AudioDuration : SelectionEnd;
    for (size_t i = 0; i < AnalyzedBeatTimes.size(); ++i)
    {
        if ((int)(i % divisor) != 0) continue;
        double t = AnalyzedBeatTimes[i];
        if (t + 1e-6 < SelectionStart || t > selEnd + 1e-6) continue;
        out.push_back(t);
    }
    return out;
}

void TripSitterApp::StartSyncJob()
{
    if (!Backend.IsLoaded() || JobBusy()) return;
    std::vector<double> beats = EffectiveBeatTimes();
    if (beats.empty()) { StatusText = "No beats in selection"; return; }
    if (Worker.joinable()) Worker.join();

    // Snapshot inputs for the worker.
    Job = JobInput{};
    Job.audioPath = AudioPath;
    Job.videoPath = VideoPath;
    Job.videoPaths = VideoPaths;
    Job.outputPath = OutputPath;
    Job.multiClip = bIsMultiClip;
    Job.beats = std::move(beats);
    Job.selStart = SelectionStart;
    Job.selEnd = (SelectionEnd < 0) ? AudioDuration : SelectionEnd;

    StatusText = "Cutting video to beats...";
    { std::lock_guard<std::mutex> lk(JobMutex); WorkerStatus = "Cutting video to beats..."; }

    ActiveJob = JobType::Sync;
    bJobDone.store(false);
    JobProgress.store(0.0f);
    CancelFlag.store(0);
    bIsProcessing.store(true);
    Worker = std::thread(&TripSitterApp::RunSyncJob, this);
}

void TripSitterApp::RunSyncJob()
{
    JobResult out;

    void* writer = Backend.bs_create_video_writer();
    if (!writer)
    {
        out.status = "Failed to create video writer";
        { std::lock_guard<std::mutex> lk(JobMutex); Pending = std::move(out); }
        bJobDone.store(true);
        return;
    }

    // Wire progress + cancellation into the backend.
    if (Backend.bs_video_set_progress_callback)
        Backend.bs_video_set_progress_callback(writer, VideoProgressThunk, this);
    if (Backend.bs_video_set_cancel_flag)
        // CancelFlag is atomic<int> but layout-compatible with int; the backend
        // only reads *ptr != 0, which is safe for a one-way cancel signal.
        Backend.bs_video_set_cancel_flag(writer, reinterpret_cast<const int*>(&CancelFlag));

    const std::vector<double>& beats = Job.beats;
    const double clipDuration = (beats.size() > 1) ? (beats[1] - beats[0]) : 0.5;

    int rc;
    if (Job.multiClip && Backend.bs_video_cut_at_beats_multi && !Job.videoPaths.empty())
    {
        std::vector<const char*> ins;
        for (auto& s : Job.videoPaths) ins.push_back(s.c_str());
        rc = Backend.bs_video_cut_at_beats_multi(writer, ins.data(), ins.size(),
                                                 beats.data(), beats.size(),
                                                 Job.outputPath.c_str(), clipDuration);
    }
    else
    {
        rc = Backend.bs_video_cut_at_beats(writer, Job.videoPath.c_str(),
                                           beats.data(), beats.size(),
                                           Job.outputPath.c_str(), clipDuration);
    }

    const bool cancelled = CancelFlag.load() != 0 ||
                           (Backend.bs_video_is_cancelled && Backend.bs_video_is_cancelled(writer));

    if (cancelled)
    {
        out.status = "Cancelled";
    }
    else if (rc == 0)
    {
        { std::lock_guard<std::mutex> lk(JobMutex); WorkerStatus = "Muxing audio..."; }
        Backend.bs_video_add_audio_track(writer, Job.outputPath.c_str(), Job.audioPath.c_str(),
                                         Job.outputPath.c_str(), 1, Job.selStart, Job.selEnd);
        out.ok = true;
        out.status = "Done: " + Job.outputPath;
        JobProgress.store(1.0f);
    }
    else
    {
        const char* err = Backend.bs_video_get_last_error(writer);
        out.status = std::string("Video processing failed: ") + (err ? err : "unknown");
    }

    Backend.bs_destroy_video_writer(writer);
    { std::lock_guard<std::mutex> lk(JobMutex); Pending = std::move(out); }
    bJobDone.store(true);
}

void TripSitterApp::Cancel()
{
    CancelFlag.store(1);
    { std::lock_guard<std::mutex> lk(JobMutex); WorkerStatus = "Cancelling..."; }
}

// --- Backend progress callbacks (worker thread) ----------------------------
void TripSitterApp::VideoProgressThunk(double progress, void* user_data)
{
    auto* self = static_cast<TripSitterApp*>(user_data);
    self->JobProgress.store(static_cast<float>(progress));
}

int TripSitterApp::AiProgressThunk(float progress, const char* stage, const char* message, void* user_data)
{
    auto* self = static_cast<TripSitterApp*>(user_data);
    self->JobProgress.store(progress);
    {
        std::lock_guard<std::mutex> lk(self->JobMutex);
        self->WorkerStatus = message ? message : (stage ? stage : "Analyzing...");
    }
    return self->CancelFlag.load(); // non-zero aborts the backend analysis
}
