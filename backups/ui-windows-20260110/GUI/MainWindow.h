#pragma once
#include <wx/wx.h>
#include <wx/filepicker.h>
#include <wx/spinctrl.h>
#include <wx/choice.h>
#include <wx/gauge.h>
#include <wx/animate.h>
#include <wx/scrolwin.h>
#include <memory>
#include <thread>

class BeatVisualizer;
class VideoPreview;
class SettingsManager;
struct ProcessingConfig;

class MainWindow : public wxFrame {
public:
    MainWindow();
    ~MainWindow();

    enum {
        ID_VIEW_LOGS = wxID_HIGHEST + 1
    };

    // Inspection helper for tests and automation: returns whether a background image was loaded
    bool HasWallpaper() const { return m_backgroundBitmap.IsOk(); }

private:
    // UI Components
    wxScrolledWindow* m_mainPanel = nullptr;
    wxStaticBitmap* m_titleImage = nullptr;
    wxFilePickerCtrl* m_audioFilePicker = nullptr;
    wxDirPickerCtrl* m_videoFolderPicker = nullptr;
    wxFilePickerCtrl* m_singleVideoPicker = nullptr;
    wxRadioButton* m_singleVideoRadio = nullptr;
    wxRadioButton* m_multiClipRadio = nullptr;
    wxFilePickerCtrl* m_outputFilePicker = nullptr;
    
    wxChoice* m_beatRateChoice = nullptr;
    wxChoice* m_analysisModeChoice = nullptr;  // Energy / BeatNet / Demucs+BeatNet
    wxChoice* m_resolutionChoice = nullptr;
    wxChoice* m_fpsChoice = nullptr;
    wxSpinCtrl* m_previewBeatsCtrl = nullptr;
    wxCheckBox* m_previewModeCheck = nullptr;

    // Effects controls
    wxCheckBox* m_colorGradeCheck = nullptr;
    wxChoice* m_colorPresetChoice = nullptr;
    wxCheckBox* m_vignetteCheck = nullptr;
    wxCheckBox* m_beatFlashCheck = nullptr;
    wxSlider* m_flashIntensitySlider = nullptr;
    wxCheckBox* m_beatZoomCheck = nullptr;
    wxSlider* m_zoomIntensitySlider = nullptr;
    wxChoice* m_effectBeatDivisorChoice = nullptr;

    // Transitions UI
    wxCheckBox* m_enableTransitionsCheck = nullptr;
    wxChoice* m_transitionChoice = nullptr;
    wxSpinCtrlDouble* m_transitionDurationCtrl = nullptr;
    wxButton* m_transitionPreviewButton = nullptr;

    wxGauge* m_progressBar = nullptr;
    wxStaticText* m_statusText = nullptr;
    wxStaticText* m_etaText = nullptr;
    wxButton* m_startButton = nullptr;
    wxButton* m_cancelButton = nullptr;
    wxButton* m_previewButton = nullptr; // Trigger a preview frame
    wxTextCtrl* m_previewTimestampCtrl = nullptr; // Preview timestamp in seconds
    
    BeatVisualizer* m_beatVisualizer;
    VideoPreview* m_videoPreview;
    
    // Animation for start button
    wxAnimationCtrl* m_startAnimation;
    
    // Backend
    std::unique_ptr<SettingsManager> m_settingsManager;
    std::unique_ptr<std::thread> m_processingThread;
    std::atomic<bool> m_cancelRequested{false};
    
    // Visuals
    wxPanel* m_backgroundPanel;
    wxBitmap m_backgroundBitmap;
    wxBitmap m_headerBitmap;
    wxStaticBitmap* m_headerCtrl = nullptr;
    wxFont m_titleFont;
    wxFont m_labelFont;
    
    // Event handlers
    void OnAudioSelected(wxFileDirPickerEvent& event);
    void OnVideoSourceChanged(wxCommandEvent& event);
    void OnStartProcessing(wxCommandEvent& event);
    void OnCancelProcessing(wxCommandEvent& event);
    void OnPreviewFrame(wxCommandEvent& event);
    void OnPreviewTransition(wxCommandEvent& event);
    void OnClose(wxCloseEvent& event);
    void OnPaint(wxPaintEvent& event);
    void OnFrameSize(wxSizeEvent& event);
    
    // Helper methods
    void CreateControls();
    void CreateLayout();
    void LoadSettings();
    void SaveSettings();
    void UpdateUIState(bool processing);
    void LoadBackgroundImage();
    void SetupFonts();
    void ApplyPsychedelicStyling();
    
    // Processing
    void StartProcessing(const ProcessingConfig& config);
    void UpdateProgress(int percent, const wxString& status, const wxString& eta);
    void OnProcessingComplete(bool success, const wxString& message);

    // Logs & diagnostics
    void OnViewLogs(wxCommandEvent& event);

#ifdef __WXUNIVERSAL__
    void SetAllChildrenTransparent(wxWindow* parent);
#endif

    wxDECLARE_EVENT_TABLE();
};

// Custom events for thread communication
wxDECLARE_EVENT(wxEVT_PROCESSING_PROGRESS, wxThreadEvent);
wxDECLARE_EVENT(wxEVT_PROCESSING_COMPLETE, wxThreadEvent);
wxDECLARE_EVENT(wxEVT_PROCESSING_ERROR, wxThreadEvent);

struct ProcessingConfig {
    wxString audioPath;
    wxString videoPath;
    bool isMultiClip;
    wxString outputPath;
    int beatRate;
    int analysisMode;  // 0=Energy, 1=BeatNet, 2=Demucs+BeatNet
    wxString resolution;
    int fps;
    bool previewMode;
    int previewBeats;
    double selectionStart = 0.0;
    double selectionEnd = -1.0;  // -1 means full track

    // Effects
    bool enableColorGrade = false;
    wxString colorPreset = "none";
    bool enableVignette = false;
    bool enableBeatFlash = false;
    double flashIntensity = 0.3;     // 0.1 to 1.0
    bool enableBeatZoom = false;
    double zoomIntensity = 0.04;     // 0.01 to 0.15
    int effectBeatDivisor = 1;       // 1=every, 2=every 2nd, 4=every 4th, 8=every 8th

    // Transitions
    bool enableTransitions = false;
    wxString transitionType = "fade";
    double transitionDuration = 0.3;
    
    // Effect region (from waveform right-click)
    double effectStartTime = 0.0;    // -1 or 0 means from beginning  
    double effectEndTime = -1.0;     // -1 means to end
};
