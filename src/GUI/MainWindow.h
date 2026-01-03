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

private:
    // UI Components
    wxScrolledWindow* m_mainPanel;
    wxFilePickerCtrl* m_audioFilePicker;
    wxDirPickerCtrl* m_videoFolderPicker;
    wxFilePickerCtrl* m_singleVideoPicker;
    wxRadioButton* m_singleVideoRadio;
    wxRadioButton* m_multiClipRadio;
    wxFilePickerCtrl* m_outputFilePicker;
    
    wxChoice* m_beatRateChoice;
    wxChoice* m_resolutionChoice;
    wxChoice* m_fpsChoice;
    wxSpinCtrl* m_previewBeatsCtrl;
    wxCheckBox* m_previewModeCheck;

    // Effects controls
    wxCheckBox* m_colorGradeCheck;
    wxChoice* m_colorPresetChoice;
    wxCheckBox* m_vignetteCheck;
    wxCheckBox* m_beatFlashCheck;
    wxCheckBox* m_beatZoomCheck;

    wxGauge* m_progressBar;
    wxStaticText* m_statusText;
    wxStaticText* m_etaText;
    wxButton* m_startButton;
    wxButton* m_cancelButton;
    wxButton* m_previewButton; // Trigger a preview frame
    wxTextCtrl* m_previewTimestampCtrl; // Preview timestamp in seconds
    
    BeatVisualizer* m_beatVisualizer;
    VideoPreview* m_videoPreview;
    
    // Backend
    std::unique_ptr<SettingsManager> m_settingsManager;
    std::unique_ptr<std::thread> m_processingThread;
    std::atomic<bool> m_cancelRequested{false};
    
    // Visuals
    wxPanel* m_backgroundPanel;
    wxBitmap m_backgroundBitmap;
    wxFont m_titleFont;
    wxFont m_labelFont;
    
    // Event handlers
    void OnAudioSelected(wxFileDirPickerEvent& event);
    void OnVideoSourceChanged(wxCommandEvent& event);
    void OnStartProcessing(wxCommandEvent& event);
    void OnCancelProcessing(wxCommandEvent& event);
    void OnPreviewFrame(wxCommandEvent& event);
    void OnClose(wxCloseEvent& event);
    void OnPaint(wxPaintEvent& event);
    
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
    bool enableBeatZoom = false;
    double bpm = 120.0;
};
