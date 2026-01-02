#include "MainWindow.h"
#include "BeatVisualizer.h"
#include "VideoPreview.h"
#include "SettingsManager.h"
#include "../audio/AudioAnalyzer.h"
#include "../video/VideoWriter.h"
#include "../video/VideoProcessor.h"
#include <wx/statbox.h>
#include <wx/gbsizer.h>
#include <wx/stdpaths.h>
#include <wx/filename.h>
#include <wx/dir.h>
#include <filesystem>
#include <chrono>
#include <algorithm>

// Custom event definitions
wxDEFINE_EVENT(wxEVT_PROCESSING_PROGRESS, wxThreadEvent);
wxDEFINE_EVENT(wxEVT_PROCESSING_COMPLETE, wxThreadEvent);
wxDEFINE_EVENT(wxEVT_PROCESSING_ERROR, wxThreadEvent);

wxBEGIN_EVENT_TABLE(MainWindow, wxFrame)
    EVT_PAINT(MainWindow::OnPaint)
    EVT_CLOSE(MainWindow::OnClose)
wxEND_EVENT_TABLE()

MainWindow::MainWindow()
    : wxFrame(nullptr, wxID_ANY, "Trip Sitter - Audio Beat Sync GUI",
              wxDefaultPosition, wxSize(1344, 768),
              wxDEFAULT_FRAME_STYLE & ~(wxRESIZE_BORDER | wxMAXIMIZE_BOX))
{
    m_settingsManager = std::make_unique<SettingsManager>();
    
    SetupFonts();
    LoadBackgroundImage();
    CreateControls();
    CreateLayout();
    ApplyPsychedelicStyling();
    LoadSettings();
    
    Centre();
    
    // Bind custom events
    Bind(wxEVT_PROCESSING_PROGRESS, [this](wxThreadEvent& evt) {
        UpdateProgress(evt.GetInt(), evt.GetString(), evt.GetExtraLong() ? 
            wxString::Format("ETA: %ds", evt.GetExtraLong()) : "");
    });
    
    Bind(wxEVT_PROCESSING_COMPLETE, [this](wxThreadEvent& evt) {
        OnProcessingComplete(evt.GetInt() == 1, evt.GetString());
    });
}

MainWindow::~MainWindow() {
    SaveSettings();
    if (m_processingThread && m_processingThread->joinable()) {
        m_cancelRequested = true;
        m_processingThread->join();
    }
}

void MainWindow::SetupFonts() {
    // Cyberpunk-style fonts
    m_titleFont = wxFont(20, wxFONTFAMILY_MODERN, wxFONTSTYLE_NORMAL, 
                         wxFONTWEIGHT_BOLD, false, "Consolas");
    m_labelFont = wxFont(10, wxFONTFAMILY_MODERN, wxFONTSTYLE_NORMAL, 
                         wxFONTWEIGHT_NORMAL, false, "Consolas");
}

void MainWindow::LoadBackgroundImage() {
    wxString bgPath = wxStandardPaths::Get().GetExecutablePath();
    bgPath = wxFileName(bgPath).GetPath() + "/assets/background.png";

    if (wxFileExists(bgPath)) {
        wxImage img(bgPath, wxBITMAP_TYPE_PNG);
        if (img.IsOk()) {
            // Use native image size (1344x768) - matches window size
            m_backgroundBitmap = wxBitmap(img);
        }
    }

    // Fallback: Create gradient background
    if (!m_backgroundBitmap.IsOk()) {
        wxBitmap bmp(1344, 768);
        wxMemoryDC dc(bmp);
        dc.GradientFillLinear(wxRect(0, 0, 1344, 768),
            wxColour(10, 10, 26), wxColour(25, 0, 50), wxSOUTH);
        m_backgroundBitmap = bmp;
    }
}

void MainWindow::ApplyPsychedelicStyling() {
    // Set window background to black
    SetBackgroundColour(wxColour(10, 10, 26));

    // Apply neon colors to UI elements
    wxColour cyan(0, 217, 255);
    wxColour purple(139, 0, 255);
    wxColour darkBg(20, 20, 40);

    if (m_mainPanel) {
        // Make panel transparent to show background image
        m_mainPanel->SetBackgroundStyle(wxBG_STYLE_PAINT);

        // Make all child static boxes transparent
        wxWindowList& children = m_mainPanel->GetChildren();
        for (wxWindowList::iterator it = children.begin(); it != children.end(); ++it) {
            wxWindow* child = *it;
            if (child->GetClassInfo()->GetClassName() == wxString("wxStaticBox")) {
                child->SetBackgroundStyle(wxBG_STYLE_PAINT);
                child->Bind(wxEVT_ERASE_BACKGROUND, [](wxEraseEvent&) { /* Skip erase */ });
            }
        }
    }
    
    // Style text inputs and dropdowns with dark backgrounds
    wxColour darkInput(20, 20, 40);
    wxColour lightText(200, 220, 255);

    // Style file pickers and their children
    if (m_audioFilePicker) {
        m_audioFilePicker->SetBackgroundColour(darkInput);
        m_audioFilePicker->SetForegroundColour(lightText);
        // Style the text control inside the file picker
        wxWindowList& children = m_audioFilePicker->GetChildren();
        for (auto child : children) {
            child->SetBackgroundColour(darkInput);
            child->SetForegroundColour(lightText);
        }
        m_audioFilePicker->Refresh();
    }
    if (m_singleVideoPicker) {
        m_singleVideoPicker->SetBackgroundColour(darkInput);
        m_singleVideoPicker->SetForegroundColour(lightText);
        wxWindowList& children = m_singleVideoPicker->GetChildren();
        for (auto child : children) {
            child->SetBackgroundColour(darkInput);
            child->SetForegroundColour(lightText);
        }
        m_singleVideoPicker->Refresh();
    }
    if (m_videoFolderPicker) {
        m_videoFolderPicker->SetBackgroundColour(darkInput);
        m_videoFolderPicker->SetForegroundColour(lightText);
        wxWindowList& children = m_videoFolderPicker->GetChildren();
        for (auto child : children) {
            child->SetBackgroundColour(darkInput);
            child->SetForegroundColour(lightText);
        }
        m_videoFolderPicker->Refresh();
    }
    if (m_outputFilePicker) {
        m_outputFilePicker->SetBackgroundColour(darkInput);
        m_outputFilePicker->SetForegroundColour(lightText);
        wxWindowList& children = m_outputFilePicker->GetChildren();
        for (auto child : children) {
            child->SetBackgroundColour(darkInput);
            child->SetForegroundColour(lightText);
        }
        m_outputFilePicker->Refresh();
    }

    // Style dropdowns
    if (m_beatRateChoice) {
        m_beatRateChoice->SetBackgroundColour(darkInput);
        m_beatRateChoice->SetForegroundColour(lightText);
        m_beatRateChoice->Refresh();
    }
    if (m_resolutionChoice) {
        m_resolutionChoice->SetBackgroundColour(darkInput);
        m_resolutionChoice->SetForegroundColour(lightText);
        m_resolutionChoice->Refresh();
    }
    if (m_fpsChoice) {
        m_fpsChoice->SetBackgroundColour(darkInput);
        m_fpsChoice->SetForegroundColour(lightText);
        m_fpsChoice->Refresh();
    }

    // Style text inputs
    if (m_previewTimestampCtrl) {
        m_previewTimestampCtrl->SetBackgroundColour(darkInput);
        m_previewTimestampCtrl->SetForegroundColour(lightText);
        m_previewTimestampCtrl->Refresh();
    }
    if (m_previewBeatsCtrl) {
        m_previewBeatsCtrl->SetBackgroundColour(darkInput);
        m_previewBeatsCtrl->SetForegroundColour(lightText);
        m_previewBeatsCtrl->Refresh();
    }

    // Style buttons
    if (m_startButton) {
        m_startButton->SetBackgroundColour(cyan);
        m_startButton->SetForegroundColour(*wxBLACK);
        m_startButton->SetFont(m_titleFont);
    }

    if (m_cancelButton) {
        m_cancelButton->SetBackgroundColour(purple);
        m_cancelButton->SetForegroundColour(*wxWHITE);
    }
    
    // Style text
    if (m_statusText) {
        m_statusText->SetForegroundColour(cyan);
        m_statusText->SetFont(m_labelFont);
    }
    
    if (m_etaText) {
        m_etaText->SetForegroundColour(purple);
        m_etaText->SetFont(m_labelFont);
    }
}

void MainWindow::CreateControls() {
    m_mainPanel = new wxPanel(this);

    // Bind paint event to draw custom background on panel
    m_mainPanel->Bind(wxEVT_PAINT, [this](wxPaintEvent& evt) {
        wxPaintDC dc(m_mainPanel);
        if (m_backgroundBitmap.IsOk()) {
            dc.DrawBitmap(m_backgroundBitmap, 0, 0, false);
        }
    });

    // Prevent default erase to avoid flicker
    m_mainPanel->Bind(wxEVT_ERASE_BACKGROUND, [](wxEraseEvent& evt) {
        // Do nothing - prevents flicker
    });
    
    // Input Files Section
    m_audioFilePicker = new wxFilePickerCtrl(m_mainPanel, wxID_ANY, "",
        "Select Audio File", 
        "Audio Files (*.wav;*.mp3;*.flac;*.ogg;*.m4a)|*.wav;*.mp3;*.flac;*.ogg;*.m4a",
        wxDefaultPosition, wxDefaultSize, wxFLP_DEFAULT_STYLE | wxFLP_USE_TEXTCTRL);
    
    m_singleVideoRadio = new wxRadioButton(m_mainPanel, wxID_ANY, "Single Video",
        wxDefaultPosition, wxDefaultSize, wxRB_GROUP);
    m_multiClipRadio = new wxRadioButton(m_mainPanel, wxID_ANY, "Multiple Clips");
    m_multiClipRadio->SetValue(true);
    
    m_singleVideoPicker = new wxFilePickerCtrl(m_mainPanel, wxID_ANY, "",
        "Select Video File", 
        "Video Files (*.mp4;*.avi;*.mov;*.mkv;*.webm)|*.mp4;*.avi;*.mov;*.mkv;*.webm");
    m_singleVideoPicker->Show(false);
    
    m_videoFolderPicker = new wxDirPickerCtrl(m_mainPanel, wxID_ANY, "",
        "Select Folder with Video Clips");
    
    // Output Section
    m_outputFilePicker = new wxFilePickerCtrl(m_mainPanel, wxID_ANY, "",
        "Save Output As", "MP4 Files (*.mp4)|*.mp4",
        wxDefaultPosition, wxDefaultSize, wxFLP_SAVE | wxFLP_USE_TEXTCTRL | wxFLP_OVERWRITE_PROMPT);
    
    // Sync Settings
    m_beatRateChoice = new wxChoice(m_mainPanel, wxID_ANY);
    m_beatRateChoice->Append("Every Beat (1x)");
    m_beatRateChoice->Append("Every 2nd Beat (1/2x)");
    m_beatRateChoice->Append("Every 4th Beat (1/4x)");
    m_beatRateChoice->Append("Every 8th Beat (1/8x)");
    m_beatRateChoice->SetSelection(0);
    
    m_resolutionChoice = new wxChoice(m_mainPanel, wxID_ANY);
    m_resolutionChoice->Append("1920x1080 (Full HD)");
    m_resolutionChoice->Append("1280x720 (HD)");
    m_resolutionChoice->Append("3840x2160 (4K)");
    m_resolutionChoice->Append("2560x1440 (2K)");
    m_resolutionChoice->SetSelection(0);
    
    m_fpsChoice = new wxChoice(m_mainPanel, wxID_ANY);
    m_fpsChoice->Append("24 fps (Cinematic)");
    m_fpsChoice->Append("30 fps (Standard)");
    m_fpsChoice->Append("60 fps (Smooth)");
    m_fpsChoice->SetSelection(0);
    
    // Preview Mode
    m_previewModeCheck = new wxCheckBox(m_mainPanel, wxID_ANY, 
        "Preview Mode (Process First N Beats Only)");
    m_previewBeatsCtrl = new wxSpinCtrl(m_mainPanel, wxID_ANY, "10",
        wxDefaultPosition, wxSize(80, -1), wxSP_ARROW_KEYS, 1, 1000, 10);
    m_previewBeatsCtrl->Enable(false);
    
    // Beat Visualizer
    m_beatVisualizer = new BeatVisualizer(m_mainPanel, wxID_ANY, 
        wxDefaultPosition, wxSize(890, 120));
    
    // Video Preview
    m_videoPreview = new VideoPreview(m_mainPanel, wxID_ANY, 
        wxDefaultPosition, wxSize(480, 270));
    
    // Progress Section
    m_progressBar = new wxGauge(m_mainPanel, wxID_ANY, 100, 
        wxDefaultPosition, wxSize(-1, 25));
    m_statusText = new wxStaticText(m_mainPanel, wxID_ANY, "Ready to sync...");
    m_etaText = new wxStaticText(m_mainPanel, wxID_ANY, "");
    
    // Buttons
    m_startButton = new wxButton(m_mainPanel, wxID_ANY, "START SYNC", 
        wxDefaultPosition, wxSize(250, 45));
    m_cancelButton = new wxButton(m_mainPanel, wxID_ANY, "CANCEL",
        wxDefaultPosition, wxSize(120, 45));
    m_cancelButton->Enable(false);

    // Preview Button (shows a single frame at timestamp 0)
    m_previewButton = new wxButton(m_mainPanel, wxID_ANY, "PREVIEW FRAME",
        wxDefaultPosition, wxSize(160, 36));

    // Event bindings
    m_audioFilePicker->Bind(wxEVT_FILEPICKER_CHANGED, &MainWindow::OnAudioSelected, this);
    m_singleVideoRadio->Bind(wxEVT_RADIOBUTTON, &MainWindow::OnVideoSourceChanged, this);
    m_multiClipRadio->Bind(wxEVT_RADIOBUTTON, &MainWindow::OnVideoSourceChanged, this);
    m_startButton->Bind(wxEVT_BUTTON, &MainWindow::OnStartProcessing, this);
    m_cancelButton->Bind(wxEVT_BUTTON, &MainWindow::OnCancelProcessing, this);
    m_previewButton->Bind(wxEVT_BUTTON, &MainWindow::OnPreviewFrame, this);
    
    m_previewModeCheck->Bind(wxEVT_CHECKBOX, [this](wxCommandEvent& e) {
        m_previewBeatsCtrl->Enable(e.IsChecked());
    });
}

void MainWindow::CreateLayout() {
    wxBoxSizer* mainSizer = new wxBoxSizer(wxVERTICAL);
    mainSizer->AddSpacer(15);
    
    // Title
    wxStaticText* title = new wxStaticText(m_mainPanel, wxID_ANY, 
        "TRIP SITTER");
    title->SetFont(wxFont(28, wxFONTFAMILY_MODERN, wxFONTSTYLE_NORMAL, 
        wxFONTWEIGHT_BOLD, false, "Impact"));
    title->SetForegroundColour(wxColour(0, 217, 255));
    mainSizer->Add(title, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5);
    
    wxStaticText* subtitle = new wxStaticText(m_mainPanel, wxID_ANY, 
        "Audio Beat Sync GUI");
    subtitle->SetFont(wxFont(14, wxFONTFAMILY_MODERN, wxFONTSTYLE_ITALIC, 
        wxFONTWEIGHT_NORMAL, false, "Consolas"));
    subtitle->SetForegroundColour(wxColour(139, 0, 255));
    mainSizer->Add(subtitle, 0, wxALIGN_CENTER_HORIZONTAL | wxBOTTOM, 10);
    
    // Input Files Section
    wxStaticText* inputSectionLabel = new wxStaticText(m_mainPanel, wxID_ANY, "══ INPUT FILES ══");
    inputSectionLabel->SetFont(m_labelFont.Bold());
    inputSectionLabel->SetForegroundColour(wxColour(0, 217, 255)); // Cyan
    mainSizer->Add(inputSectionLabel, 0, wxLEFT | wxTOP, 15);
    mainSizer->AddSpacer(5);

    wxBoxSizer* inputBox = new wxBoxSizer(wxVERTICAL);
    wxGridBagSizer* inputGrid = new wxGridBagSizer(8, 10);
    
    wxStaticText* audioLabel = new wxStaticText(m_mainPanel, wxID_ANY, "Audio File:");
    audioLabel->SetForegroundColour(*wxWHITE);
    inputGrid->Add(audioLabel, wxGBPosition(0, 0), wxDefaultSpan, wxALIGN_CENTER_VERTICAL);
    inputGrid->Add(m_audioFilePicker, wxGBPosition(0, 1), wxDefaultSpan, wxEXPAND);
    
    wxStaticText* sourceLabel = new wxStaticText(m_mainPanel, wxID_ANY, "Video Source:");
    sourceLabel->SetForegroundColour(*wxWHITE);
    inputGrid->Add(sourceLabel, wxGBPosition(1, 0), wxDefaultSpan, wxALIGN_CENTER_VERTICAL);
    
    wxBoxSizer* radioSizer = new wxBoxSizer(wxHORIZONTAL);
    m_singleVideoRadio->SetForegroundColour(*wxWHITE);
    m_multiClipRadio->SetForegroundColour(*wxWHITE);
    radioSizer->Add(m_singleVideoRadio, 0, wxALL, 5);
    radioSizer->Add(m_multiClipRadio, 0, wxALL, 5);
    inputGrid->Add(radioSizer, wxGBPosition(1, 1));
    
    inputGrid->Add(m_singleVideoPicker, wxGBPosition(2, 1), wxDefaultSpan, wxEXPAND);
    inputGrid->Add(m_videoFolderPicker, wxGBPosition(3, 1), wxDefaultSpan, wxEXPAND);
    
    inputGrid->AddGrowableCol(1);
    inputBox->Add(inputGrid, 1, wxEXPAND | wxALL, 10);
    mainSizer->Add(inputBox, 0, wxEXPAND | wxLEFT | wxRIGHT, 15);
    mainSizer->AddSpacer(10);

    // Sync Settings Section
    wxStaticText* settingsSectionLabel = new wxStaticText(m_mainPanel, wxID_ANY, "══ SYNC SETTINGS ══");
    settingsSectionLabel->SetFont(m_labelFont.Bold());
    settingsSectionLabel->SetForegroundColour(wxColour(139, 0, 255)); // Purple
    mainSizer->Add(settingsSectionLabel, 0, wxLEFT | wxTOP, 15);
    mainSizer->AddSpacer(5);

    wxBoxSizer* settingsBox = new wxBoxSizer(wxVERTICAL);
    wxGridBagSizer* settingsGrid = new wxGridBagSizer(8, 10);
    
    wxStaticText* beatLabel = new wxStaticText(m_mainPanel, wxID_ANY, "Beat Sync Rate:");
    beatLabel->SetForegroundColour(*wxWHITE);
    settingsGrid->Add(beatLabel, wxGBPosition(0, 0), wxDefaultSpan, wxALIGN_CENTER_VERTICAL);
    settingsGrid->Add(m_beatRateChoice, wxGBPosition(0, 1), wxDefaultSpan, wxEXPAND);
    
    wxStaticText* resLabel = new wxStaticText(m_mainPanel, wxID_ANY, "Output Resolution:");
    resLabel->SetForegroundColour(*wxWHITE);
    settingsGrid->Add(resLabel, wxGBPosition(1, 0), wxDefaultSpan, wxALIGN_CENTER_VERTICAL);
    settingsGrid->Add(m_resolutionChoice, wxGBPosition(1, 1), wxDefaultSpan, wxEXPAND);
    
    wxStaticText* fpsLabel = new wxStaticText(m_mainPanel, wxID_ANY, "Frame Rate:");
    fpsLabel->SetForegroundColour(*wxWHITE);
    settingsGrid->Add(fpsLabel, wxGBPosition(2, 0), wxDefaultSpan, wxALIGN_CENTER_VERTICAL);
    settingsGrid->Add(m_fpsChoice, wxGBPosition(2, 1), wxDefaultSpan, wxEXPAND);
    
    wxBoxSizer* previewSizer = new wxBoxSizer(wxHORIZONTAL);
    m_previewModeCheck->SetForegroundColour(*wxWHITE);
    previewSizer->Add(m_previewModeCheck, 0, wxALIGN_CENTER_VERTICAL);
    previewSizer->AddSpacer(10);
    previewSizer->Add(m_previewBeatsCtrl, 0);
    settingsGrid->Add(previewSizer, wxGBPosition(3, 0), wxGBSpan(1, 2));
    
    settingsGrid->AddGrowableCol(1);
    settingsBox->Add(settingsGrid, 1, wxEXPAND | wxALL, 10);
    mainSizer->Add(settingsBox, 0, wxEXPAND | wxLEFT | wxRIGHT, 15);
    mainSizer->AddSpacer(10);
    
    // Beat Visualizer
    wxStaticText* vizLabel = new wxStaticText(m_mainPanel, wxID_ANY, "Beat Visualization:");
    vizLabel->SetForegroundColour(wxColour(255, 136, 0));
    mainSizer->Add(vizLabel, 0, wxLEFT, 15);
    mainSizer->Add(m_beatVisualizer, 0, wxEXPAND | wxLEFT | wxRIGHT, 15);
    mainSizer->AddSpacer(10);
    
    // Video Preview
    wxStaticText* prevLabel = new wxStaticText(m_mainPanel, wxID_ANY, "Video Preview:");
    prevLabel->SetForegroundColour(wxColour(255, 136, 0));
    mainSizer->Add(prevLabel, 0, wxLEFT, 15);

    // Add preview button and timestamp input above the preview area
    wxBoxSizer* previewHeader = new wxBoxSizer(wxHORIZONTAL);
    previewHeader->Add(m_previewButton, 0, wxLEFT, 15);

    wxStaticText* tsLabel = new wxStaticText(m_mainPanel, wxID_ANY, "Timestamp (s):");
    tsLabel->SetForegroundColour(*wxWHITE);
    m_previewTimestampCtrl = new wxTextCtrl(m_mainPanel, wxID_ANY, "0.5",
        wxDefaultPosition, wxSize(90, -1));
    previewHeader->Add(tsLabel, 0, wxALIGN_CENTER_VERTICAL | wxLEFT, 10);
    previewHeader->Add(m_previewTimestampCtrl, 0, wxALIGN_CENTER_VERTICAL | wxLEFT, 5);

    previewHeader->AddStretchSpacer(1);
    mainSizer->Add(previewHeader, 0, wxEXPAND | wxRIGHT | wxLEFT, 15);

    mainSizer->Add(m_videoPreview, 0, wxALIGN_CENTER_HORIZONTAL);
    mainSizer->AddSpacer(10);

    // Output Section
    wxStaticText* outputSectionLabel = new wxStaticText(m_mainPanel, wxID_ANY, "══ OUTPUT ══");
    outputSectionLabel->SetFont(m_labelFont.Bold());
    outputSectionLabel->SetForegroundColour(wxColour(0, 217, 255)); // Cyan
    mainSizer->Add(outputSectionLabel, 0, wxLEFT | wxTOP, 15);
    mainSizer->AddSpacer(5);

    wxBoxSizer* outputBox = new wxBoxSizer(wxHORIZONTAL);
    wxStaticText* outputLabel = new wxStaticText(m_mainPanel, wxID_ANY, "Save To:");
    outputLabel->SetForegroundColour(*wxWHITE);
    outputBox->Add(outputLabel, 0, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    outputBox->Add(m_outputFilePicker, 1, wxEXPAND | wxALL, 5);
    mainSizer->Add(outputBox, 0, wxEXPAND | wxLEFT | wxRIGHT, 15);
    mainSizer->AddSpacer(15);
    
    // Action Buttons
    wxBoxSizer* buttonSizer = new wxBoxSizer(wxHORIZONTAL);
    buttonSizer->Add(m_startButton, 0, wxALL, 5);
    buttonSizer->Add(m_cancelButton, 0, wxALL, 5);
    mainSizer->Add(buttonSizer, 0, wxALIGN_CENTER_HORIZONTAL);
    mainSizer->AddSpacer(15);

    // Progress Section
    wxStaticText* progressSectionLabel = new wxStaticText(m_mainPanel, wxID_ANY, "══ PROGRESS ══");
    progressSectionLabel->SetFont(m_labelFont.Bold());
    progressSectionLabel->SetForegroundColour(wxColour(139, 0, 255)); // Purple
    mainSizer->Add(progressSectionLabel, 0, wxLEFT | wxTOP, 15);
    mainSizer->AddSpacer(5);

    wxBoxSizer* progressBox = new wxBoxSizer(wxVERTICAL);
    progressBox->Add(m_statusText, 0, wxALL | wxEXPAND, 5);
    progressBox->Add(m_progressBar, 0, wxEXPAND | wxALL, 5);
    progressBox->Add(m_etaText, 0, wxALL | wxALIGN_RIGHT, 5);
    mainSizer->Add(progressBox, 0, wxEXPAND | wxLEFT | wxRIGHT, 15);
    mainSizer->AddSpacer(15);
    
    m_mainPanel->SetSizer(mainSizer);
    
    wxBoxSizer* frameSizer = new wxBoxSizer(wxVERTICAL);
    frameSizer->Add(m_mainPanel, 1, wxEXPAND);
    SetSizer(frameSizer);
}

void MainWindow::OnVideoSourceChanged(wxCommandEvent& event) {
    bool isSingle = m_singleVideoRadio->GetValue();
    m_singleVideoPicker->Show(isSingle);
    m_videoFolderPicker->Show(!isSingle);
    m_mainPanel->Layout();
}

void MainWindow::OnAudioSelected(wxFileDirPickerEvent& event) {
    wxString audioPath = event.GetPath();
    if (!audioPath.IsEmpty() && wxFileExists(audioPath)) {
        // Analyze audio and update beat visualizer
        m_statusText->SetLabel("Analyzing audio...");
        m_beatVisualizer->LoadAudio(audioPath);
        m_statusText->SetLabel("Audio loaded - Ready to sync");
    }
}

void MainWindow::OnPreviewFrame(wxCommandEvent& event) {
    // Determine selected video file
    wxString videoPath;
    if (m_multiClipRadio->GetValue()) {
        wxString folder = m_videoFolderPicker->GetPath();
        if (folder.IsEmpty() || !wxDirExists(folder)) {
            wxMessageBox("Please select a video folder to preview", "No Video", wxOK | wxICON_WARNING, this);
            return;
        }
        // find first video file (common extensions)
        wxArrayString files;
        wxDir::GetAllFiles(folder, &files, "*.mp4;*.avi;*.mov;*.mkv;*.webm", wxDIR_FILES);
        if (files.IsEmpty()) {
            wxMessageBox("No video files found in folder", "No Video", wxOK | wxICON_WARNING, this);
            return;
        }
        videoPath = files[0];
    } else {
        videoPath = m_singleVideoPicker->GetPath();
        if (videoPath.IsEmpty() || !wxFileExists(videoPath)) {
            wxMessageBox("Please select a video file to preview", "No Video", wxOK | wxICON_WARNING, this);
            return;
        }
    }

    // Read timestamp from input (seconds)
    double ts = 0.5;
    wxString tsStr = m_previewTimestampCtrl->GetValue();
    if (!tsStr.IsEmpty()) {
        ts = wxAtof(tsStr);
    }
    if (ts < 0.0) ts = 0.0;

    // Clamp to video duration if possible
    {
        BeatSync::VideoProcessor vp;
        if (vp.open(videoPath.ToStdString())) {
            double dur = vp.getInfo().duration;
            if (dur > 0 && ts > dur - 0.1) ts = std::max(0.0, dur - 0.1);
        }
    }

    m_videoPreview->LoadFrame(videoPath, ts);
}



void MainWindow::OnStartProcessing(wxCommandEvent& event) {
    // Validate inputs
    wxString audioPath = m_audioFilePicker->GetPath();
    wxString videoPath = m_multiClipRadio->GetValue() ? 
        m_videoFolderPicker->GetPath() : m_singleVideoPicker->GetPath();
    wxString outputPath = m_outputFilePicker->GetPath();
    
    if (audioPath.IsEmpty()) {
        wxMessageBox("Please select an audio file", "Input Required", 
            wxOK | wxICON_WARNING, this);
        return;
    }
    
    if (videoPath.IsEmpty()) {
        wxMessageBox("Please select a video source", "Input Required", 
            wxOK | wxICON_WARNING, this);
        return;
    }
    
    if (outputPath.IsEmpty()) {
        wxMessageBox("Please specify an output file", "Input Required", 
            wxOK | wxICON_WARNING, this);
        return;
    }
    
    // Build config
    ProcessingConfig config;
    config.audioPath = audioPath;
    config.videoPath = videoPath;
    config.isMultiClip = m_multiClipRadio->GetValue();
    config.outputPath = outputPath;
    config.beatRate = m_beatRateChoice->GetSelection();
    config.resolution = m_resolutionChoice->GetStringSelection();
    config.fps = wxAtoi(m_fpsChoice->GetStringSelection().BeforeFirst(' '));
    config.previewMode = m_previewModeCheck->GetValue();
    config.previewBeats = m_previewBeatsCtrl->GetValue();
    
    UpdateUIState(true);
    StartProcessing(config);
}

void MainWindow::OnCancelProcessing(wxCommandEvent& event) {
    if (wxMessageBox("Are you sure you want to cancel processing?", 
        "Confirm Cancel", wxYES_NO | wxICON_QUESTION, this) == wxYES) {
        m_cancelRequested = true;
        m_statusText->SetLabel("Cancelling...");
    }
}

void MainWindow::OnClose(wxCloseEvent& event) {
    if (m_processingThread && m_processingThread->joinable()) {
        if (wxMessageBox("Processing is in progress. Are you sure you want to exit?", 
            "Confirm Exit", wxYES_NO | wxICON_QUESTION, this) != wxYES) {
            event.Veto();
            return;
        }
        m_cancelRequested = true;
        m_processingThread->join();
    }
    event.Skip();
}

void MainWindow::OnPaint(wxPaintEvent& event) {
    wxPaintDC dc(this);
    if (m_backgroundBitmap.IsOk()) {
        dc.DrawBitmap(m_backgroundBitmap, 0, 0, false);
    }
}

void MainWindow::UpdateUIState(bool processing) {
    m_audioFilePicker->Enable(!processing);
    m_singleVideoPicker->Enable(!processing);
    m_videoFolderPicker->Enable(!processing);
    m_singleVideoRadio->Enable(!processing);
    m_multiClipRadio->Enable(!processing);
    m_outputFilePicker->Enable(!processing);
    m_beatRateChoice->Enable(!processing);
    m_resolutionChoice->Enable(!processing);
    m_fpsChoice->Enable(!processing);
    m_previewModeCheck->Enable(!processing);
    m_previewBeatsCtrl->Enable(!processing && m_previewModeCheck->GetValue());
    m_startButton->Enable(!processing);
    m_cancelButton->Enable(processing);
    
    if (!processing) {
        m_progressBar->SetValue(0);
        m_etaText->SetLabel("");
    }
}

void MainWindow::StartProcessing(const ProcessingConfig& config) {
    m_cancelRequested = false;
    
    m_processingThread = std::make_unique<std::thread>([this, config]() {
        auto startTime = std::chrono::steady_clock::now();
        
        try {
            // Step 1: Analyze audio
            wxThreadEvent* evt = new wxThreadEvent(wxEVT_PROCESSING_PROGRESS);
            evt->SetInt(5);
            evt->SetString("Analyzing audio beats...");
            wxQueueEvent(this, evt);
            
            BeatSync::AudioAnalyzer analyzer;
            BeatSync::BeatGrid beatGrid = analyzer.analyze(config.audioPath.ToStdString());
            
            if (m_cancelRequested) {
                wxThreadEvent* cancelEvt = new wxThreadEvent(wxEVT_PROCESSING_COMPLETE);
                cancelEvt->SetInt(0);
                cancelEvt->SetString("Cancelled by user");
                wxQueueEvent(this, cancelEvt);
                return;
            }
            
            // Apply beat rate
            int beatMultiplier = 1 << config.beatRate; // 0->1, 1->2, 2->4, 3->8
            std::vector<double> filteredBeats;
            for (size_t i = 0; i < beatGrid.getBeats().size(); i += beatMultiplier) {
                filteredBeats.push_back(beatGrid.getBeats()[i]);
            }
            
            // Preview mode
            if (config.previewMode && filteredBeats.size() > config.previewBeats) {
                filteredBeats.resize(config.previewBeats);
            }
            
            evt = new wxThreadEvent(wxEVT_PROCESSING_PROGRESS);
            evt->SetInt(10);
            evt->SetString(wxString::Format("Found %zu beats", filteredBeats.size()));
            wxQueueEvent(this, evt);
            
            // Step 2: Get video clips
            std::vector<std::string> videoFiles;
            if (config.isMultiClip) {
                namespace fs = std::filesystem;
                for (const auto& entry : fs::directory_iterator(config.videoPath.ToStdString())) {
                    if (entry.is_regular_file()) {
                        std::string ext = entry.path().extension().string();
                        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                        if (ext == ".mp4" || ext == ".avi" || ext == ".mov" || 
                            ext == ".mkv" || ext == ".webm") {
                            videoFiles.push_back(entry.path().string());
                        }
                    }
                }
                
                if (videoFiles.empty()) {
                    throw std::runtime_error("No video files found in folder");
                }
            } else {
                videoFiles.push_back(config.videoPath.ToStdString());
            }
            
            // Step 3: Extract resolution
            int width = 1920, height = 1080;
            if (config.resolution.Contains("1280x720")) {
                width = 1280; height = 720;
            } else if (config.resolution.Contains("3840x2160")) {
                width = 3840; height = 2160;
            } else if (config.resolution.Contains("2560x1440")) {
                width = 2560; height = 1440;
            }
            
            // Step 4: Process segments
            BeatSync::VideoWriter writer;
            std::vector<std::string> segmentPaths;
            
            for (size_t i = 0; i < filteredBeats.size(); i++) {
                if (m_cancelRequested) {
                    // Cleanup temp files
                    for (const auto& seg : segmentPaths) {
                        std::remove(seg.c_str());
                    }
                    wxThreadEvent* cancelEvt = new wxThreadEvent(wxEVT_PROCESSING_COMPLETE);
                    cancelEvt->SetInt(0);
                    cancelEvt->SetString("Cancelled by user");
                    wxQueueEvent(this, cancelEvt);
                    return;
                }
                
                double segStartTime = filteredBeats[i];
                double duration = (i + 1 < filteredBeats.size()) ? 
                    (filteredBeats[i + 1] - segStartTime) : 
                    (beatGrid.getAudioDuration() - segStartTime);
                
                std::string videoFile = videoFiles[i % videoFiles.size()];
                std::string segmentPath = "temp_segment_" + std::to_string(i) + ".mp4";
                
                writer.copySegmentFast(videoFile, segStartTime, duration, segmentPath);
                segmentPaths.push_back(segmentPath);
                
                int progress = 10 + (int)(80.0 * (i + 1) / filteredBeats.size());
                auto elapsed = std::chrono::steady_clock::now() - startTime;
                auto elapsedSec = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
                int eta = (elapsedSec * filteredBeats.size()) / (i + 1) - elapsedSec;
                
                evt = new wxThreadEvent(wxEVT_PROCESSING_PROGRESS);
                evt->SetInt(progress);
                evt->SetString(wxString::Format("Processing segment %zu/%zu", i + 1, filteredBeats.size()));
                evt->SetExtraLong(eta);
                wxQueueEvent(this, evt);
            }
            
            // Step 5: Concatenate
            evt = new wxThreadEvent(wxEVT_PROCESSING_PROGRESS);
            evt->SetInt(90);
            evt->SetString("Concatenating segments...");
            wxQueueEvent(this, evt);
            
            writer.concatenateVideos(segmentPaths, "temp_video.mp4");
            
            // Step 6: Mux with audio
            evt = new wxThreadEvent(wxEVT_PROCESSING_PROGRESS);
            evt->SetInt(95);
            evt->SetString("Muxing audio...");
            wxQueueEvent(this, evt);
            
            writer.addAudioTrack("temp_video.mp4", config.audioPath.ToStdString(), 
                config.outputPath.ToStdString(), false);
            
            // Cleanup
            for (const auto& seg : segmentPaths) {
                std::remove(seg.c_str());
            }
            std::remove("temp_video.mp4");
            
            // Success
            wxThreadEvent* successEvt = new wxThreadEvent(wxEVT_PROCESSING_COMPLETE);
            successEvt->SetInt(1);
            successEvt->SetString("Processing complete!");
            wxQueueEvent(this, successEvt);
            
        } catch (const std::exception& e) {
            wxThreadEvent* errorEvt = new wxThreadEvent(wxEVT_PROCESSING_COMPLETE);
            errorEvt->SetInt(0);
            errorEvt->SetString(wxString::Format("Error: %s", e.what()));
            wxQueueEvent(this, errorEvt);
        }
    });
}

void MainWindow::UpdateProgress(int percent, const wxString& status, const wxString& eta) {
    m_progressBar->SetValue(percent);
    m_statusText->SetLabel(status);
    m_etaText->SetLabel(eta);
}

void MainWindow::OnProcessingComplete(bool success, const wxString& message) {
    UpdateUIState(false);
    
    if (m_processingThread && m_processingThread->joinable()) {
        m_processingThread->join();
    }
    
    if (success) {
        m_progressBar->SetValue(100);
        m_statusText->SetLabel(message);
        wxMessageBox(message, "Success", wxOK | wxICON_INFORMATION, this);
    } else {
        m_statusText->SetLabel(message);
        wxMessageBox(message, "Error", wxOK | wxICON_ERROR, this);
    }
}

void MainWindow::LoadSettings() {
    m_beatRateChoice->SetSelection(m_settingsManager->GetInt("BeatRate", 0));
    m_resolutionChoice->SetSelection(m_settingsManager->GetInt("Resolution", 0));
    m_fpsChoice->SetSelection(m_settingsManager->GetInt("FPS", 0));
    m_previewModeCheck->SetValue(m_settingsManager->GetBool("PreviewMode", false));
    m_previewBeatsCtrl->SetValue(m_settingsManager->GetInt("PreviewBeats", 10));
    m_previewTimestampCtrl->SetValue(m_settingsManager->GetString("PreviewTimestamp", "0.5"));
    
    wxString lastAudio = m_settingsManager->GetString("LastAudioPath", "");
    if (!lastAudio.IsEmpty() && wxFileExists(lastAudio)) {
        m_audioFilePicker->SetPath(lastAudio);
    }
    
    wxString lastVideo = m_settingsManager->GetString("LastVideoPath", "");
    if (!lastVideo.IsEmpty()) {
        if (wxDirExists(lastVideo)) {
            m_videoFolderPicker->SetPath(lastVideo);
            m_multiClipRadio->SetValue(true);
        } else if (wxFileExists(lastVideo)) {
            m_singleVideoPicker->SetPath(lastVideo);
            m_singleVideoRadio->SetValue(true);
        }
    }
    
    OnVideoSourceChanged(wxCommandEvent());
}

void MainWindow::SaveSettings() {
    m_settingsManager->SetInt("BeatRate", m_beatRateChoice->GetSelection());
    m_settingsManager->SetInt("Resolution", m_resolutionChoice->GetSelection());
    m_settingsManager->SetInt("FPS", m_fpsChoice->GetSelection());
    m_settingsManager->SetBool("PreviewMode", m_previewModeCheck->GetValue());
    m_settingsManager->SetInt("PreviewBeats", m_previewBeatsCtrl->GetValue());
    m_settingsManager->SetString("PreviewTimestamp", m_previewTimestampCtrl->GetValue());
    m_settingsManager->SetString("LastAudioPath", m_audioFilePicker->GetPath());
    m_settingsManager->SetString("LastVideoPath", 
        m_multiClipRadio->GetValue() ? m_videoFolderPicker->GetPath() : m_singleVideoPicker->GetPath());
}
