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
#include <wx/dcbuffer.h>
#include <wx/clipbrd.h>
#include <wx/textdlg.h>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <sstream>
#include "../utils/LogArchiver.h"

#ifdef __WXUNIVERSAL__
#include "PsychedelicTheme.h"
#endif

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
              wxDefaultPosition, wxSize(1344, 950),
              wxDEFAULT_FRAME_STYLE & ~(wxRESIZE_BORDER | wxMAXIMIZE_BOX))
{
    m_settingsManager = std::make_unique<SettingsManager>();
    
    SetupFonts();
    LoadBackgroundImage();
    CreateControls();
    CreateLayout();
    ApplyPsychedelicStyling();

#ifdef __WXUNIVERSAL__
    // Make all child windows have dark backgrounds to complement psychedelic background
    SetAllChildrenTransparent(m_mainPanel);
    // Force refresh to apply the new colors
    m_mainPanel->Refresh();
    m_mainPanel->Update();
#endif

    LoadSettings();
    
    Centre();

    // Add menu bar with Help -> View Logs & Diagnostics
    const int ID_VIEW_LOGS = wxID_HIGHEST + 1;
    wxMenuBar* menuBar = new wxMenuBar();
    wxMenu* helpMenu = new wxMenu();
    helpMenu->Append(ID_VIEW_LOGS, "View Logs & Diagnostics...");
    menuBar->Append(helpMenu, "&Help");
    SetMenuBar(menuBar);
    Bind(wxEVT_MENU, &MainWindow::OnViewLogs, this, ID_VIEW_LOGS);
    
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
    wxString exePath = wxStandardPaths::Get().GetExecutablePath();
    wxString assetsDir = wxFileName(exePath).GetPath() + "/assets/";

    // Try the upscaled version first, then fall back to background.png
    wxArrayString candidates;
    candidates.Add(assetsDir + "ComfyUI_03324_.png");
    candidates.Add(assetsDir + "background.png");

    wxImage img;
    for (size_t i = 0; i < candidates.GetCount(); ++i) {
        if (wxFileExists(candidates[i])) {
            img.LoadFile(candidates[i], wxBITMAP_TYPE_PNG);
            if (img.IsOk()) {
                break;
            }
        }
    }

    if (img.IsOk()) {
        // Scale image to fit the window size (1344x768) while preserving aspect ratio
        int targetW = 1344;
        int targetH = 768;

        int imgW = img.GetWidth();
        int imgH = img.GetHeight();

        // Calculate scaling to cover the entire window
        double scaleW = (double)targetW / imgW;
        double scaleH = (double)targetH / imgH;
        double scale = std::max(scaleW, scaleH);  // Use max to cover, min to fit

        int newW = (int)(imgW * scale);
        int newH = (int)(imgH * scale);

        img.Rescale(newW, newH, wxIMAGE_QUALITY_HIGH);
        m_backgroundBitmap = wxBitmap(img);
    }

    // Fallback: Create gradient background if no image loaded
    if (!m_backgroundBitmap.IsOk()) {
        wxBitmap bmp(1344, 768);
        wxMemoryDC dc(bmp);
        dc.GradientFillLinear(wxRect(0, 0, 1344, 768),
            wxColour(10, 10, 26), wxColour(25, 0, 50), wxSOUTH);
        m_backgroundBitmap = bmp;
    }
}

void MainWindow::ApplyPsychedelicStyling() {
#ifdef __WXUNIVERSAL__
    // When using wxUniversal, the PsychedelicTheme handles all styling
    // Just set the basic background color
    SetBackgroundColour(TripSitter::PsychedelicColors::Background());
    if (m_mainPanel) {
        m_mainPanel->SetBackgroundColour(TripSitter::PsychedelicColors::Background());
    }
    return;
#endif

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
    m_mainPanel = new wxScrolledWindow(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxHSCROLL | wxVSCROLL | wxCLIP_CHILDREN | wxFULL_REPAINT_ON_RESIZE);
    m_mainPanel->SetScrollRate(10, 10);
    m_mainPanel->SetVirtualSize(1344, 1400);  // Taller virtual size for scrolling
    m_mainPanel->SetDoubleBuffered(true);  // Enable double buffering

#ifdef __WXUNIVERSAL__
    // wxUniversal: Custom paint for static background
    m_mainPanel->SetBackgroundStyle(wxBG_STYLE_PAINT);

    // Disable erase background to prevent flicker
    m_mainPanel->Bind(wxEVT_ERASE_BACKGROUND, [](wxEraseEvent& evt) {
        // Do nothing - we handle everything in paint
    });

    m_mainPanel->Bind(wxEVT_PAINT, [this](wxPaintEvent& evt) {
        wxBufferedPaintDC dc(m_mainPanel);

        // Get the scroll position and client size
        int scrollX, scrollY;
        m_mainPanel->GetViewStart(&scrollX, &scrollY);
        int scrollUnitX, scrollUnitY;
        m_mainPanel->GetScrollPixelsPerUnit(&scrollUnitX, &scrollUnitY);

        // Convert scroll units to pixels
        int scrollPixelsX = scrollX * scrollUnitX;
        int scrollPixelsY = scrollY * scrollUnitY;

        // Get client size (viewport size in pixels)
        wxSize clientSize = m_mainPanel->GetClientSize();

        // Draw static background ALWAYS at 0,0 regardless of scroll position
        // Fill the entire client area (stretch to fit)
        if (m_backgroundBitmap.IsOk()) {
            wxImage img = m_backgroundBitmap.ConvertToImage();

            // Calculate scaling to cover entire area (aspect-fill)
            wxSize bmpSize = m_backgroundBitmap.GetSize();
            double scaleW = (double)clientSize.x / bmpSize.x;
            double scaleH = (double)clientSize.y / bmpSize.y;
            double scale = std::max(scaleW, scaleH);

            int scaledW = (int)(bmpSize.x * scale);
            int scaledH = (int)(bmpSize.y * scale);

            // Center the image
            int offsetX = (clientSize.x - scaledW) / 2;
            int offsetY = (clientSize.y - scaledH) / 2;

            // Only rescale if size changed (cache the result)
            static wxSize lastClientSize(0, 0);
            static wxBitmap cachedScaledBitmap;

            if (lastClientSize != clientSize || !cachedScaledBitmap.IsOk()) {
                wxImage scaledImg = img.Scale(scaledW, scaledH, wxIMAGE_QUALITY_BILINEAR);
                cachedScaledBitmap = wxBitmap(scaledImg);
                lastClientSize = clientSize;
            }

            dc.DrawBitmap(cachedScaledBitmap, offsetX, offsetY, false);
        } else {
            dc.GradientFillLinear(wxRect(0, 0, clientSize.x, clientSize.y),
                wxColour(10, 10, 26), wxColour(25, 0, 50), wxSOUTH);
        }

        // Now manually draw children with scroll offset applied
        // This gives us full control over the paint order
        dc.SetDeviceOrigin(-scrollPixelsX, -scrollPixelsY);

        // CRITICAL: Now let wxWidgets paint children with the scroll offset
        evt.Skip();
    });

    // Force full repaint on scroll to avoid glitching
    m_mainPanel->Bind(wxEVT_SCROLLWIN_TOP, [this](wxScrollWinEvent& evt) { evt.Skip(); m_mainPanel->RefreshRect(m_mainPanel->GetClientRect(), false); m_mainPanel->Update(); });
    m_mainPanel->Bind(wxEVT_SCROLLWIN_BOTTOM, [this](wxScrollWinEvent& evt) { evt.Skip(); m_mainPanel->RefreshRect(m_mainPanel->GetClientRect(), false); m_mainPanel->Update(); });
    m_mainPanel->Bind(wxEVT_SCROLLWIN_LINEUP, [this](wxScrollWinEvent& evt) { evt.Skip(); m_mainPanel->RefreshRect(m_mainPanel->GetClientRect(), false); m_mainPanel->Update(); });
    m_mainPanel->Bind(wxEVT_SCROLLWIN_LINEDOWN, [this](wxScrollWinEvent& evt) { evt.Skip(); m_mainPanel->RefreshRect(m_mainPanel->GetClientRect(), false); m_mainPanel->Update(); });
    m_mainPanel->Bind(wxEVT_SCROLLWIN_PAGEUP, [this](wxScrollWinEvent& evt) { evt.Skip(); m_mainPanel->RefreshRect(m_mainPanel->GetClientRect(), false); m_mainPanel->Update(); });
    m_mainPanel->Bind(wxEVT_SCROLLWIN_PAGEDOWN, [this](wxScrollWinEvent& evt) { evt.Skip(); m_mainPanel->RefreshRect(m_mainPanel->GetClientRect(), false); m_mainPanel->Update(); });
    m_mainPanel->Bind(wxEVT_SCROLLWIN_THUMBTRACK, [this](wxScrollWinEvent& evt) { evt.Skip(); m_mainPanel->RefreshRect(m_mainPanel->GetClientRect(), false); m_mainPanel->Update(); });
    m_mainPanel->Bind(wxEVT_SCROLLWIN_THUMBRELEASE, [this](wxScrollWinEvent& evt) { evt.Skip(); m_mainPanel->RefreshRect(m_mainPanel->GetClientRect(), false); m_mainPanel->Update(); });
#else
    // Native widgets: Complex paint handling for static background
    m_mainPanel->SetBackgroundStyle(wxBG_STYLE_PAINT);

    // Aggressive refresh: refresh child rectangles on scroll/resize to minimize trailing artifacts
    auto refreshChildRects = [this]() {
        wxWindowList& children = m_mainPanel->GetChildren();
        for (wxWindowList::iterator it = children.begin(); it != children.end(); ++it) {
            wxWindow* child = *it;
            if (!child->IsShown()) continue;
            wxRect r = child->GetRect();
            // Refresh the area occupied by this child to force a redraw
            m_mainPanel->RefreshRect(r, false);
        }
        // Fallback to refresh whole client area (non-blocking)
        m_mainPanel->Refresh(false);
    };

    m_mainPanel->Bind(wxEVT_MOUSEWHEEL, [this, refreshChildRects](wxMouseEvent& ev){
        ev.Skip();
        // Use CallAfter to coalesce rapid events
        m_mainPanel->CallAfter(refreshChildRects);
    });

    // Scroll events (thumb track/release) — refresh child rects
    m_mainPanel->Bind(wxEVT_SCROLLWIN_THUMBTRACK, [this, refreshChildRects](wxScrollWinEvent&){ m_mainPanel->CallAfter(refreshChildRects); });
    m_mainPanel->Bind(wxEVT_SCROLLWIN_THUMBRELEASE, [this, refreshChildRects](wxScrollWinEvent&){ m_mainPanel->CallAfter(refreshChildRects); });

    m_mainPanel->Bind(wxEVT_SIZE, [this, refreshChildRects](wxSizeEvent&){ m_mainPanel->CallAfter(refreshChildRects); });

    // Draw background at fixed position (compensate for scroll offset)
    m_mainPanel->Bind(wxEVT_PAINT, [this](wxPaintEvent& evt) {
        // Use auto-buffered paint DC to avoid flicker
        wxAutoBufferedPaintDC dc(m_mainPanel);

        // Get scroll position (in scroll units) and pixels per unit
        int viewX, viewY;
        m_mainPanel->GetViewStart(&viewX, &viewY);
        int pixelX, pixelY;
        m_mainPanel->GetScrollPixelsPerUnit(&pixelX, &pixelY);

        // Compute negated offset to keep bitmap fixed relative to window
        int offsetX = -viewX * pixelX;
        int offsetY = -viewY * pixelY;

        if (m_backgroundBitmap.IsOk()) {
            // Tile the bitmap to ensure the visible client area is fully covered
            wxSize bmpSize = m_backgroundBitmap.GetSize();
            wxSize client = m_mainPanel->GetClientSize();
            for (int y = offsetY - bmpSize.y; y < client.y + bmpSize.y; y += bmpSize.y) {
                for (int x = offsetX - bmpSize.x; x < client.x + bmpSize.x; x += bmpSize.x) {
                    dc.DrawBitmap(m_backgroundBitmap, x, y, false);
                }
            }
        } else {
            // Clear to background colour if no bitmap
            dc.SetBackground(wxBrush(GetBackgroundColour()));
            dc.Clear();
        }

        // Prepare DC so child controls are drawn in the correct scrolled coordinates
        m_mainPanel->DoPrepareDC(dc);
    });

    // Prevent default erase to avoid flicker
    m_mainPanel->Bind(wxEVT_ERASE_BACKGROUND, [](wxEraseEvent& evt) {
        // Do nothing - prevents flicker
    });
#endif
    
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

    // Effects Controls
    m_colorGradeCheck = new wxCheckBox(m_mainPanel, wxID_ANY, "Color Grade");
    m_colorPresetChoice = new wxChoice(m_mainPanel, wxID_ANY);
    m_colorPresetChoice->Append("Warm");
    m_colorPresetChoice->Append("Cool");
    m_colorPresetChoice->Append("Vintage");
    m_colorPresetChoice->Append("Vibrant");
    m_colorPresetChoice->SetSelection(0);
    m_colorPresetChoice->Enable(false);

    m_vignetteCheck = new wxCheckBox(m_mainPanel, wxID_ANY, "Vignette");
    m_beatFlashCheck = new wxCheckBox(m_mainPanel, wxID_ANY, "Beat Flash");
    m_beatZoomCheck = new wxCheckBox(m_mainPanel, wxID_ANY, "Beat Zoom Pulse");

    m_colorGradeCheck->Bind(wxEVT_CHECKBOX, [this](wxCommandEvent& e) {
        m_colorPresetChoice->Enable(e.IsChecked());
    });

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
    wxStaticText* inputSectionLabel = new wxStaticText(m_mainPanel, wxID_ANY, "INPUT FILES");
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
    wxStaticText* settingsSectionLabel = new wxStaticText(m_mainPanel, wxID_ANY, "SYNC SETTINGS");
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

    // Effects Section
    wxStaticText* effectsSectionLabel = new wxStaticText(m_mainPanel, wxID_ANY, "EFFECTS");
    effectsSectionLabel->SetFont(m_labelFont.Bold());
    effectsSectionLabel->SetForegroundColour(wxColour(255, 0, 128)); // Pink
    mainSizer->Add(effectsSectionLabel, 0, wxLEFT | wxTOP, 15);
    mainSizer->AddSpacer(5);

    wxStaticBox* effectsStaticBox = new wxStaticBox(m_mainPanel, wxID_ANY, "");
    effectsStaticBox->SetForegroundColour(wxColour(255, 0, 128));
    wxStaticBoxSizer* effectsBox = new wxStaticBoxSizer(effectsStaticBox, wxVERTICAL);

    wxBoxSizer* effectsRow1 = new wxBoxSizer(wxHORIZONTAL);
    m_colorGradeCheck->SetForegroundColour(*wxWHITE);
    effectsRow1->Add(m_colorGradeCheck, 0, wxALIGN_CENTER_VERTICAL | wxRIGHT, 5);
    effectsRow1->Add(m_colorPresetChoice, 0, wxRIGHT, 20);
    m_vignetteCheck->SetForegroundColour(*wxWHITE);
    effectsRow1->Add(m_vignetteCheck, 0, wxALIGN_CENTER_VERTICAL);
    effectsBox->Add(effectsRow1, 0, wxALL, 5);

    wxBoxSizer* effectsRow2 = new wxBoxSizer(wxHORIZONTAL);
    m_beatFlashCheck->SetForegroundColour(*wxWHITE);
    effectsRow2->Add(m_beatFlashCheck, 0, wxALIGN_CENTER_VERTICAL | wxRIGHT, 20);
    m_beatZoomCheck->SetForegroundColour(*wxWHITE);
    effectsRow2->Add(m_beatZoomCheck, 0, wxALIGN_CENTER_VERTICAL);
    effectsBox->Add(effectsRow2, 0, wxALL, 5);

    mainSizer->Add(effectsBox, 0, wxEXPAND | wxLEFT | wxRIGHT, 15);
    mainSizer->AddSpacer(10);

    // Beat Visualizer Section
    wxStaticText* vizLabel = new wxStaticText(m_mainPanel, wxID_ANY, "BEAT VISUALIZATION");
    vizLabel->SetFont(m_labelFont.Bold());
    vizLabel->SetForegroundColour(wxColour(255, 136, 0));
    mainSizer->Add(vizLabel, 0, wxLEFT | wxTOP, 15);
    mainSizer->AddSpacer(5);
    mainSizer->Add(m_beatVisualizer, 0, wxEXPAND | wxLEFT | wxRIGHT, 15);
    mainSizer->AddSpacer(20);

    // Video Preview Section
    wxStaticText* prevLabel = new wxStaticText(m_mainPanel, wxID_ANY, "VIDEO PREVIEW");
    prevLabel->SetFont(m_labelFont.Bold());
    prevLabel->SetForegroundColour(wxColour(255, 136, 0));
    mainSizer->Add(prevLabel, 0, wxLEFT, 15);

    // Preview controls
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
    wxStaticText* outputSectionLabel = new wxStaticText(m_mainPanel, wxID_ANY, "OUTPUT");
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
    wxStaticText* progressSectionLabel = new wxStaticText(m_mainPanel, wxID_ANY, "PROGRESS");
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
    
    // Normalize output extension to .mp4 if the user omitted it
    {
        wxFileName outFn(outputPath);
        if (!outFn.HasExt()) {
            outFn.SetExt("mp4");
            outputPath = outFn.GetFullPath();
            m_outputFilePicker->SetPath(outputPath);
        }
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

    // Get selection range from beat visualizer
    auto [selStart, selEnd] = m_beatVisualizer->GetSelectionRange();
    config.selectionStart = selStart;
    config.selectionEnd = selEnd;

    // Effects settings
    config.enableColorGrade = m_colorGradeCheck->GetValue();
    if (config.enableColorGrade) {
        wxString preset = m_colorPresetChoice->GetStringSelection().Lower();
        config.colorPreset = preset;
    }
    config.enableVignette = m_vignetteCheck->GetValue();
    config.enableBeatFlash = m_beatFlashCheck->GetValue();
    config.enableBeatZoom = m_beatZoomCheck->GetValue();

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
#ifdef __WXUNIVERSAL__
    wxPaintDC dc(this);
    wxSize clientSize = GetClientSize();

    // Draw static background on frame
    if (m_backgroundBitmap.IsOk()) {
        // Draw background image at original size (don't scale - let it tile/crop naturally)
        dc.DrawBitmap(m_backgroundBitmap, 0, 0, false);

        // Fill any remaining area if window is larger than bitmap
        wxSize bmpSize = m_backgroundBitmap.GetSize();
        if (clientSize.x > bmpSize.x || clientSize.y > bmpSize.y) {
            dc.SetBrush(wxBrush(wxColour(10, 10, 26)));
            dc.SetPen(*wxTRANSPARENT_PEN);
            if (clientSize.x > bmpSize.x) {
                dc.DrawRectangle(bmpSize.x, 0, clientSize.x - bmpSize.x, clientSize.y);
            }
            if (clientSize.y > bmpSize.y) {
                dc.DrawRectangle(0, bmpSize.y, clientSize.x, clientSize.y - bmpSize.y);
            }
        }
    } else {
        // Fallback gradient (background image didn't load)
        dc.GradientFillLinear(GetClientRect(),
            wxColour(10, 10, 26), wxColour(25, 0, 50), wxSOUTH);
    }
#else
    // Native: Skip to allow default handling
    event.Skip();
#endif
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
                double beatTime = beatGrid.getBeats()[i];
                // Filter by selection range
                if (beatTime >= config.selectionStart && beatTime <= config.selectionEnd) {
                    filteredBeats.push_back(beatTime);
                }
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
                std::string folderPath = config.videoPath.ToStdString();

                // Debug log to file
                FILE* debugLog = fopen("tripsitter_debug.log", "a");
                if (debugLog) {
                    fprintf(debugLog, "\n=== Processing Started ===\n");
                    fprintf(debugLog, "Scanning folder for videos: %s\n", folderPath.c_str());
                    fclose(debugLog);
                }

                std::cout << "Scanning folder for videos: " << folderPath << "\n";

                try {
                    for (const auto& entry : fs::directory_iterator(folderPath)) {
                        if (entry.is_regular_file()) {
                            std::string ext = entry.path().extension().string();
                            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                            if (ext == ".mp4" || ext == ".avi" || ext == ".mov" ||
                                ext == ".mkv" || ext == ".webm") {
                                std::cout << "  Found video: " << entry.path().filename().string() << "\n";
                                videoFiles.push_back(entry.path().string());

                                // Debug log
                                debugLog = fopen("tripsitter_debug.log", "a");
                                if (debugLog) {
                                    fprintf(debugLog, "  Found video: %s\n", entry.path().string().c_str());
                                    fclose(debugLog);
                                }
                            }
                        }
                    }
                } catch (const std::exception& e) {
                    debugLog = fopen("tripsitter_debug.log", "a");
                    if (debugLog) {
                        fprintf(debugLog, "ERROR scanning folder: %s\n", e.what());
                        fclose(debugLog);
                    }
                    throw std::runtime_error(std::string("Error scanning folder: ") + e.what());
                }

                std::cout << "Total videos found: " << videoFiles.size() << "\n";

                debugLog = fopen("tripsitter_debug.log", "a");
                if (debugLog) {
                    fprintf(debugLog, "Total videos found: %zu\n", videoFiles.size());
                    fclose(debugLog);
                }

                if (videoFiles.empty()) {
                    throw std::runtime_error("No video files found in folder: " + folderPath);
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
            writer.setOutputSettings(width, height, config.fps);

            // Configure effects
            BeatSync::EffectsConfig effects;
            effects.enableColorGrade = config.enableColorGrade;
            effects.colorPreset = config.colorPreset.ToStdString();
            effects.enableVignette = config.enableVignette;
            effects.vignetteStrength = 0.5;
            effects.enableBeatFlash = config.enableBeatFlash;
            effects.enableBeatZoom = config.enableBeatZoom;
            effects.bpm = config.bpm;
            writer.setEffectsConfig(effects);

            bool hasEffects = config.enableColorGrade || config.enableVignette ||
                              config.enableBeatFlash || config.enableBeatZoom;

            FILE* debugLog = fopen("tripsitter_debug.log", "a");
            if (debugLog) {
                fprintf(debugLog, "Processing %zu beats (filtered from %zu total)\n", filteredBeats.size(), beatGrid.getBeats().size());
                fclose(debugLog);
            }

            if (filteredBeats.empty()) {
                throw std::runtime_error("No beats to process in selected range");
            }

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
                
                // Calculate segment duration based on beat timing
                double beatTime = filteredBeats[i];

                // Clamp segment end to selection end to avoid overrun
                double selectionEnd = config.selectionEnd;
                if (selectionEnd < 0 || selectionEnd > beatGrid.getAudioDuration()) {
                    selectionEnd = beatGrid.getAudioDuration();
                }
                double nextMark = (i + 1 < filteredBeats.size()) ? filteredBeats[i + 1] : selectionEnd;
                double segmentEnd = std::min(nextMark, selectionEnd);
                double duration = segmentEnd - beatTime;
                if (duration <= 0.0) {
                    continue;  // nothing to cut for this beat
                }

                std::string videoFile = videoFiles[i % videoFiles.size()];
                std::string segmentPath = "temp_segment_" + std::to_string(i) + ".mp4";

                // Extract from the BEGINNING of each video clip (0.0), not the beat timestamp
                // The beat timing determines duration, but each clip starts from 0
                if (!writer.copySegmentFast(videoFile, 0.0, duration, segmentPath)) {
                    // Fallback: try with small offset to avoid keyframe issues
                    if (!writer.copySegmentFast(videoFile, 0.1, duration, segmentPath)) {
                        // Cleanup temp files and report error to UI
                        for (const auto& s : segmentPaths) std::remove(s.c_str());
                        wxThreadEvent* errorEvt = new wxThreadEvent(wxEVT_PROCESSING_COMPLETE);
                        errorEvt->SetInt(0);
                        errorEvt->SetString(wxString::Format("Error extracting segment: %s", wxString(writer.getLastError())));
                        wxQueueEvent(this, errorEvt);
                        return;
                    }
                }

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
            
            if (!writer.concatenateVideos(segmentPaths, "temp_video.mp4")) {
                for (const auto& s : segmentPaths) std::remove(s.c_str());
                wxThreadEvent* errorEvt = new wxThreadEvent(wxEVT_PROCESSING_COMPLETE);
                errorEvt->SetInt(0);
                errorEvt->SetString(wxString::Format("Concatenation failed: %s (see beatsync_ffmpeg_concat.log)", wxString(writer.getLastError())));
                wxQueueEvent(this, errorEvt);
                return;
            }

            // Step 5.5: Apply effects (if any enabled)
            std::string videoForAudio = "temp_video.mp4";
            if (hasEffects) {
                evt = new wxThreadEvent(wxEVT_PROCESSING_PROGRESS);
                evt->SetInt(92);
                evt->SetString("Applying effects...");
                wxQueueEvent(this, evt);

                if (!writer.applyEffects("temp_video.mp4", "temp_video_fx.mp4")) {
                    for (const auto& s : segmentPaths) std::remove(s.c_str());
                    std::remove("temp_video.mp4");
                    wxThreadEvent* errorEvt = new wxThreadEvent(wxEVT_PROCESSING_COMPLETE);
                    errorEvt->SetInt(0);
                    errorEvt->SetString(wxString::Format("Effects processing failed: %s", wxString(writer.getLastError())));
                    wxQueueEvent(this, errorEvt);
                    return;
                }
                videoForAudio = "temp_video_fx.mp4";
            }

            // Step 6: Mux with audio
            evt = new wxThreadEvent(wxEVT_PROCESSING_PROGRESS);
            evt->SetInt(95);
            evt->SetString("Muxing audio...");
            wxQueueEvent(this, evt);

            if (!writer.addAudioTrack(videoForAudio, config.audioPath.ToStdString(),
                config.outputPath.ToStdString(), true,
                config.selectionStart, config.selectionEnd)) {
                for (const auto& s : segmentPaths) std::remove(s.c_str());
                std::remove("temp_video.mp4");
                if (hasEffects) std::remove("temp_video_fx.mp4");
                wxThreadEvent* errorEvt = new wxThreadEvent(wxEVT_PROCESSING_COMPLETE);
                errorEvt->SetInt(0);
                errorEvt->SetString(wxString::Format("Adding audio failed: %s", wxString(writer.getLastError())));
                wxQueueEvent(this, errorEvt);
                return;
            }

            // Fallback: ensure final output exists even if mux step produced nothing (e.g., odd path without extension).
            {
                const std::string finalOut = config.outputPath.ToStdString();
                std::filesystem::path outPath(finalOut);
                std::error_code ec;
                auto ensureParent = outPath.parent_path();
                if (!ensureParent.empty()) {
                    std::filesystem::create_directories(ensureParent, ec);
                }

                bool outMissing = !std::filesystem::exists(outPath, ec);
                bool outTiny = false;
                if (!outMissing) {
                    auto sz = std::filesystem::file_size(outPath, ec);
                    outTiny = (!ec && sz < 1024);
                }

                if (outMissing || outTiny) {
                    const char* src = hasEffects ? "temp_video_fx.mp4" : "temp_video.mp4";
                    std::filesystem::copy_file(src, outPath, std::filesystem::copy_options::overwrite_existing, ec);
                    if (ec) {
                        for (const auto& s : segmentPaths) std::remove(s.c_str());
                        std::remove("temp_video.mp4");
                        if (hasEffects) std::remove("temp_video_fx.mp4");
                        wxThreadEvent* errorEvt = new wxThreadEvent(wxEVT_PROCESSING_COMPLETE);
                        errorEvt->SetInt(0);
                        errorEvt->SetString(wxString::Format("Output copy failed: %s", wxString(ec.message())));
                        wxQueueEvent(this, errorEvt);
                        return;
                    }
                }
            }

            // Cleanup
            for (const auto& seg : segmentPaths) {
                std::remove(seg.c_str());
            }
            std::remove("temp_video.mp4");
            if (hasEffects) std::remove("temp_video_fx.mp4");
            
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
        // Reuse the full logs dialog for errors
        m_statusText->SetLabel(message);
        OnViewLogs(wxCommandEvent());
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

    // Add a Help hint with Logs access tooltip
    if (GetMenuBar()) {
        wxMenuItem* helpItem = GetMenuBar()->FindItem(ID_VIEW_LOGS);
        if (helpItem) helpItem->SetHelp("View FFmpeg logs and diagnostics collected during processing");
    }

}

void MainWindow::OnViewLogs(wxCommandEvent& WXUNUSED(event)) {
    const std::string logPath = "beatsync_ffmpeg_concat.log";

    std::ostringstream full;
    if (wxFileExists(logPath)) {
        std::ifstream fin(logPath);
        std::string line;
        while (std::getline(fin, line)) {
            full << line << "\n";
        }
    } else {
        full << "No logs found at: " << logPath << "\n";
    }

    // Add simple diagnostics
    full << "\n--- Diagnostics ---\n";

    // Check BEATSYNC_FFMPEG_PATH env var
    const char* envPath = std::getenv("BEATSYNC_FFMPEG_PATH");
    if (envPath && envPath[0] != '\0') {
        full << "FFmpeg Path (env): " << envPath << "\n";
    } else {
        // Try 'where ffmpeg' on Windows
        FILE* pipe = _popen("where ffmpeg 2>nul", "r");
        if (pipe) {
            char buf[512];
            if (fgets(buf, sizeof(buf), pipe) != nullptr) {
                full << "FFmpeg Path (where): " << buf;
            } else {
                full << "FFmpeg Path: not found in PATH\n";
            }
            _pclose(pipe);
        } else {
            full << "FFmpeg Path: (where check failed)\n";
        }
    }

    // Show the resolved fallback used by VideoWriter for clarity
    BeatSync::VideoWriter diagWriter;
    full << "FFmpeg Path (resolved): " << diagWriter.resolveFfmpegPath() << "\n";

    full << "wxWidgets Version: " << wxVERSION_STRING << "\n";

    // Show dialog with full log and actions
    wxDialog dlg(this, wxID_ANY, "Logs & Diagnostics", wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER);
    wxBoxSizer* top = new wxBoxSizer(wxVERTICAL);
    std::string fullStr = full.str();
    wxTextCtrl* logTex = new wxTextCtrl(&dlg, wxID_ANY, wxString(fullStr), wxDefaultPosition, wxSize(800, 360), wxTE_MULTILINE | wxTE_READONLY | wxTE_DONTWRAP);
    top->Add(logTex, 1, wxALL | wxEXPAND, 8);

    // Compression preference checkbox (persisted setting)
    bool preferDeflate = m_settingsManager ? m_settingsManager->GetBool("ZipUseDeflate", false) : false;
    wxCheckBox* deflateChk = new wxCheckBox(&dlg, wxID_ANY, "Use DEFLATE compression (requires zlib)");
    deflateChk->SetValue(preferDeflate);
    deflateChk->SetToolTip("If enabled, saved logs will be compressed with DEFLATE. (See note below - not implemented yet.)");
    top->Add(deflateChk, 0, wxLEFT | wxRIGHT | wxBOTTOM | wxEXPAND, 8);

    // Developer note explaining how to enable/implement DEFLATE later
    wxStaticText* deflateNote = new wxStaticText(&dlg, wxID_ANY,
        "Note: DEFLATE is not implemented. To enable it later, add zlib to the build (e.g. add `find_package(ZLIB REQUIRED)` to CMake and link the library), implement DEFLATE compression in `BeatSync::createZip` (\"src/utils/LogArchiver.cpp\"), and re-enable this option via the 'ZipUseDeflate' setting.");
    deflateNote->SetForegroundColour(wxColour(200, 200, 200));
    deflateNote->SetFont(wxFont(8, wxFONTFAMILY_SWISS, wxFONTSTYLE_ITALIC, wxFONTWEIGHT_NORMAL));
    top->Add(deflateNote, 0, wxLEFT | wxRIGHT | wxBOTTOM, 8);

    wxBoxSizer* btns = new wxBoxSizer(wxHORIZONTAL);
    wxButton* openBtn = new wxButton(&dlg, wxID_ANY, "Open Log");
    wxButton* copyBtn = new wxButton(&dlg, wxID_ANY, "Copy Log");
    wxButton* saveBtn = new wxButton(&dlg, wxID_ANY, "Save Logs...");
    wxButton* closeBtn = new wxButton(&dlg, wxID_CANCEL, "Close");
    btns->Add(openBtn, 0, wxALL, 4);
    btns->Add(copyBtn, 0, wxALL, 4);
    btns->Add(saveBtn, 0, wxALL, 4);
    btns->AddStretchSpacer(1);
    btns->Add(closeBtn, 0, wxALL, 4);
    top->Add(btns, 0, wxEXPAND | wxALL, 4);

    // Persist checkbox when toggled
    if (m_settingsManager) {
        deflateChk->Bind(wxEVT_CHECKBOX, [this, deflateChk](wxCommandEvent&) {
            m_settingsManager->SetBool("ZipUseDeflate", deflateChk->GetValue());
        });
    }

    dlg.SetSizerAndFit(top);

    openBtn->Bind(wxEVT_BUTTON, [this, logPath](wxCommandEvent&){ if (wxFileExists(logPath)) wxLaunchDefaultApplication(logPath); else wxMessageBox("Log not found", "Error", wxOK | wxICON_ERROR, this); });
    copyBtn->Bind(wxEVT_BUTTON, [this, fullStr, &dlg](wxCommandEvent&){ if (wxTheClipboard->Open()) { wxTheClipboard->SetData(new wxTextDataObject(wxString(fullStr))); wxTheClipboard->Close(); wxMessageBox("Log copied to clipboard", "Copied", wxOK | wxICON_INFORMATION, &dlg); } else { wxMessageBox("Could not open clipboard", "Error", wxOK | wxICON_ERROR, &dlg); } });

    saveBtn->Bind(wxEVT_BUTTON, [this, &dlg](wxCommandEvent&){
        // Ask where to save the zip
        wxFileDialog save(&dlg, "Save logs as", "", "beatsync-logs.zip", "ZIP archives (*.zip)|*.zip", wxFD_SAVE | wxFD_OVERWRITE_PROMPT);
        if (save.ShowModal() == wxID_CANCEL) return;
        wxString dest = save.GetPath();

        // Gather candidate logs
        std::vector<std::string> candidates = {"beatsync_ffmpeg_concat.log", "repro_copy.log", "repro_reencode.log", "beatsync.log"};
        std::vector<std::string> logs;
        for (auto& f : candidates) {
            if (wxFileExists(f)) logs.push_back(std::string(f));
        }

        if (logs.empty()) {
            wxMessageBox("No log files found to save.", "Save Logs", wxOK | wxICON_INFORMATION, &dlg);
            return;
        }

        // Check user's compression preference
        bool preferDeflate = m_settingsManager ? m_settingsManager->GetBool("ZipUseDeflate", false) : false;
        if (preferDeflate) {
            // Inform user DEFLATE not implemented yet, proceed with store method
            wxMessageBox("DEFLATE compression is not implemented yet. Logs will be saved using the fast store method.", "Save Logs", wxOK | wxICON_INFORMATION, &dlg);
        }

        // Use pure C++ zip creation (no external dependencies, store method)
        std::string destStr = std::string(dest.mb_str());
        std::string err;
        std::vector<std::string> logFiles;
        for (auto &s : logs) logFiles.push_back(s);
        if (!BeatSync::createZip(logFiles, destStr, err)) {
            wxMessageBox(wxString::Format("Failed to create zip: %s", wxString(err)), "Error", wxOK | wxICON_ERROR, &dlg);
            return;
        }
        wxMessageBox("Logs saved to: " + dest, "Save Logs", wxOK | wxICON_INFORMATION, &dlg);
    });

    dlg.ShowModal();
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

#ifdef __WXUNIVERSAL__
void MainWindow::SetAllChildrenTransparent(wxWindow* parent) {
    if (!parent) return;

    wxWindowList& children = parent->GetChildren();
    for (wxWindowList::iterator it = children.begin(); it != children.end(); ++it) {
        wxWindow* child = *it;

        // CRITICAL: Tell children NOT to paint their own backgrounds
        child->SetBackgroundStyle(wxBG_STYLE_TRANSPARENT);

        // Set foreground color to ensure text is visible
        child->SetForegroundColour(wxColour(200, 220, 255));

        // Recursively process all descendants
        SetAllChildrenTransparent(child);
    }
}
#endif
