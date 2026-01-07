#pragma once
#include <wx/wx.h>
#include <wx/menu.h>
#include <vector>
#include <utility>

class BeatVisualizer : public wxPanel {
public:
    // Set/get effect beat divisor for overlay
    void SetEffectBeatDivisor(int divisor) { m_effectBeatDivisor = divisor; Refresh(); }
    int GetEffectBeatDivisor() const { return m_effectBeatDivisor; }
    BeatVisualizer(wxWindow* parent, wxWindowID id = wxID_ANY,
        const wxPoint& pos = wxDefaultPosition,
        const wxSize& size = wxDefaultSize);

    void LoadAudio(const wxString& audioPath);
    void GenerateWaveform(const wxString& audioPath);
    void Clear();

    // Selection range (in seconds) - for trimming audio
    void SetSelectionRange(double start, double end);
    std::pair<double, double> GetSelectionRange() const;
    double GetAudioDuration() const { return m_audioDuration; }

    // Effect region (in seconds) - for effect in/out points
    void SetEffectRegion(double start, double end);
    std::pair<double, double> GetEffectRegion() const;
    bool HasEffectRegion() const { return m_effectStart >= 0 && m_effectEnd > m_effectStart; }
    void ClearEffectRegion();

    // Zoom control
    void ZoomIn();
    void ZoomOut();
    void ZoomToFit();
    void SetZoomLevel(double level);
    double GetZoomLevel() const { return m_zoomLevel; }

public:
    // Debug overlay toggle
    void SetShowEffectBeatOverlay(bool show) { m_showEffectBeatOverlay = show; Refresh(); }
    bool GetShowEffectBeatOverlay() const { return m_showEffectBeatOverlay; }

private:
    int m_effectBeatDivisor = 1;
    void OnPaint(wxPaintEvent& event);
    void OnSize(wxSizeEvent& event);
    void OnLeftDown(wxMouseEvent& event);
    void OnLeftUp(wxMouseEvent& event);
    void OnMotion(wxMouseEvent& event);
    void OnMouseWheel(wxMouseEvent& event);
    void OnRightDown(wxMouseEvent& event);
    void OnMiddleDown(wxMouseEvent& event);
    void OnMiddleUp(wxMouseEvent& event);

    // Context menu handlers
    void OnSetEffectIn(wxCommandEvent& event);
    void OnSetEffectOut(wxCommandEvent& event);
    void OnClearEffectRegion(wxCommandEvent& event);
    void OnZoomIn(wxCommandEvent& event);
    void OnZoomOut(wxCommandEvent& event);
    void OnZoomToFit(wxCommandEvent& event);

    // Convert between pixel X position and time (accounting for zoom/scroll)
    double PixelToTime(int x) const;
    int TimeToPixel(double time) const;
    int GetHandleAtPos(int x) const;  // Returns 0=none, 1=left, 2=right, 3=effect-in, 4=effect-out

    std::vector<double> m_beatTimes;
    std::vector<float> m_waveform;
    std::vector<float> m_waveformData;
    double m_audioDuration = 0.0;

    // Selection state (for audio trimming)
    double m_selectionStart = 0.0;
    double m_selectionEnd = -1.0;  // -1 means full track
    int m_draggingHandle = 0;      // 0=none, 1=left, 2=right, 3=effect-in, 4=effect-out

    // Effect region (for effects in/out points)
    double m_effectStart = -1.0;   // -1 means no effect region set
    double m_effectEnd = -1.0;

    // Zoom and scroll state
    double m_zoomLevel = 1.0;      // 1.0 = fit to view, >1 = zoomed in
    double m_scrollOffset = 0.0;   // Scroll position (0.0 to 1.0 - viewWidth)
    bool m_isPanning = false;
    int m_panStartX = 0;
    double m_panStartOffset = 0.0;

    // Context menu position (in time)
    double m_contextMenuTime = 0.0;

    // Layout constants
    static constexpr int MARGIN = 10;
    static constexpr int HANDLE_WIDTH = 8;
    static constexpr double MIN_ZOOM = 1.0;
    static constexpr double MAX_ZOOM = 50.0;

    // Menu IDs
    enum {
        ID_SET_EFFECT_IN = wxID_HIGHEST + 100,
        ID_SET_EFFECT_OUT,
        ID_CLEAR_EFFECT_REGION,
        ID_ZOOM_IN,
        ID_ZOOM_OUT,
        ID_ZOOM_FIT
    };

    bool m_showEffectBeatOverlay = false;
    wxDECLARE_EVENT_TABLE();
};
