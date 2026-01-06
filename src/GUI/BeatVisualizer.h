#pragma once
#include <wx/wx.h>
#include <vector>
#include <utility>

class BeatVisualizer : public wxPanel {
public:
    BeatVisualizer(wxWindow* parent, wxWindowID id = wxID_ANY,
        const wxPoint& pos = wxDefaultPosition,
        const wxSize& size = wxDefaultSize);

    void LoadAudio(const wxString& audioPath);
    void GenerateWaveform(const wxString& audioPath);
    void Clear();

    // Selection range (in seconds)
    void SetSelectionRange(double start, double end);
    std::pair<double, double> GetSelectionRange() const;
    double GetAudioDuration() const { return m_audioDuration; }

private:
    void OnPaint(wxPaintEvent& event);
    void OnSize(wxSizeEvent& event);
    void OnLeftDown(wxMouseEvent& event);
    void OnLeftUp(wxMouseEvent& event);
    void OnMotion(wxMouseEvent& event);

    // Convert between pixel X position and time
    double PixelToTime(int x) const;
    int TimeToPixel(double time) const;
    int GetHandleAtPos(int x) const;  // Returns 0=none, 1=left, 2=right

    std::vector<double> m_beatTimes;
    std::vector<float> m_waveform;
    std::vector<float> m_waveformData;
    double m_audioDuration = 0.0;

    // Selection state
    double m_selectionStart = 0.0;
    double m_selectionEnd = -1.0;  // -1 means full track
    int m_draggingHandle = 0;      // 0=none, 1=left, 2=right

    // Layout constants
    static constexpr int MARGIN = 10;
    static constexpr int HANDLE_WIDTH = 8;

    wxDECLARE_EVENT_TABLE();
};
