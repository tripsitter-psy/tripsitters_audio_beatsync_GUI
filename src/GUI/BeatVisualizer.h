#pragma once
#include <wx/wx.h>
#include <vector>

class BeatVisualizer : public wxPanel {
public:
    BeatVisualizer(wxWindow* parent, wxWindowID id = wxID_ANY,
        const wxPoint& pos = wxDefaultPosition,
        const wxSize& size = wxDefaultSize);
    
    void LoadAudio(const wxString& audioPath);
    void Clear();
    
private:
    void OnPaint(wxPaintEvent& event);
    void OnSize(wxSizeEvent& event);
    
    std::vector<double> m_beatTimes;
    std::vector<float> m_waveform;
    double m_audioDuration;
    
    wxDECLARE_EVENT_TABLE();
};
