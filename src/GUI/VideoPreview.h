#pragma once
#include <wx/wx.h>

class VideoPreview : public wxPanel {
public:
    VideoPreview(wxWindow* parent, wxWindowID id = wxID_ANY,
        const wxPoint& pos = wxDefaultPosition,
        const wxSize& size = wxDefaultSize);
    
    void LoadFrame(const wxString& videoPath, double timestamp);
    void Clear();
    
private:
    void OnPaint(wxPaintEvent& event);
    
    wxBitmap m_frameBitmap;
    
    wxDECLARE_EVENT_TABLE();
};
