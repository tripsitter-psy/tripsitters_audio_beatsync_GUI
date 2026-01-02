#include "BeatVisualizer.h"
#include <wx/dcbuffer.h>
#include "../audio/AudioAnalyzer.h"

wxBEGIN_EVENT_TABLE(BeatVisualizer, wxPanel)
    EVT_PAINT(BeatVisualizer::OnPaint)
    EVT_SIZE(BeatVisualizer::OnSize)
wxEND_EVENT_TABLE()

BeatVisualizer::BeatVisualizer(wxWindow* parent, wxWindowID id,
    const wxPoint& pos, const wxSize& size)
    : wxPanel(parent, id, pos, size)
    , m_audioDuration(0.0)
{
    SetBackgroundColour(*wxBLACK);
}

void BeatVisualizer::LoadAudio(const wxString& audioPath) {
    try {
        BeatSync::AudioAnalyzer analyzer;
        BeatSync::BeatGrid grid = analyzer.analyze(audioPath.ToStdString());
        m_beatTimes.clear();
        const auto& beats = grid.getBeats();
        m_beatTimes.reserve(beats.size());
        for (double t : beats) m_beatTimes.push_back(t);
        m_audioDuration = grid.getAudioDuration();
        Refresh();
    } catch (...) {
        m_beatTimes.clear();
        m_audioDuration = 0.0;
        Refresh();
    }
}

void BeatVisualizer::Clear() {
    m_beatTimes.clear();
    m_audioDuration = 0.0;
    Refresh();
}

void BeatVisualizer::OnSize(wxSizeEvent& event) {
    Refresh();
    event.Skip();
}

void BeatVisualizer::OnPaint(wxPaintEvent& event) {
    wxAutoBufferedPaintDC dc(this);
    wxSize sz = GetClientSize();

    dc.SetBackground(*wxBLACK_BRUSH);
    dc.Clear();

    if (m_beatTimes.empty() || m_audioDuration <= 0.0) {
        dc.SetTextForeground(wxColour(120, 120, 120));
        dc.DrawLabel("Load audio to see beat visualization", wxRect(0,0,sz.x,sz.y), wxALIGN_CENTER);
        return;
    }

    // Draw timeline background
    int margin = 10;
    int barY = sz.y / 2;
    int barH = 6;
    wxRect barRect(margin, barY - barH/2, sz.x - 2*margin, barH);
    dc.SetBrush(wxBrush(wxColour(50,50,50)));
    dc.SetPen(*wxTRANSPARENT_PEN);
    dc.DrawRectangle(barRect);

    // Draw beats as small markers
    dc.SetPen(wxPen(wxColour(0,217,255), 2));
    for (double t : m_beatTimes) {
        double pos = t / m_audioDuration;
        if (pos < 0.0) pos = 0.0;
        if (pos > 1.0) pos = 1.0;
        int x = margin + static_cast<int>(pos * (sz.x - 2*margin));
        dc.DrawLine(x, barY - 10, x, barY + 10);
    }

    // Draw time labels: 0s and duration
    dc.SetTextForeground(wxColour(180,180,180));
    dc.DrawText("0s", margin, barY + 12);
    wxString durStr = wxString::Format("%.2fs", m_audioDuration);
    int tw, th; dc.GetTextExtent(durStr, &tw, &th);
    dc.DrawText(durStr, sz.x - margin - tw, barY + 12);
}
