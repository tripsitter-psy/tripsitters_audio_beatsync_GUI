#include "BeatVisualizer.h"
#include <wx/dcbuffer.h>
#include "../audio/AudioAnalyzer.h"

wxBEGIN_EVENT_TABLE(BeatVisualizer, wxPanel)
    EVT_PAINT(BeatVisualizer::OnPaint)
    EVT_SIZE(BeatVisualizer::OnSize)
    EVT_LEFT_DOWN(BeatVisualizer::OnLeftDown)
    EVT_LEFT_UP(BeatVisualizer::OnLeftUp)
    EVT_MOTION(BeatVisualizer::OnMotion)
wxEND_EVENT_TABLE()

BeatVisualizer::BeatVisualizer(wxWindow* parent, wxWindowID id,
    const wxPoint& pos, const wxSize& size)
    : wxPanel(parent, id, pos, size)
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

        // Reset selection to full track
        m_selectionStart = 0.0;
        m_selectionEnd = m_audioDuration;

        Refresh();
    } catch (...) {
        m_beatTimes.clear();
        m_audioDuration = 0.0;
        m_selectionStart = 0.0;
        m_selectionEnd = -1.0;
        Refresh();
    }
}

void BeatVisualizer::Clear() {
    m_beatTimes.clear();
    m_audioDuration = 0.0;
    m_selectionStart = 0.0;
    m_selectionEnd = -1.0;
    Refresh();
}

void BeatVisualizer::SetSelectionRange(double start, double end) {
    m_selectionStart = std::max(0.0, start);
    m_selectionEnd = (end < 0) ? m_audioDuration : std::min(end, m_audioDuration);
    Refresh();
}

std::pair<double, double> BeatVisualizer::GetSelectionRange() const {
    double endTime = (m_selectionEnd < 0) ? m_audioDuration : m_selectionEnd;
    return {m_selectionStart, endTime};
}

double BeatVisualizer::PixelToTime(int x) const {
    wxSize sz = GetClientSize();
    int trackWidth = sz.x - 2 * MARGIN;
    if (trackWidth <= 0 || m_audioDuration <= 0) return 0.0;

    double ratio = static_cast<double>(x - MARGIN) / trackWidth;
    ratio = std::max(0.0, std::min(1.0, ratio));
    return ratio * m_audioDuration;
}

int BeatVisualizer::TimeToPixel(double time) const {
    wxSize sz = GetClientSize();
    int trackWidth = sz.x - 2 * MARGIN;
    if (m_audioDuration <= 0) return MARGIN;

    double ratio = time / m_audioDuration;
    ratio = std::max(0.0, std::min(1.0, ratio));
    return MARGIN + static_cast<int>(ratio * trackWidth);
}

int BeatVisualizer::GetHandleAtPos(int x) const {
    if (m_audioDuration <= 0) return 0;

    int leftX = TimeToPixel(m_selectionStart);
    int rightX = TimeToPixel(m_selectionEnd < 0 ? m_audioDuration : m_selectionEnd);

    if (std::abs(x - leftX) <= HANDLE_WIDTH) return 1;
    if (std::abs(x - rightX) <= HANDLE_WIDTH) return 2;
    return 0;
}

void BeatVisualizer::OnLeftDown(wxMouseEvent& event) {
    if (m_audioDuration <= 0) return;

    int handle = GetHandleAtPos(event.GetX());
    if (handle > 0) {
        m_draggingHandle = handle;
        CaptureMouse();
    }
    event.Skip();
}

void BeatVisualizer::OnLeftUp(wxMouseEvent& event) {
    if (m_draggingHandle > 0) {
        m_draggingHandle = 0;
        if (HasCapture()) ReleaseMouse();

        // Ensure start < end
        if (m_selectionEnd >= 0 && m_selectionStart > m_selectionEnd) {
            std::swap(m_selectionStart, m_selectionEnd);
        }
        Refresh();
    }
    event.Skip();
}

void BeatVisualizer::OnMotion(wxMouseEvent& event) {
    if (m_draggingHandle > 0 && event.LeftIsDown()) {
        double time = PixelToTime(event.GetX());

        if (m_draggingHandle == 1) {
            m_selectionStart = time;
        } else if (m_draggingHandle == 2) {
            m_selectionEnd = time;
        }
        Refresh();
    } else {
        // Change cursor when hovering over handles
        int handle = GetHandleAtPos(event.GetX());
        if (handle > 0) {
            SetCursor(wxCursor(wxCURSOR_SIZEWE));
        } else {
            SetCursor(wxNullCursor);
        }
    }
    event.Skip();
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

    int barY = sz.y / 2;
    int barH = 6;
    int trackWidth = sz.x - 2 * MARGIN;

    // Draw timeline background
    wxRect barRect(MARGIN, barY - barH/2, trackWidth, barH);
    dc.SetBrush(wxBrush(wxColour(50,50,50)));
    dc.SetPen(*wxTRANSPARENT_PEN);
    dc.DrawRectangle(barRect);

    // Draw selection region (shaded)
    double selEnd = (m_selectionEnd < 0) ? m_audioDuration : m_selectionEnd;
    int selStartX = TimeToPixel(m_selectionStart);
    int selEndX = TimeToPixel(selEnd);

    if (selEndX > selStartX) {
        dc.SetBrush(wxBrush(wxColour(0, 217, 255, 40)));  // Semi-transparent cyan
        dc.SetPen(*wxTRANSPARENT_PEN);
        dc.DrawRectangle(selStartX, barY - 20, selEndX - selStartX, 40);
    }

    // Draw beats as small markers
    dc.SetPen(wxPen(wxColour(0,217,255), 2));
    for (double t : m_beatTimes) {
        double pos = t / m_audioDuration;
        if (pos < 0.0) pos = 0.0;
        if (pos > 1.0) pos = 1.0;
        int x = MARGIN + static_cast<int>(pos * trackWidth);

        // Dim beats outside selection
        if (t < m_selectionStart || t > selEnd) {
            dc.SetPen(wxPen(wxColour(0, 80, 100), 1));
        } else {
            dc.SetPen(wxPen(wxColour(0, 217, 255), 2));
        }
        dc.DrawLine(x, barY - 10, x, barY + 10);
    }

    // Draw left handle (green)
    dc.SetPen(wxPen(wxColour(0, 255, 100), 2));
    dc.SetBrush(wxBrush(wxColour(0, 255, 100)));
    dc.DrawLine(selStartX, barY - 25, selStartX, barY + 25);
    wxPoint leftTriangle[3] = {
        {selStartX, barY - 25},
        {selStartX + 8, barY - 20},
        {selStartX + 8, barY - 30}
    };
    dc.DrawPolygon(3, leftTriangle);

    // Draw right handle (red/orange)
    dc.SetPen(wxPen(wxColour(255, 100, 0), 2));
    dc.SetBrush(wxBrush(wxColour(255, 100, 0)));
    dc.DrawLine(selEndX, barY - 25, selEndX, barY + 25);
    wxPoint rightTriangle[3] = {
        {selEndX, barY - 25},
        {selEndX - 8, barY - 20},
        {selEndX - 8, barY - 30}
    };
    dc.DrawPolygon(3, rightTriangle);

    // Draw time labels at handles
    dc.SetTextForeground(wxColour(0, 255, 100));
    wxString startStr = wxString::Format("%.1fs", m_selectionStart);
    dc.DrawText(startStr, selStartX + 2, barY + 28);

    dc.SetTextForeground(wxColour(255, 100, 0));
    wxString endStr = wxString::Format("%.1fs", selEnd);
    int tw, th;
    dc.GetTextExtent(endStr, &tw, &th);
    dc.DrawText(endStr, selEndX - tw - 2, barY + 28);

    // Draw duration label
    dc.SetTextForeground(wxColour(120, 120, 120));
    wxString durStr = wxString::Format("Duration: %.1fs", selEnd - m_selectionStart);
    dc.GetTextExtent(durStr, &tw, &th);
    dc.DrawText(durStr, (sz.x - tw) / 2, sz.y - th - 5);
}
