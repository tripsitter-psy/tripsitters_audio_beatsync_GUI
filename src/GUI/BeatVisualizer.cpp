#include "BeatVisualizer.h"
#include <wx/dcbuffer.h>
#include "../audio/AudioAnalyzer.h"
#include <algorithm>
#include <cmath>

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
    // Semi-transparent dark background
    SetBackgroundColour(wxColour(10, 10, 26, 200));
    SetBackgroundStyle(wxBG_STYLE_PAINT);
}

void BeatVisualizer::GenerateWaveform(const wxString& audioPath) {
    m_waveformData.clear();

    // Use AudioAnalyzer to get audio samples
    try {
        BeatSync::AudioAnalyzer analyzer;
        auto audioData = analyzer.loadAudioFile(audioPath.ToStdString());

        if (audioData.samples.empty()) {
            return;
        }

        // Target ~2000 samples for visualization
        const int targetSamples = 2000;
        size_t totalSamples = audioData.samples.size();
        size_t samplesPerChunk = std::max(size_t(1), totalSamples / targetSamples);

        m_waveformData.reserve(targetSamples);

        for (size_t i = 0; i < totalSamples; i += samplesPerChunk) {
            // Find peak amplitude in this chunk
            float peak = 0.0f;
            size_t end = std::min(i + samplesPerChunk, totalSamples);
            for (size_t j = i; j < end; ++j) {
                peak = std::max(peak, std::abs(audioData.samples[j]));
            }
            m_waveformData.push_back(peak);
        }
    } catch (const std::exception&) {
        // Failed to load waveform - visualization will be empty
        m_waveformData.clear();
    }
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

        // Generate waveform data
        GenerateWaveform(audioPath);

        // Reset selection to full track
        m_selectionStart = 0.0;
        m_selectionEnd = m_audioDuration;

        Refresh();
    } catch (...) {
        m_beatTimes.clear();
        m_waveformData.clear();
        m_audioDuration = 0.0;
        m_selectionStart = 0.0;
        m_selectionEnd = -1.0;
        Refresh();
    }
}

void BeatVisualizer::Clear() {
    m_beatTimes.clear();
    m_waveformData.clear();
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

    // Make background transparent so parent background shows through
    dc.SetBackground(wxBrush(wxColour(0, 0, 0, 0)));
    dc.Clear();

    if (m_audioDuration <= 0.0) {
        dc.SetTextForeground(wxColour(120, 120, 120));
        dc.DrawLabel("Load audio to see beat visualization", wxRect(0,0,sz.x,sz.y), wxALIGN_CENTER);
        return;
    }

    int centerY = sz.y / 2;
    int trackWidth = sz.x - 2 * MARGIN;
    int waveformHeight = sz.y - 80;  // Leave room for handles and labels

    // Draw selection region (shaded background)
    double selEnd = (m_selectionEnd < 0) ? m_audioDuration : m_selectionEnd;
    int selStartX = TimeToPixel(m_selectionStart);
    int selEndX = TimeToPixel(selEnd);

    if (selEndX > selStartX) {
        dc.SetBrush(wxBrush(wxColour(0, 50, 80)));
        dc.SetPen(*wxTRANSPARENT_PEN);
        dc.DrawRectangle(selStartX, 20, selEndX - selStartX, sz.y - 40);
    }

    // Draw waveform
    if (!m_waveformData.empty() && trackWidth > 0) {
        int numSamples = static_cast<int>(m_waveformData.size());

        for (int x = 0; x < trackWidth; ++x) {
            // Map pixel to waveform sample
            int sampleIdx = (x * numSamples) / trackWidth;
            sampleIdx = std::min(sampleIdx, numSamples - 1);

            float amplitude = m_waveformData[sampleIdx];
            int barHeight = static_cast<int>(amplitude * waveformHeight / 2);

            // Determine if this position is within selection
            double timeAtX = (static_cast<double>(x) / trackWidth) * m_audioDuration;
            bool inSelection = (timeAtX >= m_selectionStart && timeAtX <= selEnd);

            // Color based on selection
            if (inSelection) {
                // Gradient from cyan to purple
                int r = static_cast<int>(0 + (139 - 0) * amplitude);
                int g = static_cast<int>(217 - 217 * amplitude);
                int b = 255;
                dc.SetPen(wxPen(wxColour(r, g, b)));
            } else {
                dc.SetPen(wxPen(wxColour(40, 60, 80)));
            }

            int drawX = MARGIN + x;
            dc.DrawLine(drawX, centerY - barHeight, drawX, centerY + barHeight);
        }
    }

    // Draw beat markers - only show every 2nd or 4th beat to reduce clutter
    int beatSkip = (m_beatTimes.size() > 200) ? 4 : ((m_beatTimes.size() > 100) ? 2 : 1);

    for (size_t i = 0; i < m_beatTimes.size(); i += beatSkip) {
        double t = m_beatTimes[i];
        double pos = t / m_audioDuration;
        if (pos < 0.0 || pos > 1.0) continue;
        int x = MARGIN + static_cast<int>(pos * trackWidth);

        // Dim beats outside selection
        if (t < m_selectionStart || t > selEnd) {
            dc.SetPen(wxPen(wxColour(80, 80, 100), 1));
            dc.SetBrush(wxBrush(wxColour(80, 80, 100, 128)));
        } else {
            dc.SetPen(wxPen(wxColour(255, 255, 100), 3));  // Yellow for visibility
            dc.SetBrush(wxBrush(wxColour(255, 255, 100, 180)));
        }

        // Draw as a filled rectangle for better visibility
        dc.DrawRectangle(x - 1, centerY - 35, 3, 70);
    }

    // Draw left handle (green)
    dc.SetPen(wxPen(wxColour(0, 255, 100), 2));
    dc.SetBrush(wxBrush(wxColour(0, 255, 100)));
    dc.DrawLine(selStartX, 15, selStartX, sz.y - 15);
    wxPoint leftTriangle[3] = {
        {selStartX, 10},
        {selStartX + 10, 20},
        {selStartX - 10, 20}
    };
    dc.DrawPolygon(3, leftTriangle);

    // Draw right handle (red/orange)
    dc.SetPen(wxPen(wxColour(255, 100, 0), 2));
    dc.SetBrush(wxBrush(wxColour(255, 100, 0)));
    dc.DrawLine(selEndX, 15, selEndX, sz.y - 15);
    wxPoint rightTriangle[3] = {
        {selEndX, 10},
        {selEndX + 10, 20},
        {selEndX - 10, 20}
    };
    dc.DrawPolygon(3, rightTriangle);

    // Draw time labels
    dc.SetTextForeground(wxColour(0, 255, 100));
    wxString startStr = wxString::Format("%.1fs", m_selectionStart);
    dc.DrawText(startStr, selStartX + 5, sz.y - 18);

    dc.SetTextForeground(wxColour(255, 100, 0));
    wxString endStr = wxString::Format("%.1fs", selEnd);
    int tw, th;
    dc.GetTextExtent(endStr, &tw, &th);
    dc.DrawText(endStr, selEndX - tw - 5, sz.y - 18);

    // Draw duration label at top
    dc.SetTextForeground(wxColour(200, 200, 200));
    wxString durStr = wxString::Format("Selection: %.1fs (%.1fs - %.1fs)",
                                        selEnd - m_selectionStart, m_selectionStart, selEnd);
    dc.GetTextExtent(durStr, &tw, &th);
    dc.DrawText(durStr, (sz.x - tw) / 2, 2);
}
