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
    EVT_MOUSEWHEEL(BeatVisualizer::OnMouseWheel)
    EVT_RIGHT_DOWN(BeatVisualizer::OnRightDown)
    EVT_MIDDLE_DOWN(BeatVisualizer::OnMiddleDown)
    EVT_MIDDLE_UP(BeatVisualizer::OnMiddleUp)
    EVT_MENU(BeatVisualizer::ID_SET_EFFECT_IN, BeatVisualizer::OnSetEffectIn)
    EVT_MENU(BeatVisualizer::ID_SET_EFFECT_OUT, BeatVisualizer::OnSetEffectOut)
    EVT_MENU(BeatVisualizer::ID_CLEAR_EFFECT_REGION, BeatVisualizer::OnClearEffectRegion)
    EVT_MENU(BeatVisualizer::ID_ZOOM_IN, BeatVisualizer::OnZoomIn)
    EVT_MENU(BeatVisualizer::ID_ZOOM_OUT, BeatVisualizer::OnZoomOut)
    EVT_MENU(BeatVisualizer::ID_ZOOM_FIT, BeatVisualizer::OnZoomToFit)
wxEND_EVENT_TABLE()

BeatVisualizer::BeatVisualizer(wxWindow* parent, wxWindowID id,
    const wxPoint& pos, const wxSize& size)
    : wxPanel(parent, id, pos, size)
{
    // Semi-transparent dark background
    SetBackgroundColour(wxColour(10, 10, 26, 200));
    SetBackgroundStyle(wxBG_STYLE_PAINT);
    m_showEffectBeatOverlay = false;
    m_effectBeatDivisor = 1;
    // Add right-click menu for debug overlay
    Bind(wxEVT_CONTEXT_MENU, [this](wxContextMenuEvent& evt) {
        wxMenu menu;
        menu.AppendCheckItem(10001, "Show Effect Beat Overlay");
        menu.Check(10001, m_showEffectBeatOverlay);
        menu.Bind(wxEVT_MENU, [this](wxCommandEvent& e) {
            m_showEffectBeatOverlay = !m_showEffectBeatOverlay;
            Refresh();
        }, 10001);
        PopupMenu(&menu);
    });
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
    m_effectStart = -1.0;
    m_effectEnd = -1.0;
    m_zoomLevel = 1.0;
    m_scrollOffset = 0.0;
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

// Effect region methods
void BeatVisualizer::SetEffectRegion(double start, double end) {
    m_effectStart = std::max(0.0, start);
    m_effectEnd = (end < 0) ? m_audioDuration : std::min(end, m_audioDuration);
    Refresh();
}

std::pair<double, double> BeatVisualizer::GetEffectRegion() const {
    if (!HasEffectRegion()) {
        return {0.0, m_audioDuration};  // Default to full track
    }
    return {m_effectStart, m_effectEnd};
}

void BeatVisualizer::ClearEffectRegion() {
    m_effectStart = -1.0;
    m_effectEnd = -1.0;
    Refresh();
}

// Zoom methods
void BeatVisualizer::ZoomIn() {
    SetZoomLevel(m_zoomLevel * 1.5);
}

void BeatVisualizer::ZoomOut() {
    SetZoomLevel(m_zoomLevel / 1.5);
}

void BeatVisualizer::ZoomToFit() {
    m_zoomLevel = 1.0;
    m_scrollOffset = 0.0;
    Refresh();
}

void BeatVisualizer::SetZoomLevel(double level) {
    double oldZoom = m_zoomLevel;
    m_zoomLevel = std::max(MIN_ZOOM, std::min(MAX_ZOOM, level));
    
    // Adjust scroll to keep center of view stable
    if (m_zoomLevel != oldZoom) {
        double viewWidth = 1.0 / m_zoomLevel;
        double centerTime = m_scrollOffset + (1.0 / oldZoom) / 2.0;
        m_scrollOffset = centerTime - viewWidth / 2.0;
        
        // Clamp scroll offset
        double maxScroll = 1.0 - viewWidth;
        m_scrollOffset = std::max(0.0, std::min(maxScroll, m_scrollOffset));
    }
    Refresh();
}

double BeatVisualizer::PixelToTime(int x) const {
    wxSize sz = GetClientSize();
    int trackWidth = sz.x - 2 * MARGIN;
    if (trackWidth <= 0 || m_audioDuration <= 0) return 0.0;

    // Convert pixel to normalized position (0-1) within visible area
    double visibleRatio = static_cast<double>(x - MARGIN) / trackWidth;
    visibleRatio = std::max(0.0, std::min(1.0, visibleRatio));
    
    // Convert to actual time considering zoom and scroll
    double viewWidth = 1.0 / m_zoomLevel;  // Fraction of track visible
    double normalizedTime = m_scrollOffset + visibleRatio * viewWidth;
    
    return normalizedTime * m_audioDuration;
}

int BeatVisualizer::TimeToPixel(double time) const {
    wxSize sz = GetClientSize();
    int trackWidth = sz.x - 2 * MARGIN;
    if (m_audioDuration <= 0) return MARGIN;

    // Convert time to normalized position (0-1)
    double normalizedTime = time / m_audioDuration;
    
    // Convert to visible area considering zoom and scroll
    double viewWidth = 1.0 / m_zoomLevel;
    double visibleRatio = (normalizedTime - m_scrollOffset) / viewWidth;
    
    return MARGIN + static_cast<int>(visibleRatio * trackWidth);
}

int BeatVisualizer::GetHandleAtPos(int x) const {
    if (m_audioDuration <= 0) return 0;

    int leftX = TimeToPixel(m_selectionStart);
    int rightX = TimeToPixel(m_selectionEnd < 0 ? m_audioDuration : m_selectionEnd);

    // Check effect region handles first (they take priority for easier access)
    if (HasEffectRegion()) {
        int effectInX = TimeToPixel(m_effectStart);
        int effectOutX = TimeToPixel(m_effectEnd);
        
        if (std::abs(x - effectInX) <= HANDLE_WIDTH) return 3;  // Effect in
        if (std::abs(x - effectOutX) <= HANDLE_WIDTH) return 4; // Effect out
    }

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
        if (HasCapture()) ReleaseMouse();

        // Ensure start < end for selection
        if (m_draggingHandle <= 2) {
            if (m_selectionEnd >= 0 && m_selectionStart > m_selectionEnd) {
                std::swap(m_selectionStart, m_selectionEnd);
            }
        }
        // Ensure start < end for effect region
        else if (m_draggingHandle >= 3) {
            if (m_effectEnd >= 0 && m_effectStart > m_effectEnd) {
                std::swap(m_effectStart, m_effectEnd);
            }
        }
        
        m_draggingHandle = 0;
        Refresh();
    }
    event.Skip();
}

void BeatVisualizer::OnMotion(wxMouseEvent& event) {
    // Handle panning (middle mouse button)
    if (m_isPanning && event.MiddleIsDown()) {
        wxSize sz = GetClientSize();
        int trackWidth = sz.x - 2 * MARGIN;
        if (trackWidth > 0 && m_zoomLevel > 1.0) {
            double deltaPixels = event.GetX() - m_panStartX;
            double viewWidth = 1.0 / m_zoomLevel;
            double deltaNormalized = -deltaPixels / trackWidth * viewWidth;
            
            m_scrollOffset = m_panStartOffset + deltaNormalized;
            double maxScroll = 1.0 - viewWidth;
            m_scrollOffset = std::max(0.0, std::min(maxScroll, m_scrollOffset));
            Refresh();
        }
        return;
    }

    if (m_draggingHandle > 0 && event.LeftIsDown()) {
        double time = PixelToTime(event.GetX());
        time = std::max(0.0, std::min(m_audioDuration, time));

        if (m_draggingHandle == 1) {
            m_selectionStart = time;
        } else if (m_draggingHandle == 2) {
            m_selectionEnd = time;
        } else if (m_draggingHandle == 3) {
            m_effectStart = time;
        } else if (m_draggingHandle == 4) {
            m_effectEnd = time;
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

void BeatVisualizer::OnMouseWheel(wxMouseEvent& event) {
    if (m_audioDuration <= 0) return;
    
    // Get time at mouse position before zoom
    double timeAtMouse = PixelToTime(event.GetX());
    double normalizedTimeAtMouse = timeAtMouse / m_audioDuration;
    
    // Calculate new zoom level
    double zoomFactor = (event.GetWheelRotation() > 0) ? 1.2 : (1.0 / 1.2);
    double newZoom = std::max(MIN_ZOOM, std::min(MAX_ZOOM, m_zoomLevel * zoomFactor));
    
    if (newZoom != m_zoomLevel) {
        // Calculate pixel position of mouse within track
        wxSize sz = GetClientSize();
        int trackWidth = sz.x - 2 * MARGIN;
        double pixelRatio = static_cast<double>(event.GetX() - MARGIN) / trackWidth;
        pixelRatio = std::max(0.0, std::min(1.0, pixelRatio));
        
        // Update zoom
        m_zoomLevel = newZoom;
        
        // Adjust scroll so the time at mouse stays under the mouse
        double newViewWidth = 1.0 / m_zoomLevel;
        m_scrollOffset = normalizedTimeAtMouse - pixelRatio * newViewWidth;
        
        // Clamp scroll offset
        double maxScroll = 1.0 - newViewWidth;
        m_scrollOffset = std::max(0.0, std::min(maxScroll, m_scrollOffset));
        
        Refresh();
    }
}

void BeatVisualizer::OnRightDown(wxMouseEvent& event) {
    if (m_audioDuration <= 0) return;
    
    m_contextMenuTime = PixelToTime(event.GetX());
    m_contextMenuTime = std::max(0.0, std::min(m_audioDuration, m_contextMenuTime));
    
    wxMenu menu;
    
    // Effect markers submenu
    wxString timeStr = wxString::Format("%.2fs", m_contextMenuTime);
    menu.Append(ID_SET_EFFECT_IN, wxString::Format("Set Effect IN at %s", timeStr));
    menu.Append(ID_SET_EFFECT_OUT, wxString::Format("Set Effect OUT at %s", timeStr));
    
    if (HasEffectRegion()) {
        menu.AppendSeparator();
        menu.Append(ID_CLEAR_EFFECT_REGION, "Clear Effect Region");
    }
    
    menu.AppendSeparator();
    menu.Append(ID_ZOOM_IN, "Zoom In\tCtrl+Plus");
    menu.Append(ID_ZOOM_OUT, "Zoom Out\tCtrl+Minus");
    menu.Append(ID_ZOOM_FIT, "Zoom to Fit\tCtrl+0");
    
    PopupMenu(&menu, event.GetPosition());
}

void BeatVisualizer::OnMiddleDown(wxMouseEvent& event) {
    if (m_zoomLevel > 1.0) {
        m_isPanning = true;
        m_panStartX = event.GetX();
        m_panStartOffset = m_scrollOffset;
        CaptureMouse();
        SetCursor(wxCursor(wxCURSOR_HAND));
    }
}

void BeatVisualizer::OnMiddleUp(wxMouseEvent& event) {
    if (m_isPanning) {
        m_isPanning = false;
        if (HasCapture()) ReleaseMouse();
        SetCursor(wxNullCursor);
    }
}

// Context menu handlers
void BeatVisualizer::OnSetEffectIn(wxCommandEvent& event) {
    m_effectStart = m_contextMenuTime;
    // If no effect end set, default to end of track
    if (m_effectEnd < 0 || m_effectEnd <= m_effectStart) {
        m_effectEnd = m_audioDuration;
    }
    Refresh();
}

void BeatVisualizer::OnSetEffectOut(wxCommandEvent& event) {
    m_effectEnd = m_contextMenuTime;
    // If no effect start set, default to beginning
    if (m_effectStart < 0 || m_effectStart >= m_effectEnd) {
        m_effectStart = 0.0;
    }
    Refresh();
}

void BeatVisualizer::OnClearEffectRegion(wxCommandEvent& event) {
    ClearEffectRegion();
}

void BeatVisualizer::OnZoomIn(wxCommandEvent& event) {
    ZoomIn();
}

void BeatVisualizer::OnZoomOut(wxCommandEvent& event) {
    ZoomOut();
}

void BeatVisualizer::OnZoomToFit(wxCommandEvent& event) {
    ZoomToFit();
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

    // Calculate visible time range
    double viewWidth = 1.0 / m_zoomLevel;
    double visibleStartNorm = m_scrollOffset;
    double visibleEndNorm = m_scrollOffset + viewWidth;
    double visibleStartTime = visibleStartNorm * m_audioDuration;
    double visibleEndTime = visibleEndNorm * m_audioDuration;

    // Draw selection region (shaded background)
    double selEnd = (m_selectionEnd < 0) ? m_audioDuration : m_selectionEnd;
    int selStartX = TimeToPixel(m_selectionStart);
    int selEndX = TimeToPixel(selEnd);

    // Clamp to visible area
    selStartX = std::max(MARGIN, std::min(MARGIN + trackWidth, selStartX));
    selEndX = std::max(MARGIN, std::min(MARGIN + trackWidth, selEndX));

    if (selEndX > selStartX) {
        dc.SetBrush(wxBrush(wxColour(0, 50, 80)));
        dc.SetPen(*wxTRANSPARENT_PEN);
        dc.DrawRectangle(selStartX, 20, selEndX - selStartX, sz.y - 40);
    }

    // Draw effect region (purple shading, more prominent)
    if (HasEffectRegion()) {
        int effectStartX = TimeToPixel(m_effectStart);
        int effectEndX = TimeToPixel(m_effectEnd);
        
        // Clamp to visible area
        effectStartX = std::max(MARGIN, std::min(MARGIN + trackWidth, effectStartX));
        effectEndX = std::max(MARGIN, std::min(MARGIN + trackWidth, effectEndX));
        
        if (effectEndX > effectStartX) {
            dc.SetBrush(wxBrush(wxColour(80, 0, 80, 100)));  // Purple overlay
            dc.SetPen(*wxTRANSPARENT_PEN);
            dc.DrawRectangle(effectStartX, centerY - 45, effectEndX - effectStartX, 90);
        }
    }

    // Draw waveform (accounting for zoom/scroll)
    if (!m_waveformData.empty() && trackWidth > 0) {
        int numSamples = static_cast<int>(m_waveformData.size());

        for (int x = 0; x < trackWidth; ++x) {
            // Convert pixel to time considering zoom/scroll
            double pixelRatio = static_cast<double>(x) / trackWidth;
            double normalizedTime = visibleStartNorm + pixelRatio * viewWidth;
            double timeAtX = normalizedTime * m_audioDuration;
            
            // Map to waveform sample
            int sampleIdx = static_cast<int>(normalizedTime * numSamples);
            sampleIdx = std::max(0, std::min(numSamples - 1, sampleIdx));

            float amplitude = m_waveformData[sampleIdx];
            int barHeight = static_cast<int>(amplitude * waveformHeight / 2);

            // Determine coloring based on regions
            bool inSelection = (timeAtX >= m_selectionStart && timeAtX <= selEnd);
            bool inEffect = HasEffectRegion() && (timeAtX >= m_effectStart && timeAtX <= m_effectEnd);

            // Color based on region
            if (inEffect) {
                // Magenta/purple for effect region
                int r = static_cast<int>(180 + 75 * amplitude);
                int g = static_cast<int>(50 * (1.0 - amplitude));
                int b = static_cast<int>(220 + 35 * amplitude);
                dc.SetPen(wxPen(wxColour(r, g, b)));
            } else if (inSelection) {
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

    // Draw beat markers (all detected beats)
    int beatSkip = 1;
    if (m_zoomLevel < 2.0) {
        beatSkip = (m_beatTimes.size() > 200) ? 4 : ((m_beatTimes.size() > 100) ? 2 : 1);
    }
    for (size_t i = 0; i < m_beatTimes.size(); i += beatSkip) {
        double t = m_beatTimes[i];
        if (t < visibleStartTime - 0.1 || t > visibleEndTime + 0.1) continue;
        int x = TimeToPixel(t);
        if (x < MARGIN || x > MARGIN + trackWidth) continue;
        dc.SetPen(wxPen(wxColour(255, 255, 100), 2));
        dc.SetBrush(wxBrush(wxColour(255, 255, 100, 180)));
        dc.DrawRectangle(x - 1, centerY - 35, 2, 70);
    }

    // Draw filtered effect beats overlay (red lines)
    if (m_showEffectBeatOverlay) {
        // Compute filtered beats (same logic as VideoWriter)
        int effectDivisor = m_effectBeatDivisor;
        double effectStart = HasEffectRegion() ? m_effectStart : 0.0;
        double effectEnd = HasEffectRegion() ? m_effectEnd : m_audioDuration;
        for (size_t i = 0; i < m_beatTimes.size(); ++i) {
            if (effectDivisor > 1 && (i % effectDivisor) != 0) continue;
            double t = m_beatTimes[i];
            if (t < effectStart || t > effectEnd) continue;
            if (t < visibleStartTime - 0.1 || t > visibleEndTime + 0.1) continue;
            int x = TimeToPixel(t);
            if (x < MARGIN || x > MARGIN + trackWidth) continue;
            dc.SetPen(wxPen(wxColour(255, 0, 0), 3));
            dc.DrawLine(x, centerY - 40, x, centerY + 40);
        }
    }

    // Draw selection handles (green = start, orange = end)
    if (selStartX >= MARGIN && selStartX <= MARGIN + trackWidth) {
        dc.SetPen(wxPen(wxColour(0, 255, 100), 2));
        dc.SetBrush(wxBrush(wxColour(0, 255, 100)));
        dc.DrawLine(selStartX, 15, selStartX, sz.y - 15);
        wxPoint leftTriangle[3] = {
            {selStartX, 10},
            {selStartX + 10, 20},
            {selStartX - 10, 20}
        };
        dc.DrawPolygon(3, leftTriangle);
    }

    if (selEndX >= MARGIN && selEndX <= MARGIN + trackWidth) {
        dc.SetPen(wxPen(wxColour(255, 100, 0), 2));
        dc.SetBrush(wxBrush(wxColour(255, 100, 0)));
        dc.DrawLine(selEndX, 15, selEndX, sz.y - 15);
        wxPoint rightTriangle[3] = {
            {selEndX, 10},
            {selEndX + 10, 20},
            {selEndX - 10, 20}
        };
        dc.DrawPolygon(3, rightTriangle);
    }

    // Draw effect region handles (magenta/purple diamonds)
    if (HasEffectRegion()) {
        int effectInX = TimeToPixel(m_effectStart);
        int effectOutX = TimeToPixel(m_effectEnd);
        
        if (effectInX >= MARGIN && effectInX <= MARGIN + trackWidth) {
            dc.SetPen(wxPen(wxColour(200, 50, 200), 2));
            dc.SetBrush(wxBrush(wxColour(200, 50, 200)));
            dc.DrawLine(effectInX, 25, effectInX, sz.y - 25);
            // Diamond shape for effect markers
            wxPoint inDiamond[4] = {
                {effectInX, sz.y - 30},
                {effectInX + 8, sz.y - 22},
                {effectInX, sz.y - 14},
                {effectInX - 8, sz.y - 22}
            };
            dc.DrawPolygon(4, inDiamond);
        }
        
        if (effectOutX >= MARGIN && effectOutX <= MARGIN + trackWidth) {
            dc.SetPen(wxPen(wxColour(200, 50, 200), 2));
            dc.SetBrush(wxBrush(wxColour(200, 50, 200)));
            dc.DrawLine(effectOutX, 25, effectOutX, sz.y - 25);
            wxPoint outDiamond[4] = {
                {effectOutX, sz.y - 30},
                {effectOutX + 8, sz.y - 22},
                {effectOutX, sz.y - 14},
                {effectOutX - 8, sz.y - 22}
            };
            dc.DrawPolygon(4, outDiamond);
        }
    }

    // Draw time labels for selection handles
    int tw, th;
    if (selStartX >= MARGIN - 20 && selStartX <= MARGIN + trackWidth + 20) {
        dc.SetTextForeground(wxColour(0, 255, 100));
        wxString startStr = wxString::Format("%.1fs", m_selectionStart);
        dc.DrawText(startStr, std::max(MARGIN, selStartX + 5), sz.y - 18);
    }

    if (selEndX >= MARGIN - 20 && selEndX <= MARGIN + trackWidth + 20) {
        dc.SetTextForeground(wxColour(255, 100, 0));
        wxString endStr = wxString::Format("%.1fs", selEnd);
        dc.GetTextExtent(endStr, &tw, &th);
        dc.DrawText(endStr, std::min(MARGIN + trackWidth - tw, selEndX - tw - 5), sz.y - 18);
    }

    // Draw effect region labels
    if (HasEffectRegion()) {
        dc.SetTextForeground(wxColour(200, 100, 200));
        wxString effectStr = wxString::Format("Effects: %.1fs - %.1fs", m_effectStart, m_effectEnd);
        dc.GetTextExtent(effectStr, &tw, &th);
        dc.DrawText(effectStr, MARGIN + 5, sz.y - 32);
    }

    // Draw header info (selection + zoom level)
    dc.SetTextForeground(wxColour(200, 200, 200));
    wxString durStr = wxString::Format("Selection: %.1fs - %.1fs", m_selectionStart, selEnd);
    if (m_zoomLevel > 1.01) {
        durStr += wxString::Format("  |  Zoom: %.1fx  (Scroll wheel to zoom, middle-click to pan)", m_zoomLevel);
    } else {
        durStr += "  |  Scroll wheel to zoom";
    }
    dc.GetTextExtent(durStr, &tw, &th);
    dc.DrawText(durStr, (sz.x - tw) / 2, 2);
}
