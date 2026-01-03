#ifdef __WXUNIVERSAL__

#include "PsychedelicTheme.h"
#include <wx/settings.h>
#include <wx/dc.h>
#include <wx/dcclient.h>

namespace TripSitter {

// ============================================================================
// PsychedelicColourScheme
// ============================================================================

wxColour PsychedelicColourScheme::Get(StdColour col) const {
    switch (col) {
        // Window backgrounds
        case WINDOW:
        case CONTROL:
        case TITLEBAR:
            return PsychedelicColors::Background();

        // Control surfaces
        case CONTROL_PRESSED:
            return PsychedelicColors::Surface();

        case CONTROL_CURRENT:
            return PsychedelicColors::Primary();

        // Text colors
        case CONTROL_TEXT:
        case TITLEBAR_TEXT:
            return PsychedelicColors::Text();

        case CONTROL_TEXT_DISABLED:
            return wxColour(80, 80, 100);

        // Highlights and selections
        case HIGHLIGHT:
        case HIGHLIGHT_TEXT:
            return PsychedelicColors::Primary();

        // Scrollbar
        case SCROLLBAR:
        case SCROLLBAR_PRESSED:
            return PsychedelicColors::Surface();

        // Borders
        case SHADOW_DARK:
        case SHADOW_HIGHLIGHT:
        case SHADOW_IN:
        case SHADOW_OUT:
            return PsychedelicColors::Primary().ChangeLightness(50);

        // Gauge/Progress
        case GAUGE:
            return PsychedelicColors::Primary();

        default:
            return PsychedelicColors::Background();
    }
}

wxColour PsychedelicColourScheme::GetBackground(wxWindow* win) const {
    return PsychedelicColors::Background();
}

// ============================================================================
// PsychedelicRenderer
// ============================================================================

PsychedelicRenderer::PsychedelicRenderer(wxRenderer* renderer)
    : wxDelegateRenderer(renderer)
{
}

void PsychedelicRenderer::DrawBackground(wxDC& dc, const wxColour& col,
                                         const wxRect& rect, int flags,
                                         wxWindow* window) {
    // Use our dark background
    dc.SetBrush(wxBrush(PsychedelicColors::Background()));
    dc.SetPen(*wxTRANSPARENT_PEN);
    dc.DrawRectangle(rect);
}

void PsychedelicRenderer::DrawBorder(wxDC& dc, wxBorder border,
                                     const wxRect& rect, int flags,
                                     wxRect* rectIn) {
    wxColour borderColor = PsychedelicColors::Primary().ChangeLightness(60);

    if (flags & wxCONTROL_FOCUSED) {
        borderColor = PsychedelicColors::Primary();
    }

    dc.SetPen(wxPen(borderColor, 1));
    dc.SetBrush(*wxTRANSPARENT_BRUSH);
    dc.DrawRectangle(rect);

    if (rectIn) {
        *rectIn = rect;
        rectIn->Deflate(1);
    }
}

void PsychedelicRenderer::DrawGradientButton(wxDC& dc, const wxRect& rect,
                                             const wxColour& colTop,
                                             const wxColour& colBottom,
                                             int flags) {
    // Draw gradient background
    dc.GradientFillLinear(rect, colTop, colBottom, wxSOUTH);

    // Draw border
    wxColour borderCol = (flags & wxCONTROL_FOCUSED) ?
                         PsychedelicColors::Primary() :
                         PsychedelicColors::Primary().ChangeLightness(70);
    dc.SetPen(wxPen(borderCol, 1));
    dc.SetBrush(*wxTRANSPARENT_BRUSH);
    dc.DrawRoundedRectangle(rect, 3);
}

void PsychedelicRenderer::DrawGlowEffect(wxDC& dc, const wxRect& rect,
                                         const wxColour& glowColor) {
    // Simple glow effect by drawing slightly larger rectangle with alpha
    wxRect glowRect = rect;
    glowRect.Inflate(2);

    dc.SetPen(wxPen(glowColor.ChangeLightness(70), 2));
    dc.SetBrush(*wxTRANSPARENT_BRUSH);
    dc.DrawRoundedRectangle(glowRect, 4);
}

void PsychedelicRenderer::DrawButtonSurface(wxDC& dc, const wxColour& col,
                                            const wxRect& rect, int flags) {
    wxColour topCol, bottomCol;

    if (flags & wxCONTROL_PRESSED) {
        // Pressed state - darker, inverted gradient
        topCol = PsychedelicColors::Primary().ChangeLightness(60);
        bottomCol = PsychedelicColors::Primary().ChangeLightness(80);
    } else if (flags & wxCONTROL_CURRENT) {
        // Hover state - brighter
        topCol = PsychedelicColors::Primary().ChangeLightness(90);
        bottomCol = PsychedelicColors::Primary().ChangeLightness(70);
        DrawGlowEffect(dc, rect, PsychedelicColors::Primary());
    } else {
        // Normal state
        topCol = PsychedelicColors::Surface().ChangeLightness(120);
        bottomCol = PsychedelicColors::Surface();
    }

    DrawGradientButton(dc, rect, topCol, bottomCol, flags);
}

void PsychedelicRenderer::DrawButtonLabel(wxDC& dc, const wxString& label,
                                          const wxBitmap& image,
                                          const wxRect& rect, int flags,
                                          int alignment, int indexAccel,
                                          wxRect* rectBounds) {
    dc.SetTextForeground(
        (flags & wxCONTROL_DISABLED) ?
        wxColour(80, 80, 100) :
        PsychedelicColors::Text()
    );

    // Let the base class handle the actual drawing
    wxDelegateRenderer::DrawButtonLabel(dc, label, image, rect, flags,
                                        alignment, indexAccel, rectBounds);
}

void PsychedelicRenderer::DrawCheckButton(wxDC& dc, const wxString& label,
                                          const wxBitmap& bitmap,
                                          const wxRect& rect, int flags,
                                          wxAlignment align, int indexAccel) {
    // Draw custom checkbox
    wxRect boxRect(rect.x, rect.y + (rect.height - 16) / 2, 16, 16);

    // Background
    dc.SetBrush(wxBrush(PsychedelicColors::Surface()));
    dc.SetPen(wxPen(PsychedelicColors::Primary().ChangeLightness(70)));
    dc.DrawRoundedRectangle(boxRect, 2);

    // Checkmark if checked
    if (flags & wxCONTROL_CHECKED) {
        dc.SetPen(wxPen(PsychedelicColors::Primary(), 2));
        dc.DrawLine(boxRect.x + 3, boxRect.y + 8,
                   boxRect.x + 6, boxRect.y + 11);
        dc.DrawLine(boxRect.x + 6, boxRect.y + 11,
                   boxRect.x + 13, boxRect.y + 4);
    }

    // Label
    dc.SetTextForeground(PsychedelicColors::Text());
    wxRect labelRect = rect;
    labelRect.x += 22;
    labelRect.width -= 22;
    dc.DrawLabel(label, labelRect, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL);
}

void PsychedelicRenderer::DrawRadioButton(wxDC& dc, const wxString& label,
                                          const wxBitmap& bitmap,
                                          const wxRect& rect, int flags,
                                          wxAlignment align, int indexAccel) {
    // Draw custom radio button
    wxRect circleRect(rect.x, rect.y + (rect.height - 16) / 2, 16, 16);

    // Background circle
    dc.SetBrush(wxBrush(PsychedelicColors::Surface()));
    dc.SetPen(wxPen(PsychedelicColors::Primary().ChangeLightness(70)));
    dc.DrawEllipse(circleRect);

    // Inner dot if selected
    if (flags & wxCONTROL_CHECKED) {
        dc.SetBrush(wxBrush(PsychedelicColors::Primary()));
        dc.SetPen(*wxTRANSPARENT_PEN);
        wxRect innerRect = circleRect;
        innerRect.Deflate(4);
        dc.DrawEllipse(innerRect);
    }

    // Label
    dc.SetTextForeground(PsychedelicColors::Text());
    wxRect labelRect = rect;
    labelRect.x += 22;
    labelRect.width -= 22;
    dc.DrawLabel(label, labelRect, wxALIGN_LEFT | wxALIGN_CENTER_VERTICAL);
}

void PsychedelicRenderer::DrawTextLine(wxDC& dc, const wxString& text,
                                       const wxRect& rect, int selStart,
                                       int selEnd, int flags) {
    dc.SetTextForeground(PsychedelicColors::Text());
    dc.SetTextBackground(PsychedelicColors::Surface());

    wxDelegateRenderer::DrawTextLine(dc, text, rect, selStart, selEnd, flags);
}

void PsychedelicRenderer::DrawScrollbarThumb(wxDC& dc, wxOrientation orient,
                                             const wxRect& rect, int flags) {
    wxColour thumbCol = (flags & wxCONTROL_CURRENT) ?
                        PsychedelicColors::Primary() :
                        PsychedelicColors::Primary().ChangeLightness(60);

    dc.SetBrush(wxBrush(thumbCol));
    dc.SetPen(*wxTRANSPARENT_PEN);
    dc.DrawRoundedRectangle(rect, 3);
}

void PsychedelicRenderer::DrawScrollbarShaft(wxDC& dc, wxOrientation orient,
                                             const wxRect& rect, int flags) {
    dc.SetBrush(wxBrush(PsychedelicColors::Surface()));
    dc.SetPen(*wxTRANSPARENT_PEN);
    dc.DrawRectangle(rect);
}

// ============================================================================
// PsychedelicTheme
// ============================================================================

PsychedelicTheme::PsychedelicTheme()
    : wxDelegateTheme("win32")  // Inherit from win32 theme
    , m_renderer(nullptr)
    , m_colourScheme(nullptr)
{
}

PsychedelicTheme::~PsychedelicTheme() {
    delete m_renderer;
    delete m_colourScheme;
}

wxRenderer* PsychedelicTheme::GetRenderer() {
    if (!m_renderer) {
        // Get the base renderer and wrap it
        wxRenderer* baseRenderer = wxDelegateTheme::GetRenderer();
        m_renderer = new PsychedelicRenderer(baseRenderer);
    }
    return m_renderer;
}

wxColourScheme* PsychedelicTheme::GetColourScheme() {
    if (!m_colourScheme) {
        m_colourScheme = new PsychedelicColourScheme();
    }
    return m_colourScheme;
}

} // namespace TripSitter

#endif // __WXUNIVERSAL__
