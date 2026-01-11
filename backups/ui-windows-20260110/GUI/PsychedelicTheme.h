#pragma once

#ifdef __WXUNIVERSAL__

#include <wx/univ/theme.h>
#include <wx/univ/renderer.h>
#include <wx/univ/colschem.h>
#include <wx/univ/inphand.h>

namespace TripSitter {

// Color palette for psychedelic theme
struct PsychedelicColors {
    static wxColour Primary()     { return wxColour(0, 217, 255); }    // Neon Cyan
    static wxColour Secondary()   { return wxColour(139, 0, 255); }    // Neon Purple
    static wxColour Background()  { return wxColour(10, 10, 26); }     // Dark Blue-Black
    static wxColour Surface()     { return wxColour(20, 20, 40); }     // Dark Gray-Blue
    static wxColour Text()        { return wxColour(200, 220, 255); }  // Light Blue-White
    static wxColour Accent()      { return wxColour(255, 0, 128); }    // Hot Pink
    static wxColour Warning()     { return wxColour(255, 136, 0); }    // Orange
    static wxColour Success()     { return wxColour(0, 255, 100); }    // Neon Green
};

/**
 * Custom colour scheme for the psychedelic theme
 */
class PsychedelicColourScheme : public wxColourScheme {
public:
    virtual wxColour Get(StdColour col) const override;
    virtual wxColour GetBackground(wxWindow* win) const override;
};

/**
 * Custom renderer that draws controls with psychedelic styling
 */
class PsychedelicRenderer : public wxDelegateRenderer {
public:
    PsychedelicRenderer(wxRenderer* renderer);

    // Background and borders
    virtual void DrawBackground(wxDC& dc, const wxColour& col,
                                const wxRect& rect, int flags,
                                wxWindow* window = nullptr) override;

    virtual void DrawBorder(wxDC& dc, wxBorder border,
                           const wxRect& rect, int flags = 0,
                           wxRect* rectIn = nullptr) override;

    // Button rendering
    virtual void DrawButtonSurface(wxDC& dc, const wxColour& col,
                                   const wxRect& rect, int flags) override;

    virtual void DrawButtonLabel(wxDC& dc, const wxString& label,
                                const wxBitmap& image, const wxRect& rect,
                                int flags, int alignment,
                                int indexAccel, wxRect* rectBounds) override;

    // Checkbox and radio button
    virtual void DrawCheckButton(wxDC& dc, const wxString& label,
                                const wxBitmap& bitmap, const wxRect& rect,
                                int flags, wxAlignment align = wxALIGN_LEFT,
                                int indexAccel = -1) override;

    virtual void DrawRadioButton(wxDC& dc, const wxString& label,
                                const wxBitmap& bitmap, const wxRect& rect,
                                int flags, wxAlignment align = wxALIGN_LEFT,
                                int indexAccel = -1) override;

    // Text control
    virtual void DrawTextLine(wxDC& dc, const wxString& text,
                             const wxRect& rect, int selStart, int selEnd,
                             int flags) override;

    // Scrollbar
    virtual void DrawScrollbarThumb(wxDC& dc, wxOrientation orient,
                                   const wxRect& rect, int flags) override;

    virtual void DrawScrollbarShaft(wxDC& dc, wxOrientation orient,
                                   const wxRect& rect, int flags) override;

private:
    void DrawGradientButton(wxDC& dc, const wxRect& rect,
                           const wxColour& colTop, const wxColour& colBottom,
                           int flags);
    void DrawGlowEffect(wxDC& dc, const wxRect& rect, const wxColour& glowColor);
};

/**
 * The main psychedelic theme class
 */
class PsychedelicTheme : public wxDelegateTheme {
public:
    PsychedelicTheme();
    virtual ~PsychedelicTheme();

    virtual wxRenderer* GetRenderer() override;
    virtual wxColourScheme* GetColourScheme() override;

private:
    PsychedelicRenderer* m_renderer;
    PsychedelicColourScheme* m_colourScheme;
};

} // namespace TripSitter

#endif // __WXUNIVERSAL__
