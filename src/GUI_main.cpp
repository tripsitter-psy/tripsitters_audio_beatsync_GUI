#include <wx/wx.h>
#include <wx/image.h>
#include "gui/MainWindow.h"

#ifdef __WXUNIVERSAL__
#include <wx/univ/theme.h>
#include "gui/PsychedelicTheme.h"
#endif

class TripSitterApp : public wxApp {
public:
    virtual bool OnInit() override {
        // Initialize image handlers for PNG, JPEG, etc.
        wxInitAllImageHandlers();

#ifdef __WXUNIVERSAL__
        // Set the psychedelic theme for wxUniversal builds
        wxTheme::Set(new TripSitter::PsychedelicTheme());
#endif

        SetAppName("MTV Trip Sitter");
        SetVendorName("MTV Trip Sitter");

        MainWindow* frame = new MainWindow();
        frame->Show(true);

        return true;
    }
};

wxIMPLEMENT_APP(TripSitterApp);
