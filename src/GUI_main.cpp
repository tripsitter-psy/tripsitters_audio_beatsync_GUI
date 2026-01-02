#include <wx/wx.h>
#include <wx/image.h>
#include "gui/MainWindow.h"

class TripSitterApp : public wxApp {
public:
    virtual bool OnInit() override {
        // Initialize image handlers for PNG, JPEG, etc.
        wxInitAllImageHandlers();

        SetAppName("TripSitter");
        SetVendorName("TripSitter");

        MainWindow* frame = new MainWindow();
        frame->Show(true);

        return true;
    }
};

wxIMPLEMENT_APP(TripSitterApp);
