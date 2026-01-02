#include <wx/wx.h>
#include "GUI/MainWindow.h"

class TripSitterApp : public wxApp {
public:
    virtual bool OnInit() override {
        SetAppName("TripSitter");
        SetVendorName("TripSitter");
        
        MainWindow* frame = new MainWindow();
        frame->Show(true);
        
        return true;
    }
};

wxIMPLEMENT_APP(TripSitterApp);
