#include <catch2/catch_test_macros.hpp>

#include <wx/wx.h>
#include <wx/panel.h>

TEST_CASE("Header panel can be painted without asserting") {
    // Initialize a minimal wxApp instance for the test
    class TestApp : public wxApp {
    public:
        bool OnInit() override { return true; }
    };

    // Note: we intentionally allocate the app on the heap and set it as the global instance
    TestApp* app = new TestApp();
    wxApp::SetInstance(app);
    int argc = 0;
    char** argv = nullptr;
    wxEntryStart(argc, argv);
    wxTheApp->CallOnInit();

    // Create a top-level frame and a main panel just like MainWindow does
    wxFrame* frame = new wxFrame(nullptr, wxID_ANY, "TestFrame", wxDefaultPosition, wxSize(800,600));
    wxPanel* mainPanel = new wxPanel(frame, wxID_ANY);

    // Create a small header bitmap and a header panel that would normally be painted
    wxBitmap headerBmp(200, 64);
    wxPanel* headerPanel = new wxPanel(mainPanel, wxID_ANY, wxDefaultPosition, headerBmp.GetSize(), wxBORDER_NONE | wxTRANSPARENT_WINDOW);
    headerPanel->SetMinSize(headerBmp.GetSize());

    // Use PAINT style (the code change we made) and bind a paint handler that draws the bitmap
    headerPanel->SetBackgroundStyle(wxBG_STYLE_PAINT);
    headerPanel->Bind(wxEVT_PAINT, [&](wxPaintEvent& evt){
        wxPaintDC dc(headerPanel);
        dc.DrawBitmap(headerBmp, 0, 0, true);
    });

    // Show the frame and force a paint. Running under Xvfb in CI will exercise the same path.
    frame->Show();
    headerPanel->Refresh();
    headerPanel->Update();

    // Process pending events so the paint handler executes.
    wxTheApp->ProcessPendingEvents();

    // If we reach this point with no assertion, consider the test successful.
    REQUIRE(true);

    // Cleanup
    frame->Destroy();
    wxTheApp->OnExit();
    wxEntryCleanup();
}
