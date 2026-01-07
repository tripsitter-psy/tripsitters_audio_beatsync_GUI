#include <wx/wx.h>
#include <wx/image.h>
#include "GUI/MainWindow.h"
#include <fstream>
#include <string>
#include <stdexcept>

// SEH translator, minidump on crash
#ifdef _WIN32
#include <windows.h>
#include <eh.h>
#include <dbghelp.h>
#include <sstream>
#include <iomanip>
#pragma comment(lib, "Dbghelp.lib")

static std::string to_hex(unsigned int code) {
    std::ostringstream ss;
    ss << std::hex << std::showbase << code;
    return ss.str();
}

static void sehTranslator(unsigned int code, EXCEPTION_POINTERS* ep) {
    // Try to write a minidump for offline analysis
    HANDLE hFile = CreateFileA("tripsitter_crash.dmp", GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile != INVALID_HANDLE_VALUE) {
        MINIDUMP_EXCEPTION_INFORMATION mei;
        mei.ThreadId = GetCurrentThreadId();
        mei.ExceptionPointers = ep;
        mei.ClientPointers = FALSE;
        MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), hFile, MiniDumpNormal, &mei, NULL, NULL);
        CloseHandle(hFile);
    }

    throw std::runtime_error(std::string("SEH exception: ") + to_hex(code));
}
#endif

#ifdef __WXUNIVERSAL__
#include <wx/univ/theme.h>
#include "GUI/PsychedelicTheme.h"
#endif

#include "tracing/Tracing.h"

class TripSitterApp : public wxApp {
public:
    virtual bool OnInit() override {
    // Set SEH translator so crashes produce a minidump and become C++ exceptions we can catch.
#ifdef _WIN32
    _set_se_translator(sehTranslator);
#endif
    // Initialize image handlers for PNG, JPEG, etc.
    wxInitAllImageHandlers();

#ifdef __WXUNIVERSAL__
    // Set the psychedelic theme for wxUniversal builds
    wxTheme::Set(new TripSitter::PsychedelicTheme());
#endif

    SetAppName("MTV Trip Sitter");
    SetVendorName("MTV Trip Sitter");

    // Initialize tracing for GUI app
    Tracing::Init("tripsitter");

        try {
            MainWindow* frame = new MainWindow();
            frame->Show(true);
        } catch (const std::exception& ex) {
            // Write exception and minidump location to log (no UI popups)
            std::ofstream fout("tripsitter_exception.log", std::ios::app);
            fout << "Exception: " << ex.what() << std::endl;
            fout << "If a crash occurred, check tripsitter_crash.dmp in the working directory." << std::endl;
            fout.close();
            Tracing::Shutdown();
            return false;
        } catch (...) {
            std::ofstream fout("tripsitter_exception.log", std::ios::app);
            fout << "Unknown exception during MainWindow creation" << std::endl;
            fout << "If a crash occurred, check tripsitter_crash.dmp in the working directory." << std::endl;
            fout.close();
            Tracing::Shutdown();
            return false;
        }
        return true;
    }

    virtual int OnExit() override {
        // Ensure tracing is cleanly shut down on exit
        Tracing::Shutdown();
        return wxApp::OnExit();
    }
};

wxIMPLEMENT_APP(TripSitterApp);
