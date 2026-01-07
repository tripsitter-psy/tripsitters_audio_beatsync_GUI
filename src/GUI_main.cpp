#include <wx/wx.h>
#include <wx/image.h>
#include <fstream>
#include <ctime>
#include "GUI/MainWindow.h"

// Very early startup marker: runs during static initialization before main/OnInit.
// This helps determine whether the process begins executing at all on the user's machine.
static int WriteEarlyStartupMarker() {
    try {
        std::ofstream dbg("tripsitter_debug.log", std::ios::app);
        dbg << "EARLY_START_MARKER: pid=" << (unsigned)GetCurrentProcessId() << " ts=" << std::time(nullptr) << std::endl;
        dbg.flush();
    } catch(...) {
        // best-effort only
    }
    return 0;
}
static int g_earlyStartupMarker = WriteEarlyStartupMarker();

#ifdef __WXUNIVERSAL__
#include <wx/univ/theme.h>
#include "GUI/PsychedelicTheme.h"
#endif

#ifdef _WIN32
#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")

static void LogStack(const CONTEXT* ctx = nullptr) {
    HANDLE proc = GetCurrentProcess();
    SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME);
    // Use Microsoft symbol server for better symbol resolution
    SymSetSearchPath(proc, "srv*");
    SymInitialize(proc, NULL, TRUE);

    void* stack[62];
    USHORT frames = 0;
    if (ctx) {
        // If context provided, use StackWalk (complex); fallback to CaptureStackBackTrace
        frames = CaptureStackBackTrace(0, _countof(stack), stack, NULL);
    } else {
        frames = CaptureStackBackTrace(0, _countof(stack), stack, NULL);
    }

    std::ofstream dbg("tripsitter_debug.log", std::ios::app);
    dbg << "=== Crash stack (frames=" << frames << ") ===" << std::endl;
    for (USHORT i = 0; i < frames; ++i) {
        DWORD64 addr = (DWORD64)(stack[i]);
        char symbolBuffer[sizeof(SYMBOL_INFO) + 256];
        PSYMBOL_INFO pSymbol = (PSYMBOL_INFO)symbolBuffer;
        pSymbol->SizeOfStruct = sizeof(SYMBOL_INFO);
        pSymbol->MaxNameLen = 255;
        DWORD64 displacement = 0;
        if (SymFromAddr(proc, addr, &displacement, pSymbol)) {
            dbg << i << ": " << pSymbol->Name << " + 0x" << std::hex << displacement << std::dec << " (0x" << std::hex << addr << std::dec << ")" << std::endl;
        } else {
            dbg << i << ": 0x" << std::hex << addr << std::dec << std::endl;
        }
    }
    dbg << "=== End stack ===" << std::endl;
}

static LONG WINAPI TripSitterCrashHandler(EXCEPTION_POINTERS* ex) {
    std::ofstream dbg("tripsitter_debug.log", std::ios::app);
    dbg << "Unhandled exception code: 0x" << std::hex << ex->ExceptionRecord->ExceptionCode << std::dec << std::endl;
    LogStack(ex->ContextRecord);
    dbg.flush();
    // Let default handler create crash dump if needed
    return EXCEPTION_EXECUTE_HANDLER;
}
#endif

class TripSitterApp : public wxApp {
public:
    virtual bool OnInit() override {
#ifdef _WIN32
        SetUnhandledExceptionFilter(TripSitterCrashHandler);
#endif
        // Initialize image handlers for PNG, JPEG, etc.
        wxInitAllImageHandlers();

        // Write an OnInit marker file so we can detect that OnInit ran even when there's no console
        try {
            const char* tmp = std::getenv("TMP");
            if (!tmp) tmp = std::getenv("TEMP");
            std::string tempPath = tmp ? tmp : ".";
            std::string p = tempPath + "\\tripsitter_marker_OnInit_start.txt";
            std::ofstream m(p, std::ios::out);
            m << "OnInit_start pid=" << (unsigned)GetCurrentProcessId() << " ts=" << std::time(nullptr) << std::endl;
            m.flush();
        } catch(...) {}


#ifdef __WXUNIVERSAL__
        // Set the psychedelic theme for wxUniversal builds
        wxTheme::Set(new TripSitter::PsychedelicTheme());
#endif

        SetAppName("MTV Trip Sitter");
        SetVendorName("MTV Trip Sitter");

        MainWindow* frame = new MainWindow();
        {
            try {
                const char* tmp = std::getenv("TMP");
                if (!tmp) tmp = std::getenv("TEMP");
                std::string tempPath = tmp ? tmp : ".";
                std::string p = tempPath + "\\tripsitter_marker_MainWindow_created.txt";
                std::ofstream m(p, std::ios::out);
                m << "MainWindow_created pid=" << (unsigned)GetCurrentProcessId() << " ts=" << std::time(nullptr) << std::endl;
                m.flush();
            } catch(...) {}
        }
        frame->Show(true);
        try {
            const char* tmp = std::getenv("TMP");
            if (!tmp) tmp = std::getenv("TEMP");
            std::string tempPath = tmp ? tmp : ".";
            std::string p = tempPath + "\\tripsitter_marker_MainWindow_shown.txt";
            std::ofstream m(p, std::ios::out);
            m << "MainWindow_shown pid=" << (unsigned)GetCurrentProcessId() << " ts=" << std::time(nullptr) << std::endl;
            m.flush();
        } catch(...) {}

        return true;
    }
};

wxIMPLEMENT_APP(TripSitterApp);
