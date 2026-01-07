#include <wx/wx.h>
#include <wx/image.h>
#include <fstream>
#include "GUI/MainWindow.h"

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
