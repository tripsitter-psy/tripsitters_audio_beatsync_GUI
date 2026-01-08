#include <wx/wx.h>
#include <wx/image.h>
#include <fstream>
#include <ctime>
#include <iostream>
#include <string>
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

    // Attempt to write a minidump to %TEMP% so we can analyze locally
    try {
        char tmpPath[MAX_PATH] = {0};
        DWORD r = GetTempPathA(MAX_PATH, tmpPath);
        if (r > 0 && r < MAX_PATH) {
            char fname[MAX_PATH];
            SYSTEMTIME st; GetLocalTime(&st);
            sprintf_s(fname, "%stripsitter_crash_%lu_%04u%02u%02u_%02u%02u%02u.dmp", tmpPath, (unsigned)GetCurrentProcessId(), st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
            HANDLE hFile = CreateFileA(fname, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
            if (hFile != INVALID_HANDLE_VALUE) {
                MINIDUMP_EXCEPTION_INFORMATION mei;
                mei.ThreadId = GetCurrentThreadId();
                mei.ExceptionPointers = ex;
                mei.ClientPointers = FALSE;
                BOOL ok = MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), hFile, MiniDumpWithFullMemory, &mei, NULL, NULL);
                dbg << "MiniDumpWriteDump " << (ok ? "succeeded" : "failed") << " path=" << fname << std::endl;
                CloseHandle(hFile);
            } else {
                dbg << "Failed to open dump file path=" << fname << " err=" << GetLastError() << std::endl;
            }
        } else {
            dbg << "GetTempPathA failed or path too long" << std::endl;
        }
    } catch(...) {
        dbg << "Exception while trying to write minidump" << std::endl;
    }

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

        bool checkWallpaperMode = false;
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--check-wallpaper") {
                checkWallpaperMode = true;
                break;
            }
        }

        MainWindow* frame = new MainWindow();
        if (checkWallpaperMode) {
            bool has = frame->HasWallpaper();
            std::string res = has ? "WALLPAPER_FOUND" : "WALLPAPER_MISSING";
            std::cout << res << std::endl;

            // If running in CI (or WRITE_WALLPAPER_ARTIFACT is set), write a small artifact file
            const char* githubActions = std::getenv("GITHUB_ACTIONS");
            const char* ciVar = std::getenv("CI");
            const char* writeArtifact = std::getenv("WRITE_WALLPAPER_ARTIFACT");
            if (githubActions || ciVar || writeArtifact) {
                try {
                    std::ofstream out("wallpaper_check.txt");
                    if (out.is_open()) {
                        out << res << std::endl;
                        out.close();
                    }
                } catch (...) {
                    // Ignore failures to avoid affecting CI flow
                }
            }

            // Exit immediately with 0=found, 1=missing
            std::exit(has ? 0 : 1);
        }

        frame->Show(true);

        return true; 
    }
};

wxIMPLEMENT_APP(TripSitterApp);
