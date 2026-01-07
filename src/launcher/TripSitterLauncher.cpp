#include <windows.h>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

int wmain(int argc, wchar_t* argv[])
{
    std::wstring exePath = L"TripSitter.exe";
    if (argc > 1) exePath = argv[1];

    std::ofstream dbg("tripsitter_debug.log", std::ios::app);
    dbg << "Launcher: starting child '" << std::string(exePath.begin(), exePath.end()) << "'" << std::endl;
    dbg.flush();

    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));

    BOOL ok = CreateProcessW(
        NULL,
        exePath.data(),
        NULL,
        NULL,
        FALSE,
        0,
        NULL,
        NULL,
        &si,
        &pi
    );

    if (!ok) {
        DWORD err = GetLastError();
        LPWSTR msg = NULL;
        FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPWSTR)&msg, 0, NULL);
        std::wstring wmsg = msg ? msg : L"(no message)";
        std::string emsg(wmsg.begin(), wmsg.end());
        dbg << "Launcher: CreateProcess failed, error=" << err << " msg='" << emsg << "'" << std::endl;
        dbg.flush();
        if (msg) LocalFree(msg);
        std::cerr << "CreateProcess failed: " << err << " - " << emsg << std::endl;
        return 1;
    }

    // Wait for the process (with a reasonable timeout so the launcher doesn't hang forever)
    DWORD wait = WaitForSingleObject(pi.hProcess, 15000); // 15s
    if (wait == WAIT_TIMEOUT) {
        dbg << "Launcher: child still running after timeout; detaching" << std::endl;
        dbg.flush();
        // Also check for marker files in %TEMP% to see startup progress
        char* tmp = std::getenv("TMP"); if (!tmp) tmp = std::getenv("TEMP");
        std::string tempPath = tmp ? tmp : ".";
        std::vector<std::string> markers = {"tripsitter_marker_OnInit_start.txt","tripsitter_marker_MainWindow_created.txt","tripsitter_marker_MainWindow_shown.txt","tripsitter_marker_MainWindow_ctor_start.txt","tripsitter_marker_MainWindow_ctor_end.txt","EARLY_START_MARKER"};
        for (auto &mname : markers) {
            std::string full = tempPath + "\\" + mname;
            std::ifstream f(full);
            if (f.good()) {
                dbg << "Launcher: found marker " << mname << std::endl;
            }
        }
        dbg.flush();
        // Detach and return success; leave child running
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
        return 0;
    }

    DWORD exitCode = 0;
    if (GetExitCodeProcess(pi.hProcess, &exitCode)) {
        dbg << "Launcher: child exited with code=" << exitCode << std::endl;
        dbg.flush();
        // Check markers to see where it got to
        char* tmp = std::getenv("TMP"); if (!tmp) tmp = std::getenv("TEMP");
        std::string tempPath = tmp ? tmp : ".";
        std::vector<std::string> markers = {"tripsitter_marker_OnInit_start.txt","tripsitter_marker_MainWindow_created.txt","tripsitter_marker_MainWindow_shown.txt","tripsitter_marker_MainWindow_ctor_start.txt","tripsitter_marker_MainWindow_ctor_end.txt","EARLY_START_MARKER"};
        for (auto &mname : markers) {
            std::string full = tempPath + "\\" + mname;
            std::ifstream f(full);
            if (f.good()) {
                dbg << "Launcher: found marker " << mname << std::endl;
            }
        }
        dbg.flush();
    } else {
        dbg << "Launcher: failed to get child exit code" << std::endl;
        dbg.flush();
    }

    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);

    return (int)exitCode;
}
