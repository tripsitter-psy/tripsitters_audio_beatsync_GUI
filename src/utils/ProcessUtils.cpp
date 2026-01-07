#include "ProcessUtils.h"

#ifdef _WIN32
#include <windows.h>
#include <vector>
#else
#include <cstdio>
#include <cstdlib>
#endif

namespace BeatSync {

int runHiddenCommand(const std::string& cmdLine, std::string& output) {
#ifdef _WIN32
    SECURITY_ATTRIBUTES sa;
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    sa.lpSecurityDescriptor = NULL;

    HANDLE hReadPipe, hWritePipe;
    if (!CreatePipe(&hReadPipe, &hWritePipe, &sa, 0)) {
        return -1;
    }
    SetHandleInformation(hReadPipe, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA si = {0};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESHOWWINDOW | STARTF_USESTDHANDLES;
    si.wShowWindow = SW_HIDE;
    si.hStdOutput = hWritePipe;
    si.hStdError = hWritePipe;

    PROCESS_INFORMATION pi = {0};

    std::string exePath;
    std::string args;

    if (!cmdLine.empty() && cmdLine[0] == '"') {
        size_t endQuote = cmdLine.find('"', 1);
        if (endQuote != std::string::npos) {
            exePath = cmdLine.substr(1, endQuote - 1);
            args = cmdLine.substr(endQuote + 1);
        }
    } else {
        size_t space = cmdLine.find(' ');
        if (space != std::string::npos) {
            exePath = cmdLine.substr(0, space);
            args = cmdLine.substr(space);
        } else {
            exePath = cmdLine;
        }
    }

    std::string fullCmdLine = "\"" + exePath + "\"" + args;
    std::vector<char> cmdBuf(fullCmdLine.begin(), fullCmdLine.end());
    cmdBuf.push_back('\0');

    BOOL ok = CreateProcessA(exePath.c_str(), cmdBuf.data(), NULL, NULL, TRUE,
                              CREATE_NO_WINDOW, NULL, NULL, &si, &pi);
    CloseHandle(hWritePipe);
    if (!ok) {
        CloseHandle(hReadPipe);
        return -1;
    }

    char buf[512];
    DWORD bytesRead;
    while (ReadFile(hReadPipe, buf, sizeof(buf) - 1, &bytesRead, NULL) && bytesRead > 0) {
        buf[bytesRead] = '\0';
        output += buf;
    }
    CloseHandle(hReadPipe);

    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exitCode = 0;
    GetExitCodeProcess(pi.hProcess, &exitCode);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return static_cast<int>(exitCode);
#else
    // POSIX fallback: use popen and capture stdout+stderr via shell redirection
    std::string fullCmd = cmdLine + " 2>&1";
    FILE* pipe = popen(fullCmd.c_str(), "r");
    if (!pipe) return -1;
    char buffer[512];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }
    int rc = pclose(pipe);
    return rc;
#endif
}

} // namespace BeatSync
