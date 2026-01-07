#include "ProcessUtils.h"
#include <string>
#include <cstdio>
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#include <vector>
#else
#include <unistd.h>
#include <sys/wait.h>
#endif

namespace BeatSync {

int runHiddenCommand(const std::string& command, std::string& output) {
    output.clear();

#ifdef _WIN32
    // Windows implementation using CreateProcess
    SECURITY_ATTRIBUTES saAttr;
    saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
    saAttr.bInheritHandle = TRUE;
    saAttr.lpSecurityDescriptor = NULL;

    HANDLE hChildStd_OUT_Rd = NULL;
    HANDLE hChildStd_OUT_Wr = NULL;

    if (!CreatePipe(&hChildStd_OUT_Rd, &hChildStd_OUT_Wr, &saAttr, 0)) {
        return -1;
    }

    if (!SetHandleInformation(hChildStd_OUT_Rd, HANDLE_FLAG_INHERIT, 0)) {
        CloseHandle(hChildStd_OUT_Rd);
        CloseHandle(hChildStd_OUT_Wr);
        return -1;
    }

    PROCESS_INFORMATION piProcInfo;
    STARTUPINFOA siStartInfo;
    ZeroMemory(&piProcInfo, sizeof(PROCESS_INFORMATION));
    ZeroMemory(&siStartInfo, sizeof(STARTUPINFOA));
    siStartInfo.cb = sizeof(STARTUPINFOA);
    siStartInfo.hStdError = hChildStd_OUT_Wr;
    siStartInfo.hStdOutput = hChildStd_OUT_Wr;
    siStartInfo.hStdInput = NULL;
    siStartInfo.dwFlags |= STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    siStartInfo.wShowWindow = SW_HIDE;

    std::vector<char> cmdBuf(command.begin(), command.end());
    cmdBuf.push_back('\0');

    if (!CreateProcessA(NULL, cmdBuf.data(), NULL, NULL, TRUE, 0, NULL, NULL, &siStartInfo, &piProcInfo)) {
        CloseHandle(hChildStd_OUT_Rd);
        CloseHandle(hChildStd_OUT_Wr);
        return -1;
    }

    CloseHandle(hChildStd_OUT_Wr);

    const size_t bufSize = 4096;
    char buf[bufSize];
    DWORD bytesRead;
    while (ReadFile(hChildStd_OUT_Rd, buf, bufSize - 1, &bytesRead, NULL) && bytesRead > 0) {
        buf[bytesRead] = '\0';
        output += buf;
    }

    WaitForSingleObject(piProcInfo.hProcess, INFINITE);
    DWORD exitCode;
    GetExitCodeProcess(piProcInfo.hProcess, &exitCode);

    CloseHandle(piProcInfo.hProcess);
    CloseHandle(piProcInfo.hThread);
    CloseHandle(hChildStd_OUT_Rd);

    return static_cast<int>(exitCode);

#else
    // POSIX implementation using popen
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        return -1;
    }

    const size_t bufSize = 4096;
    char buf[bufSize];
    while (fgets(buf, bufSize, pipe) != nullptr) {
        output += buf;
    }

    int status = pclose(pipe);
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    } else {
        return -1;
    }
#endif
}

} // namespace BeatSync