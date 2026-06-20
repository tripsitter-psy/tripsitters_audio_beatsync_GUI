#include "FileDialog.h"

#include <cstdio>
#include <array>
#include <sstream>

namespace
{
// Run a shell command and capture trimmed stdout.
std::string RunCapture(const std::string& cmd)
{
    std::string out;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return out;
    std::array<char, 4096> buf{};
    while (fgets(buf.data(), (int)buf.size(), pipe))
        out += buf.data();
    pclose(pipe);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r'))
        out.pop_back();
    return out;
}
} // namespace

#if defined(__APPLE__)

namespace
{
// Build an AppleScript "of type {\"wav\",\"mp3\"}" clause, or empty for any.
std::string TypeClause(const std::vector<std::string>& exts)
{
    if (exts.empty()) return "";
    std::ostringstream os;
    os << " of type {";
    for (size_t i = 0; i < exts.size(); ++i)
    {
        if (i) os << ", ";
        os << "\"" << exts[i] << "\"";
    }
    os << "}";
    return os.str();
}

std::string Osascript(const std::string& script)
{
    // Wrap each line as a -e argument; single-line scripts are fine here.
    return RunCapture("osascript -e '" + script + "' 2>/dev/null");
}
} // namespace

std::string FileDialog::OpenFile(const std::string& title,
                                 const std::string& /*filterDesc*/,
                                 const std::vector<std::string>& exts)
{
    std::string script =
        "set f to choose file with prompt \"" + title + "\"" + TypeClause(exts) +
        "\nPOSIX path of f";
    // Re-wrap for multi-line osascript.
    return RunCapture("osascript -e 'set f to choose file with prompt \"" + title +
                      "\"" + TypeClause(exts) + "' -e 'POSIX path of f' 2>/dev/null");
}

std::vector<std::string> FileDialog::OpenFiles(const std::string& title,
                                               const std::string& /*filterDesc*/,
                                               const std::vector<std::string>& exts)
{
    // Returns POSIX paths separated by newlines.
    std::string raw = RunCapture(
        "osascript -e 'set fs to choose file with prompt \"" + title + "\"" +
        TypeClause(exts) + " with multiple selections allowed' "
        "-e 'set out to \"\"' "
        "-e 'repeat with f in fs' "
        "-e 'set out to out & POSIX path of f & linefeed' "
        "-e 'end repeat' "
        "-e 'return out' 2>/dev/null");

    std::vector<std::string> paths;
    std::istringstream is(raw);
    std::string line;
    while (std::getline(is, line))
        if (!line.empty()) paths.push_back(line);
    return paths;
}

std::string FileDialog::SaveFile(const std::string& title,
                                 const std::string& defaultName,
                                 const std::vector<std::string>& /*exts*/)
{
    return RunCapture(
        "osascript -e 'set f to choose file name with prompt \"" + title +
        "\" default name \"" + defaultName + "\"' -e 'POSIX path of f' 2>/dev/null");
}

std::string FileDialog::PickFolder(const std::string& title)
{
    return RunCapture(
        "osascript -e 'set f to choose folder with prompt \"" + title +
        "\"' -e 'POSIX path of f' 2>/dev/null");
}

#else // ---- Windows / Linux: stubs to be implemented per-platform ----------

std::string FileDialog::OpenFile(const std::string&, const std::string&,
                                 const std::vector<std::string>&) { return ""; }
std::vector<std::string> FileDialog::OpenFiles(const std::string&, const std::string&,
                                               const std::vector<std::string>&) { return {}; }
std::string FileDialog::SaveFile(const std::string&, const std::string&,
                                 const std::vector<std::string>&) { return ""; }
std::string FileDialog::PickFolder(const std::string&) { return ""; }

#endif
