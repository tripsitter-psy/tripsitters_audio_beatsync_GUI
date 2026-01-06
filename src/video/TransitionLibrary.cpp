#include "TransitionLibrary.h"

namespace BeatSync {

bool TransitionLibrary::loadFromDirectory(const std::string& dirPath) {
    (void)dirPath;
    m_lastError = "Transition library loading not implemented (stub)";
    // TODO: iterate .glsl files, parse metadata comments for name/category
    return false;
}

const TransitionShader* TransitionLibrary::findByName(const std::string& name) const {
    for (const auto& t : m_transitions) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

} // namespace BeatSync
