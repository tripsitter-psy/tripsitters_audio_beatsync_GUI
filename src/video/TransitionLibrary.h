#pragma once

#include <string>
#include <vector>

namespace BeatSync {

// Represents a GLSL transition shader loadable from assets/transitions/.
struct TransitionShader {
    std::string name;      // e.g. "fade", "wipe", "kaleidoscope"
    std::string category;  // geometric, blend, distortion, psychedelic
    std::string glslPath;  // path to .glsl file
};

// Loads and manages a library of GLSL transition shaders compatible with
// ffmpeg-gl-transition (gl-transitions.com). Shaders live in assets/transitions/.
class TransitionLibrary {
public:
    TransitionLibrary() = default;

    // Scan assets/transitions/ directory and load available shaders.
    bool loadFromDirectory(const std::string& dirPath);

    // Get all loaded transitions.
    const std::vector<TransitionShader>& getTransitions() const { return m_transitions; }

    // Lookup a transition by name. Returns nullptr if not found.
    const TransitionShader* findByName(const std::string& name) const;

    // Build an FFmpeg gltransition filter snippet for a named transition
    // Example output: "gltransition=source='path/to/fade.glsl':duration=0.3"
    std::string buildGlTransitionFilter(const std::string& name, double duration) const;

    const std::string& getLastError() const { return m_lastError; }

private:
    std::vector<TransitionShader> m_transitions;
    std::string m_lastError;
};

} // namespace BeatSync
