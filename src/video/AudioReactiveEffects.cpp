#include "AudioReactiveEffects.h"
#include <sstream>
#include <algorithm>

namespace BeatSync {

ReactiveParams AudioReactiveEffects::bandsToParams(const FrequencyBands& bands) {
    ReactiveParams p;
    // Normalize raw energy to [0,1]; placeholder linear mapping
    p.zoomIntensity = std::min(bands.bass, 1.0f);
    p.saturation    = std::min(bands.mids, 1.0f);
    p.brightness    = std::min(bands.highs, 1.0f);
    return p;
}

std::string AudioReactiveEffects::buildFilterSnippet(const ReactiveParams& params) {
    // Placeholder: generate eq/colorbalance filters driven by params
    (void)params;
    return ""; // TODO: produce actual FFmpeg filter graph string
}

} // namespace BeatSync
