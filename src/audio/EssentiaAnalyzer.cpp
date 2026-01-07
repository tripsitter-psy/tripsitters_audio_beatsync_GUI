#include "EssentiaAnalyzer.h"
#include "BeatGrid.h"
#include <iostream>
#include <mutex>

#ifdef HAVE_ESSENTIA
#include <essentia/essentia.h>
#include <essentia/algorithmfactory.h>
#include <essentia/pool.h>
using namespace essentia;
using namespace essentia::standard;
#endif

namespace BeatSync {

BeatGrid EssentiaAnalyzer::analyze(const std::string& audioFilePath) {
#ifdef HAVE_ESSENTIA
    static std::once_flag initFlag;
    try {
        std::call_once(initFlag, [](){
            essentia::init();
        });

        AlgorithmFactory& factory = standard::AlgorithmFactory::instance();

        // Load audio file (MonoLoader will resample as needed)
        std::vector<Real> audio;
        Algorithm* loader = factory.create("MonoLoader",
                                          "filename", audioFilePath,
                                          "sampleRate", 44100);
        loader->output("audio").set(audio);
        loader->compute();
        delete loader;

        if (audio.empty()) {
            m_lastError = "Could not load audio or audio empty";
            return BeatGrid();
        }

        // RhythmExtractor2013 returns beats and various analyses
        std::vector<Real> beats;
        Real bpm = 0.0;
        Real confidence = 0.0;
        Algorithm* rhythm = factory.create("RhythmExtractor2013",
                                           "method", std::string("multifeature"));
        rhythm->input("audio").set(audio);
        rhythm->output("bpm").set(bpm);
        rhythm->output("beats").set(beats);
        rhythm->output("confidence").set(confidence);
        rhythm->compute();
        delete rhythm;

        BeatGrid grid;
        grid.setBPM(static_cast<double>(bpm));
        for (auto &b : beats) grid.addBeat(static_cast<double>(b));
        m_lastError.clear();
        return grid;
    } catch (const std::exception& e) {
        m_lastError = std::string("Essentia error: ") + e.what();
        return BeatGrid();
    }
#else
    (void)audioFilePath;
    m_lastError = "Essentia integration not enabled (stub)";
    return BeatGrid();
#endif
}

} // namespace BeatSync
