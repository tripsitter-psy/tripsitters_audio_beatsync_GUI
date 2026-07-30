#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace BeatSync {

/**
 * @brief Neural frame interpolator (RIFE) via ONNX Runtime.
 *
 * Synthesizes an intermediate frame between two RGB frames at an arbitrary
 * timestep t in (0,1). Used to make slow-motion speed ramps look smooth on
 * low-fps source footage instead of duplicating frames.
 *
 * Targets RIFE v4.x ONNX exports with the signature:
 *   inputs : img0 [1,3,H,W] float, img1 [1,3,H,W] float, timestep [1] float
 *   output : [1,3,H,W] float (RGB in [0,1])
 * (e.g. rife47_ensemble_True_scale_1_sim.onnx). Input H/W are padded up to a
 * multiple of 32 internally and the output is cropped back.
 *
 * Uses a CUDA -> DirectML -> CPU execution-provider fallback chain. If the
 * model can't be loaded (missing file, no ONNX support), isLoaded() returns
 * false and the caller should fall back to a non-neural smoothing mode.
 */
class OnnxFrameInterpolator {
public:
    OnnxFrameInterpolator();
    ~OnnxFrameInterpolator();

    OnnxFrameInterpolator(const OnnxFrameInterpolator&) = delete;
    OnnxFrameInterpolator& operator=(const OnnxFrameInterpolator&) = delete;

    /**
     * @brief Load a RIFE ONNX model.
     * @param modelPath Path to the .onnx file.
     * @param useGPU Try GPU execution providers (CUDA/DirectML) before CPU.
     * @param gpuDeviceId GPU device index.
     * @return true if the model loaded and is ready for inference.
     */
    bool loadModel(const std::string& modelPath, bool useGPU = true, int gpuDeviceId = 0);

    /** @brief True if a model is loaded and ready. */
    bool isLoaded() const;

    /**
     * @brief Interpolate a frame at timestep t between two RGB24 frames.
     * @param rgb0 First frame, width*height*3 interleaved RGB bytes.
     * @param rgb1 Second frame, same size.
     * @param width Frame width in pixels.
     * @param height Frame height in pixels.
     * @param t Interpolation position in (0,1) (0 = rgb0, 1 = rgb1).
     * @param outRgb Output frame (resized to width*height*3).
     * @return true on success.
     */
    bool interpolate(const uint8_t* rgb0, const uint8_t* rgb1,
                     int width, int height, float t,
                     std::vector<uint8_t>& outRgb);

    /** @brief Last error message. */
    std::string getLastError() const;

    /** @brief Whether ONNX Runtime support was compiled in. */
    static bool isAvailable();

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace BeatSync
