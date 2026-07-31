#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace BeatSync {

/**
 * @brief Neural image upscaler (ESRGAN-family) via ONNX Runtime.
 *
 * Upscales RGB24 frames with a compact super-resolution CNN. Targets
 * SRVGGNetCompact / Real-ESRGAN v2-v3 style exports with the signature:
 *   input  : [1,3,H,W] float (RGB in [0,1])
 *   output : [1,3,H*s,W*s] float
 * (e.g. realesr-animevideov3.onnx). The scale factor s is detected from the
 * model at load time, so 2x and 4x exports both work.
 *
 * Frames are processed in overlapping tiles so VRAM use stays bounded and
 * independent of source resolution. Uses the same CUDA -> DirectML -> CPU
 * provider chain as OnnxFrameInterpolator. If the model can't be loaded,
 * isLoaded() returns false and the caller should skip upscaling.
 */
class OnnxUpscaler {
public:
    OnnxUpscaler();
    ~OnnxUpscaler();

    OnnxUpscaler(const OnnxUpscaler&) = delete;
    OnnxUpscaler& operator=(const OnnxUpscaler&) = delete;

    /**
     * @brief Load an upscaling ONNX model.
     * @param modelPath Path to the .onnx file.
     * @param useGPU Try GPU execution providers (CUDA/DirectML) before CPU.
     * @param gpuDeviceId GPU device index.
     * @return true if the model loaded and its scale factor was detected.
     */
    bool loadModel(const std::string& modelPath, bool useGPU = true, int gpuDeviceId = 0);

    /** @brief True if a model is loaded and ready. */
    bool isLoaded() const;

    /** @brief Scale factor detected from the model (2, 3, 4...); 0 if not loaded. */
    int getScale() const;

    /**
     * @brief Maximum tile edge in source pixels (default 512). Larger tiles are
     * faster but use more VRAM; tiles overlap by 16px to hide seams.
     */
    void setTileSize(int tilePixels);

    /**
     * @brief Upscale one RGB24 frame.
     * @param rgb Source frame, width*height*3 interleaved RGB bytes.
     * @param width Source width.
     * @param height Source height.
     * @param outRgb Receives the upscaled frame (width*scale * height*scale * 3).
     * @return true on success.
     */
    bool upscale(const uint8_t* rgb, int width, int height,
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
