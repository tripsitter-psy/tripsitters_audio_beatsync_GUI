#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace BeatSync {

/**
 * @brief AI frame interpolation using RIFE (Real-Time Intermediate Flow Estimation)
 *
 * Runs vs-mlrt style RIFE ONNX exports (input [1,11,H,W]: img0, img1, timestep,
 * x/y meshgrids, multiplier planes) via ONNX Runtime with CUDA when available.
 * Used by the speed-ramp pipeline to synthesize intermediate frames for smooth
 * slow-mo instead of ffmpeg's minterpolate.
 */
class RifeInterpolator {
public:
    RifeInterpolator();
    ~RifeInterpolator();

    /**
     * @brief Load a RIFE ONNX model
     * @param modelPath Path to e.g. rife_v4.15.onnx (vs-mlrt export)
     * @param useGPU Try CUDA execution provider (falls back to CPU)
     */
    bool loadModel(const std::string& modelPath, bool useGPU = true, int gpuDeviceId = 0);

    bool isLoaded() const;

    /**
     * @brief Synthesize the frame at time t between two frames
     * @param img0 First frame, packed RGB24 (width * height * 3 bytes)
     * @param img1 Second frame, same layout
     * @param width Frame width in pixels
     * @param height Frame height in pixels
     * @param timestep Position between frames in (0,1); 0.5 = midpoint
     * @param outFrame Receives the synthesized frame, RGB24, same dimensions
     *
     * Frames are edge-padded internally to the model's required multiple of 32.
     */
    bool interpolate(const uint8_t* img0, const uint8_t* img1,
                     int width, int height, float timestep,
                     std::vector<uint8_t>& outFrame);

    std::string getLastError() const;
    std::string getActiveProvider() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace BeatSync
