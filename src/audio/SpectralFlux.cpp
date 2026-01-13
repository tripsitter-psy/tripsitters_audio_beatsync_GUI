#include "SpectralFlux.h"
#include "tracing/Tracing.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <valarray>

static const double PI = 3.14159265358979323846;

// Lightweight FFT implementation using std::complex + naive DFT for small windows would be slow.
// Instead, we use KissFFT if available in the project, otherwise fall back to a simple real FFT.
// The repo already includes a simple FFT analyzer; we'll implement a minimal real-valued FFT here
// with an iterative Cooley-Tukey radix-2 algorithm for power-of-two sizes. This keeps the detector
// self-contained and deterministic.

#include <complex>

namespace BeatSync {

// Helper: next power of two
static int nextPow2(int v) {
    int p = 1;
    while (p < v) p <<= 1;
    return p;
}

// In-place radix-2 Cooley-Tukey FFT
static void fft(std::vector<std::complex<double>>& a) {
    const int n = (int)a.size();
    int j = 0;
    for (int i = 1; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }

    for (int len = 2; len <= n; len <<= 1) {
        double ang = -2.0 * PI / len;
        std::complex<double> wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            std::complex<double> w(1);
            for (int k = 0; k < len/2; ++k) {
                std::complex<double> u = a[i+k];
                std::complex<double> v = a[i+k+len/2] * w;
                a[i+k] = u + v;
                a[i+k+len/2] = u - v;
                w *= wlen;
            }
        }
    }
}

// Hann window
static void applyHann(std::vector<double>& buf) {
    int N = (int)buf.size();
    if (N < 2) {
        // For N == 1, Hann window is 1.0; for N == 0, no operation needed
        if (N == 1) buf[0] *= 1.0;
        return;
    }
    for (int n = 0; n < N; ++n) buf[n] *= 0.5 * (1 - cos(2.0 * PI * n / (N - 1)));
}

static std::vector<double> gaussianSmooth(const std::vector<double>& in, double sigma) {
    if (sigma <= 0.0) return in;
    int radius = std::max(1, int(ceil(3 * sigma)));
    int len = 2 * radius + 1;
    std::vector<double> kernel(len);
    double sum = 0.0;
    for (int i = -radius; i <= radius; ++i) {
        double v = exp(-0.5 * (i*i) / (sigma*sigma));
        kernel[i + radius] = v;
        sum += v;
    }
    for (double &k : kernel) k /= sum;
    int N = (int)in.size();
    std::vector<double> out(N);
    for (int i = 0; i < N; ++i) {
        double s = 0.0;
        for (int k = -radius; k <= radius; ++k) {
            int idx = std::clamp(i + k, 0, N-1);
            s += in[idx] * kernel[k + radius];
        }
        out[i] = s;
    }
    return out;
}

std::vector<double> detectBeatsFromWaveform(const std::vector<float>& samples, int sampleRate,
                                            int windowSize, int hopSize,
                                            double smoothSigma, double thresholdFactor,
                                            double minBeatDistanceSeconds) {
    TRACE_FUNC();
    std::vector<double> beats;
    if (samples.empty() || sampleRate <= 0 || windowSize <= 0 || hopSize <= 0) return beats;

    int N = windowSize;
    int H = hopSize;
    int numFrames = 1 + (int(std::max(0, (int)samples.size() - N)) / H);
    if (numFrames <= 0) return beats;

    // Compute magnitude spectrogram frames
    std::vector<double> prevMag(N/2 + 1, 0.0);
    std::vector<double> flux(numFrames, 0.0);

    std::vector<std::complex<double>> buf(nextPow2(N));

    for (int f = 0; f < numFrames; ++f) {
        int offset = f * H;
        for (int i = 0; i < N; ++i) {
            double v = 0.0;
            if (offset + i < (int)samples.size()) v = samples[offset + i];
            buf[i] = std::complex<double>(v, 0.0);
        }
        for (int i = N; i < (int)buf.size(); ++i) buf[i] = 0.0;
        // Apply Hann
        std::vector<double> win(N);
        for (int i = 0; i < N; ++i) win[i] = buf[i].real();
        applyHann(win);
        for (int i = 0; i < N; ++i) buf[i] = std::complex<double>(win[i], 0.0);

        fft(buf);
        int half = N/2;
        std::vector<double> mag(half + 1);
        for (int k = 0; k <= half; ++k) {
            double re = buf[k].real(), im = buf[k].imag();
            mag[k] = sqrt(re*re + im*im);
        }

        // Spectral flux (only positive increases)
        double sumPos = 0.0;
        for (int k = 0; k <= half; ++k) {
            double diff = mag[k] - prevMag[k];
            if (diff > 0) sumPos += diff;
        }
        flux[f] = sumPos;
        prevMag = mag;
    }

    // Normalize flux
    double mean = std::accumulate(flux.begin(), flux.end(), 0.0) / flux.size();
    double sq = 0.0;
    for (double v : flux) sq += (v-mean)*(v-mean);
    double stdev = 0.0;
    if (flux.size() > 1) {
        stdev = sqrt(sq / (flux.size() - 1)); // sample stdev for small datasets
    } else if (flux.size() == 1) {
        stdev = 0.0;
    } else {
        stdev = 0.0;
    }
    for (double &v : flux) v = (v - mean) / (stdev + 1e-9);

    // Smooth
    auto smooth = gaussianSmooth(flux, smoothSigma);

    // Adaptive threshold: median + thresholdFactor * std
    std::vector<double> peaks;
    for (size_t i = 1; i + 1 < smooth.size(); ++i) {
        if (smooth[i] > smooth[i-1] && smooth[i] >= smooth[i+1]) {
            // compute local threshold using a small window
            int w = 3;
            int start = std::max<int>(0, int(i) - w);
            int end = std::min<int>(smooth.size()-1, int(i) + w);
            double localMean = 0.0;
            for (int j = start; j <= end; ++j) localMean += smooth[j];
            localMean /= (end-start+1);
            if (smooth[i] > localMean * thresholdFactor) peaks.push_back((double)i);
        }
    }

    // Convert frame indices to time and perform simple pruning (minimum distance minBeatDistanceSeconds)
    double lastT = -1e9;
    for (double p : peaks) {
        double t = p * H / double(sampleRate);
        if (t - lastT >= minBeatDistanceSeconds) {
            beats.push_back(t);
            lastT = t;
        }
    }

    return beats;
}

} // namespace BeatSync
