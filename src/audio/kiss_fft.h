/*
 *  Copyright (c) 2003-2010, Mark Borgerding. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *  SPDX-License-Identifier: BSD-3-Clause
 *
 *  Minimal single-header adaptation for BeatSync FFTAnalyzer.
 *  Only includes forward real-to-complex FFT functionality needed for frequency analysis.
 */

#ifndef KISS_FFT_H
#define KISS_FFT_H

#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    float r;
    float i;
} kiss_fft_cpx;

typedef struct kiss_fft_state* kiss_fft_cfg;

struct kiss_fft_state {
    int nfft;
    int inverse;
    int* factors;
    kiss_fft_cpx* twiddles;
};

static void kf_bfly2(kiss_fft_cpx* Fout, const size_t fstride, const kiss_fft_cfg st, int m) {
    kiss_fft_cpx* Fout2;
    kiss_fft_cpx* tw1 = st->twiddles;
    kiss_fft_cpx t;
    Fout2 = Fout + m;
    do {
        t.r = Fout2->r * tw1->r - Fout2->i * tw1->i;
        t.i = Fout2->r * tw1->i + Fout2->i * tw1->r;
        tw1 += fstride;
        Fout2->r = Fout->r - t.r;
        Fout2->i = Fout->i - t.i;
        Fout->r += t.r;
        Fout->i += t.i;
        ++Fout2;
        ++Fout;
    } while (--m);
}

static void kf_bfly4(kiss_fft_cpx* Fout, const size_t fstride, const kiss_fft_cfg st, const size_t m) {
    kiss_fft_cpx* tw1, * tw2, * tw3;
    kiss_fft_cpx scratch[6];
    size_t k = m;
    const size_t m2 = 2 * m;
    const size_t m3 = 3 * m;

    tw3 = tw2 = tw1 = st->twiddles;

    do {
        scratch[0].r = Fout[m].r * tw1->r - Fout[m].i * tw1->i;
        scratch[0].i = Fout[m].r * tw1->i + Fout[m].i * tw1->r;
        scratch[1].r = Fout[m2].r * tw2->r - Fout[m2].i * tw2->i;
        scratch[1].i = Fout[m2].r * tw2->i + Fout[m2].i * tw2->r;
        scratch[2].r = Fout[m3].r * tw3->r - Fout[m3].i * tw3->i;
        scratch[2].i = Fout[m3].r * tw3->i + Fout[m3].i * tw3->r;

        scratch[5].r = Fout->r - scratch[1].r;
        scratch[5].i = Fout->i - scratch[1].i;
        Fout->r += scratch[1].r;
        Fout->i += scratch[1].i;
        scratch[3].r = scratch[0].r + scratch[2].r;
        scratch[3].i = scratch[0].i + scratch[2].i;
        scratch[4].r = scratch[0].r - scratch[2].r;
        scratch[4].i = scratch[0].i - scratch[2].i;
        Fout[m2].r = Fout->r - scratch[3].r;
        Fout[m2].i = Fout->i - scratch[3].i;
        tw1 += fstride;
        tw2 += fstride * 2;
        tw3 += fstride * 3;
        Fout->r += scratch[3].r;
        Fout->i += scratch[3].i;

        if (st->inverse) {
            Fout[m].r = scratch[5].r - scratch[4].i;
            Fout[m].i = scratch[5].i + scratch[4].r;
            Fout[m3].r = scratch[5].r + scratch[4].i;
            Fout[m3].i = scratch[5].i - scratch[4].r;
        } else {
            Fout[m].r = scratch[5].r + scratch[4].i;
            Fout[m].i = scratch[5].i - scratch[4].r;
            Fout[m3].r = scratch[5].r - scratch[4].i;
            Fout[m3].i = scratch[5].i + scratch[4].r;
        }
        ++Fout;
    } while (--k);
}

static void kf_bfly_generic(kiss_fft_cpx* Fout, const size_t fstride, const kiss_fft_cfg st, int m, int p) {
    int u, k, q1, q;
    kiss_fft_cpx* twiddles = st->twiddles;
    kiss_fft_cpx t;
    int Norig = st->nfft;

    kiss_fft_cpx* scratch = (kiss_fft_cpx*)malloc(sizeof(kiss_fft_cpx) * p);

    for (u = 0; u < m; ++u) {
        k = u;
        for (q1 = 0; q1 < p; ++q1) {
            scratch[q1] = Fout[k];
            k += m;
        }

        k = u;
        for (q1 = 0; q1 < p; ++q1) {
            int twidx = 0;
            Fout[k].r = 0;
            Fout[k].i = 0;
            for (q = 0; q < p; ++q) {
                t.r = scratch[q].r * twiddles[twidx].r - scratch[q].i * twiddles[twidx].i;
                t.i = scratch[q].r * twiddles[twidx].i + scratch[q].i * twiddles[twidx].r;
                Fout[k].r += t.r;
                Fout[k].i += t.i;
                twidx += (int)fstride * k;
                if (twidx >= Norig) twidx -= Norig;
            }
            k += m;
        }
    }
    free(scratch);
}

static void kf_work(kiss_fft_cpx* Fout, const kiss_fft_cpx* f, const size_t fstride, int in_stride, int* factors, const kiss_fft_cfg st) {
    kiss_fft_cpx* Fout_beg = Fout;
    const int p = *factors++;
    const int m = *factors++;
    const kiss_fft_cpx* Fout_end = Fout + p * m;

    if (m == 1) {
        do {
            *Fout = *f;
            f += fstride * in_stride;
        } while (++Fout != Fout_end);
    } else {
        do {
            kf_work(Fout, f, fstride * p, in_stride, factors, st);
            f += fstride * in_stride;
        } while ((Fout += m) != Fout_end);
    }

    Fout = Fout_beg;

    switch (p) {
        case 2: kf_bfly2(Fout, fstride, st, m); break;
        case 4: kf_bfly4(Fout, fstride, st, (const size_t)m); break;
        default: kf_bfly_generic(Fout, fstride, st, m, p); break;
    }
}

static void kf_factor(int n, int* facbuf) {
    int p = 4;
    double floor_sqrt;
    floor_sqrt = floor(sqrt((double)n));

    while (n > 1) {
        while (n % p) {
            switch (p) {
                case 4: p = 2; break;
                case 2: p = 3; break;
                default: p += 2; break;
            }
            if (p > floor_sqrt) p = n;
        }
        n /= p;
        *facbuf++ = p;
        *facbuf++ = n;
    }
}

static kiss_fft_cfg kiss_fft_alloc(int nfft, int inverse_fft, void* mem, size_t* lenmem) {
    kiss_fft_cfg st = NULL;
    size_t memneeded = sizeof(struct kiss_fft_state)
        + sizeof(kiss_fft_cpx) * (size_t)(nfft - 1)
        + sizeof(int) * 2 * 32; /* twiddles & factors */

    if (lenmem == NULL) {
        st = (kiss_fft_cfg)malloc(memneeded);
    } else {
        if (mem != NULL && *lenmem >= memneeded)
            st = (kiss_fft_cfg)mem;
        *lenmem = memneeded;
    }
    if (st) {
        st->nfft = nfft;
        st->inverse = inverse_fft;

        st->twiddles = (kiss_fft_cpx*)(st + 1);
        st->factors = (int*)(st->twiddles + nfft);

        for (int i = 0; i < nfft; ++i) {
            double phase = -2 * M_PI * i / nfft;
            if (st->inverse) phase *= -1;
            st->twiddles[i].r = (float)cos(phase);
            st->twiddles[i].i = (float)sin(phase);
        }

        kf_factor(nfft, st->factors);
    }
    return st;
}

static void kiss_fft(kiss_fft_cfg cfg, const kiss_fft_cpx* fin, kiss_fft_cpx* fout) {
    if (fin == fout) {
        kiss_fft_cpx* tmpbuf = (kiss_fft_cpx*)malloc(sizeof(kiss_fft_cpx) * cfg->nfft);
        kf_work(tmpbuf, fin, 1, 1, cfg->factors, cfg);
        memcpy(fout, tmpbuf, sizeof(kiss_fft_cpx) * cfg->nfft);
        free(tmpbuf);
    } else {
        kf_work(fout, fin, 1, 1, cfg->factors, cfg);
    }
}

static void kiss_fft_free(kiss_fft_cfg cfg) {
    free(cfg);
}

/* Real-valued FFT helper: converts N real samples to N/2+1 complex bins */
static void kiss_fftr(const float* timedata, kiss_fft_cpx* freqdata, int nfft) {
    kiss_fft_cfg cfg = kiss_fft_alloc(nfft, 0, NULL, NULL);
    if (!cfg) return;

    kiss_fft_cpx* tmpbuf = (kiss_fft_cpx*)malloc(sizeof(kiss_fft_cpx) * nfft);
    if (!tmpbuf) {
        kiss_fft_free(cfg);
        return;
    }

    /* Pack real data into complex input (imaginary = 0) */
    for (int i = 0; i < nfft; i++) {
        tmpbuf[i].r = timedata[i];
        tmpbuf[i].i = 0.0f;
    }

    kiss_fft(cfg, tmpbuf, freqdata);

    free(tmpbuf);
    kiss_fft_free(cfg);
}

#ifdef __cplusplus
}
#endif

#endif /* KISS_FFT_H */
