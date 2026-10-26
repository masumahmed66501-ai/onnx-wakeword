/*
 * VoicuteKWS — Arduino wrapper for the Voicute wake-word inference SDK
 * (Causal DS-TCN INT8 backbone + float MultiProto head, same pipeline as
 * onnx-wakeword/esp32/sdk/voicute).
 *
 * Hardware isolation contract: the library knows NOTHING about boards.
 *   in : int16 PCM, 16 kHz, mono, raw amplitude (NOT divided by 32768)
 *   out: onWake callback (model index, wake word)
 * Microphones, codecs, LEDs and state machines stay in the sketch.
 *
 * Model supply, two options:
 *   - beginEmbedded(): model compiled into the sketch as a C array (default,
 *     no filesystem needed) — pair with src/kws/head.h from the same package.
 *   - beginSpiffs(): all *.tflite files in a mounted filesystem dir (multi-word).
 *
 * Head: src/kws/head.h comes from the model export package and must match the
 * .tflite backbone (KWS_MEL_TIME etc.). Replace BOTH when swapping models.
 */
#pragma once
#include <Arduino.h>
#include <stdint.h>
#include <stddef.h>

#include "kws/recognizer.h"      // pulls model_loader.h (TFLite Micro) + mel contract
#include "kws/model_head.h"        // model-specific head: KWS_MEL_TIME, prototypes, ...

class VoicuteKWS {
public:
    // modelIdx, wake word, user data
    typedef void (*WakeCallback)(int modelIdx, const char* word, void* user);

    struct Config {
        float    threshold   = 0.7f;   // wake probability threshold
        uint32_t hopSamples  = 2560;   // inference cadence (~160 ms at 16 kHz)
        // Detection gates (API.md recommended baseline)
        bool  l1 = false;   // consecutive frames (needs fast inference)
        bool  l2 = true;    // peak/background ratio > 3x
        bool  l3 = true;    // 1.5 s cooldown
        bool  l4 = false;   // burst blocking (for media playback scenarios)
        bool  l5 = true;    // RMS energy-jump filter (music/video)
        float l5Delta = 200.0f;  // 0 → SDK default (1200)
    };

    VoicuteKWS();
    ~VoicuteKWS();

    // ---- Model supply (choose ONE begin*) ----
    // Single model compiled into the firmware. wakeWord names the registry
    // entry (free text, e.g. "doubaodoubao").
    // NOTE: default arguments of type Config() are not allowed inside the
    // enclosing class (NSDMI not complete yet) — overload instead.
    bool beginEmbedded(const uint8_t* modelTflite, size_t len, const char* wakeWord);
    bool beginEmbedded(const uint8_t* modelTflite, size_t len, const char* wakeWord,
                       kws_postprocess_fn postprocess);  // nullptr → single-head kws_postprocess()
    bool beginEmbedded(const uint8_t* modelTflite, size_t len, const char* wakeWord,
                       kws_postprocess_fn postprocess, const Config& cfg);
    // Multi-model from a mounted filesystem dir containing *.tflite
    // (e.g. SPIFFS.begin(true) then beginSpiffs("/spiffs", ...)).
    bool beginSpiffs(const char* basePath, kws_postprocess_fn postprocess);
    bool beginSpiffs(const char* basePath, kws_postprocess_fn postprocess, const Config& cfg);
// Append another flash-resident model AFTER a begin* call (multi-model
// embedded sketches, e.g. wake word + command words, up to MAX_WAKE_WORDS).
// The FIRST begin* call fixes the Config AND the postprocess function: one
// postprocess serves all models and dispatches on modelIdx (see
// examples/HMI_DevKit_WakeCmd for the wake + command pattern).
bool addEmbedded(const uint8_t* modelTflite, size_t len, const char* wakeWord);
// Per-model inference gating without reload (wake <-> command state machines).
// A disabled model skips Invoke entirely. Pair every transition with
// resetSmoothing(): a stale pre-switch peak can re-fire the other model.
void setActive(int idx, bool enabled) { recognizer_set_active(idx, enabled ? 1 : 0); }
void resetSmoothing() { recognizer_reset_smooth(); }

    // ---- Audio ----
    // Feed PCM as captured. The wrapper keeps a rolling window of
    // windowSamples() and infers at most once per hopSamples — just call this
    // from loop() as data arrives; do NOT pre-buffer or resample.
    void feed(const int16_t* pcm, size_t nSamples);

    void onWake(WakeCallback cb, void* user = nullptr);

    // ---- Diagnostics ----
    float    lastProb()  const { return recognizer_get_last_prob(); }
    bool     ready()     const { return m_ready; }
    int      numModels();
    const char* modelWord(int idx);
    // Input window length in samples: (KWS_MEL_TIME-1)*160 + 672
    //   T=98 → 16000 (1.0 s) | T=150 → 24512 | T=200 → 32512
    size_t   windowSamples() const { return MEL_AUDIO_LEN; }
    // Last frame timings (ms) — mel extraction / TFLite invoke
    float    lastMelMs()    const { return recognizer_get_last_mel_ms(); }
    float    lastInvokeMs() const { return recognizer_get_last_invoke_ms(); }

private:
    static void sEventTrampoline(voice_event_t ev, voice_evt_data_t data, void* arg);
    static float sDefaultPostprocess(const int8_t* out, float scale, int zero, int idx);
    bool init(const char* basePath, const uint8_t* embeddedModel, size_t embeddedLen,
              const char* wakeWord, kws_postprocess_fn postprocess, const Config& cfg);
    void maybeInfer();

    bool       m_ready = false;
    Config     m_cfg;
    int16_t*   m_hist = nullptr;      // rolling window buffer (windowSamples)
    size_t     m_wr = 0;              // write cursor in m_hist
    size_t     m_total = 0;           // samples seen since begin
    uint32_t   m_sinceInfer = 0;
    int16_t*   m_frame = nullptr;     // contiguous window handed to the SDK
    WakeCallback m_cb = nullptr;
    void*      m_cbUser = nullptr;
    struct Bind { VoicuteKWS* self; int idx; } m_bind[MAX_WAKE_WORDS];
};
