#include "VoicuteKWS.h"
#include "kws/kws_postprocess.h"
#include "esp_timer.h"

// esp_timer_get_time(): int64 µs, no wraparound (millis() wraps at 49 days).
// Available in the arduino-esp32 core — it is IDF underneath.
static inline int64_t nowMs() { return esp_timer_get_time() / 1000LL; }

VoicuteKWS::VoicuteKWS() {
    for (int i = 0; i < MAX_WAKE_WORDS; i++) { m_bind[i] = {this, i}; }
}

VoicuteKWS::~VoicuteKWS() {
    delete[] m_hist;
    delete[] m_frame;
}

bool VoicuteKWS::beginEmbedded(const uint8_t* modelTflite, size_t len,
                               const char* wakeWord) {
    return beginEmbedded(modelTflite, len, wakeWord, nullptr);
}

bool VoicuteKWS::beginEmbedded(const uint8_t* modelTflite, size_t len,
                               const char* wakeWord,
                               kws_postprocess_fn postprocess) {
    return beginEmbedded(modelTflite, len, wakeWord, postprocess, Config());
}

bool VoicuteKWS::beginEmbedded(const uint8_t* modelTflite, size_t len,
                               const char* wakeWord,
                               kws_postprocess_fn postprocess,
                               const Config& cfg) {
    return init(nullptr, modelTflite, len, wakeWord, postprocess, cfg);
}

bool VoicuteKWS::beginSpiffs(const char* basePath,
                             kws_postprocess_fn postprocess) {
    return beginSpiffs(basePath, postprocess, Config());
}

bool VoicuteKWS::beginSpiffs(const char* basePath,
                             kws_postprocess_fn postprocess,
                             const Config& cfg) {
    return init(basePath, nullptr, 0, nullptr, postprocess, cfg);
}

bool VoicuteKWS::init(const char* basePath, const uint8_t* embeddedModel,
                      size_t embeddedLen, const char* wakeWord,
                      kws_postprocess_fn postprocess, const Config& cfg) {
    m_cfg = cfg;
    recognizer_config_t rc = {};
    if (basePath) strncpy(rc.model_path, basePath, sizeof(rc.model_path) - 1);
    rc.threshold   = cfg.threshold;
    rc.l1_enabled  = cfg.l1;
    rc.l2_enabled  = cfg.l2;
    rc.l3_enabled  = cfg.l3;
    rc.l4_enabled  = cfg.l4;
    rc.l5_enabled  = cfg.l5;
    rc.l5_delta    = cfg.l5Delta;
    rc.postprocess = postprocess ? postprocess : sDefaultPostprocess;
    rc.compiled_model = embeddedModel;
    rc.compiled_len   = embeddedLen;
    rc.wake_word      = wakeWord;
    recognizer_start(&rc);
    if (recognizer_num_models() == 0) {
        Serial.println("[VoicuteKWS] ERROR: no models loaded");
        return false;
    }

    // Rolling window + contiguous scratch for the SDK call
    delete[] m_hist; delete[] m_frame;
    m_hist  = new int16_t[MEL_AUDIO_LEN];
    m_frame = new int16_t[MEL_AUDIO_LEN];
    if (!m_hist || !m_frame) { Serial.println("[VoicuteKWS] OOM"); return false; }
    memset(m_hist, 0, MEL_AUDIO_LEN * sizeof(int16_t));
    m_wr = m_total = 0;
    m_sinceInfer = 0;
    m_ready = true;

    Serial.printf("[VoicuteKWS] ready: %d model(s), window=%u samples (%.2f s), "
                  "hop=%u, thr=%.2f\n",
                  recognizer_num_models(), (unsigned)MEL_AUDIO_LEN,
                  MEL_AUDIO_LEN / 16000.0f, (unsigned)cfg.hopSamples, cfg.threshold);
    for (int i = 0; i < recognizer_num_models(); i++)
        Serial.printf("  [%d] %s\n", i, recognizer_model_word(i));
    return true;
}

bool VoicuteKWS::addEmbedded(const uint8_t* modelTflite, size_t len,
                             const char* wakeWord) {
    if (!m_ready) {
        Serial.println("[VoicuteKWS] ERROR: addEmbedded() before begin*");
        return false;
    }
    if (recognizer_add_model(modelTflite, len, wakeWord) < 0) {
        Serial.println("[VoicuteKWS] ERROR: addEmbedded() failed");
        return false;
    }
    // If onWake() already ran, bind the new model too (same trampoline).
    if (m_cb)
        for (int i = 0; i < recognizer_num_models(); i++)
            recognizer_register_callback(i, sEventTrampoline, &m_bind[i]);
    int i = recognizer_num_models() - 1;
    Serial.printf("[VoicuteKWS] added model [%d] %s\n", i, recognizer_model_word(i));
    return true;
}

void VoicuteKWS::onWake(WakeCallback cb, void* user) {
    m_cb = cb; m_cbUser = user;
    for (int i = 0; i < recognizer_num_models(); i++)
        recognizer_register_callback(i, sEventTrampoline, &m_bind[i]);
}

void VoicuteKWS::sEventTrampoline(voice_event_t ev, voice_evt_data_t data, void* arg) {
    if (ev != VOICE_EVT_AWAKEN) return;
    Bind* b = (Bind*)arg;
    if (b && b->self && b->self->m_cb)
        b->self->m_cb(b->idx, recognizer_model_word(b->idx), b->self->m_cbUser);
}

float VoicuteKWS::sDefaultPostprocess(const int8_t* out, float scale, int zero, int idx) {
    (void)idx;
    return kws_postprocess(out, scale, zero);  // single-head MultiProto head
}

int VoicuteKWS::numModels() { return recognizer_num_models(); }
const char* VoicuteKWS::modelWord(int idx) { return recognizer_model_word(idx); }

void VoicuteKWS::feed(const int16_t* pcm, size_t n) {
    if (!m_ready || !pcm || n == 0) return;
    for (size_t i = 0; i < n; i++) {
        m_hist[m_wr++] = pcm[i];
        if (m_wr == MEL_AUDIO_LEN) m_wr = 0;
    }
    m_total += n;
    m_sinceInfer += n;
    maybeInfer();
}

void VoicuteKWS::maybeInfer() {
    // First inference as soon as the window is full, then at hop cadence.
    // Always on the NEWEST continuous window; no backlog replay.
    bool due = (m_total >= MEL_AUDIO_LEN) &&
               (m_sinceInfer >= m_cfg.hopSamples || m_total == MEL_AUDIO_LEN);
    if (!due) return;
    m_sinceInfer = 0;

    size_t tail = MEL_AUDIO_LEN - m_wr;
    memcpy(m_frame, m_hist + m_wr, tail * sizeof(int16_t));
    if (m_wr > 0) memcpy(m_frame + tail, m_hist, m_wr * sizeof(int16_t));

    int64_t sq = 0;
    for (size_t i = 0; i < MEL_AUDIO_LEN; i++) sq += (int64_t)m_frame[i] * m_frame[i];
    float rms = sqrtf((float)(sq / MEL_AUDIO_LEN));

    recognizer_run_frame(m_frame, rms, nowMs());
}
