/*
 * Doubaodoubao_Positive — positive-path smoke test, no microphone.
 *
 * Feeds a real embedded TTS utterance of the wake word ("doubaodoubao",
 * 1.69 s, 16 kHz mono) straight into Voicute, exactly like a mic would.
 * The sliding window (98 mel frames ≈ 1 s) sweeps over the utterance, so
 * prob must rise well above the noise baseline (~0.27) and ideally fire
 * WAKE while the word is inside the window.
 *
 * Pass  → model + head + mel + inference chain all correct on-device;
 *         any live failure is then in the mic/codec/I2S path.
 * Fail  → the library/head/model trio itself is broken (or the embedded
 *         model pair is mismatched).
 */

#include <Voicute.h>
#include <esp_log.h>
#include "model_doubaodoubao.h"
#include "wav_data.h"

// ESP_LOGx defaults to UART0; on a native-USB board the CDC Serial never
// sees it. Route the IDF logger into the Arduino Serial so the library's
// VOICUTE_DBG diagnostics are visible on the USB port.
static int log_to_serial(const char *fmt, va_list args) {
    char buf[256];
    vsnprintf(buf, sizeof(buf), fmt, args);
    Serial.print(buf);
    return 0;
}

Voicute kws;

void onWake(int idx, const char* word, void* user) {
    Serial.printf("\n>>> WAKE [%d] %s (prob=%.3f) <<<\n\n",
                  idx, word, kws.lastProb());
}

void setup() {
    Serial.begin(115200);
    delay(2000);
    esp_log_set_vprintf(log_to_serial);
    esp_log_level_set("*", ESP_LOG_INFO);

    Serial.println("\n=== Voicute positive-path test ===");
    Serial.printf("model: %u bytes, window: %u samples (%.2f s), wav: %u samples (%.2f s)\n",
                  (unsigned)DOUBAO_MODEL_LEN,
                  (unsigned)kws.windowSamples(), kws.windowSamples() / 16000.0f,
                  (unsigned)WAV_N, WAV_N / 16000.0f);

    if (!kws.beginEmbedded(doubaodoubao_tflite, DOUBAO_MODEL_LEN, "doubaodoubao")) {
        Serial.println("FATAL: model load failed");
        while (true) delay(1000);
    }
    kws.onWake(onWake);

    // Feed in 160-sample hops (10 ms) with real-time pacing so the hop
    // cadence matches live use; print every inference.
    const int hop = 160;
    uint32_t pos = 0, cnt = 0;
    int16_t block[hop];
    uint32_t t0 = millis();
    while (pos < WAV_N) {
        int n = min((uint32_t)hop, WAV_N - pos);
        memcpy(block, &kWav[pos], n * sizeof(int16_t));
        kws.feed(block, n);
        pos += n;
        if (++cnt % 5 == 0)  // every 50 ms
            Serial.printf("t=%4lums prob=%.4f\n", (unsigned long)(pos / 16), kws.lastProb());
        delay(10);  // ~real time
    }
    Serial.printf("done in %lu ms (wall), max prob during run printed above\n",
                  (unsigned long)(millis() - t0));
}

void loop() { delay(1000); }
