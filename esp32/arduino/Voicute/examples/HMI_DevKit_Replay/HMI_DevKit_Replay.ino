/*
 * HMI_DevKit_Replay — diagnostic: replay two clips through the streaming
 * feed path (64-sample chunks, same cadence as the live mic loop) and
 * print the peak prob for each:
 *   1. clean TTS wav (wav_data.h)  — engine sanity, expect peak ~0.99
 *   2. the board's own live dump   — PC scores it 0.576; board says ?
 * No microphone involved.
 */

#include <Voicute.h>
#include <esp_log.h>
#include "model_doubaodoubao.h"
#include "wav_data.h"
#include "dump_data.h"

static int log_to_serial(const char *fmt, va_list args) {
    char buf[256];
    vsnprintf(buf, sizeof(buf), fmt, args);
    Serial.print(buf);
    return 0;
}

Voicute kws;

static void replay(const char* name, const int16_t* data, size_t len) {
    Serial.printf("\n--- replay %s (%u samples) ---\n", name, (unsigned)len);
    float peak = 0;
    for (size_t i = 0; i < len; i += 64) {
        size_t take = (len - i) < 64 ? (len - i) : 64;
        kws.feed(data + i, take);
        float p = kws.lastProb();
        if (p > peak) peak = p;
        if (((i / 64) % 25) == 0)
            Serial.printf("i=%u prob=%.4f\n", (unsigned)i, p);
    }
    Serial.printf(">>> %s PEAK=%.4f invoke=%.1fms\n", name, peak, kws.lastInvokeMs());
}

void setup() {
    Serial.begin(115200);
    delay(2000);
    esp_log_set_vprintf(log_to_serial);
    esp_log_level_set("*", ESP_LOG_INFO);

    if (!kws.beginEmbedded(doubaodoubao_tflite, DOUBAO_MODEL_LEN, "doubaodoubao")) {
        while (true) delay(1000);
    }
    // DIAG build: replay both the clean TTS reference and the board's own
    // live capture, one shot each (regenerate dump_data.h from a fresh
    // HMI_DevKit_Wake serial capture when needed).
    replay("clean_tts", kWav, 27000);
    replay("board_dump", dump_data, DUMP_LEN);
}

void loop() { delay(1000); }
