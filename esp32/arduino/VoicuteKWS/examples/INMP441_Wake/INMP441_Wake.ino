/*
 * INMP441_Wake — live wake-word detection with an I2S MEMS microphone.
 *
 * Wiring (ESP32-S3 DevKit, 3.3 V):
 *   INMP441 VDD  → 3V3
 *   INMP441 GND  → GND
 *   INMP441 SCK  → GPIO 4   (BCLK)
 *   INMP441 WS   → GPIO 5   (LRCLK)
 *   INMP441 SD   → GPIO 6   (data out of the mic)
 *   INMP441 L/R  → GND      (left channel)
 *
 * The sketch shows the hardware-isolation boundary: ALL board-specific code
 * (I2S capture, gain, channel pick) lives here; the library only sees PCM.
 *
 * Requires the esp32 Arduino core 3.x (ESP_I2S). Speak the wake word within
 * ~1.5 s of a reset-free runtime; watch prob in the serial monitor (115200).
 * Native-USB boards (S3 DevKit etc.): enable "USB CDC On Boot" or `Serial`
 * maps to UART0 pins and the USB port stays silent.
 */

#include <VoicuteKWS.h>
#include <ESP_I2S.h>
#include "model_doubaodoubao.h"   // same export package as the head in src/kws/head.h

// ---- board-specific pins ----
#define MIC_SCK  4
#define MIC_WS   5
#define MIC_SD   6

VoicuteKWS kws;
I2SClass i2s;

void onWake(int idx, const char* word, void* user) {
    Serial.printf("\n>>> WAKE [%d] %s (prob=%.3f) <<<\n\n",
                  idx, word, kws.lastProb());
}

void setup() {
    Serial.begin(115200);
    delay(2000);

    // INMP441 delivers 24 msb in a 32-bit frame; read 32-bit mono and shift
    // down to int16 with +12 dB make-up gain (>>14). Lower the shift if loud
    // rooms clip (>>16 = no gain).
    i2s.setPins(MIC_SCK, MIC_WS, -1 /*out unused*/, MIC_SD);
    // Mono RX defaults to the LEFT slot — matches L/R tied to GND.
    if (!i2s.begin(I2S_MODE_STD, 16000, I2S_DATA_BIT_WIDTH_32BIT,
                   I2S_SLOT_MODE_MONO)) {
        Serial.println("FATAL: I2S begin failed — check wiring.");
        while (true) delay(1000);
    }

    if (!kws.beginEmbedded(doubaodoubao_tflite, DOUBAO_MODEL_LEN, "doubaodoubao")) {
        while (true) delay(1000);
    }
    kws.onWake(onWake);
    Serial.println("Listening... say the wake word.");
}

void loop() {
    static int32_t raw[256];            // 64 ms of audio per read
    static int16_t pcm[256];
    size_t n = i2s.readBytes((char*)raw, sizeof(raw)) / sizeof(int32_t);
    if (n == 0) { delay(2); return; }

    int64_t sq = 0;
    for (size_t i = 0; i < n; i++) {
        int32_t s = raw[i] >> 8;        // drop 8 low bits, keep sign
        s >>= 6;                        // +12 dB make-up gain (see note above)
        if (s > 32767) s = 32767; else if (s < -32768) s = -32768;
        pcm[i] = (int16_t)s;
        sq += (int64_t)pcm[i] * pcm[i];
    }
    kws.feed(pcm, n);

    float rms = sqrtf((float)(sq / n));
    static uint32_t cnt = 0;
    if (++cnt % 5 == 0)
        Serial.printf("prob=%.4f rms=%.0f mel=%.1fms invoke=%.1fms\n",
                      kws.lastProb(), rms,
                      kws.lastMelMs(), kws.lastInvokeMs());
}
