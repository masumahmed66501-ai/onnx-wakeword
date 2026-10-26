/*
 * Doubaodoubao_Synthetic — no-microphone smoke test for VoicuteKWS.
 *
 * Verifies the full inference chain (model load → mel → TFLite Invoke →
 * postprocess → gates) with synthetic PCM, no mic required:
 *   1. embedded doubaodoubao.tflite loads (schema, ops, arena)
 *   2. mel extraction runs within budget
 *   3. Invoke runs and returns sane probabilities
 *   4. per-frame latency on YOUR board (watch invoke_ms — ~57 ms on S3@240)
 *
 * Expected serial output (115200): a "frame" line every ~160 ms with
 * prob < 0.3 for noise-only input. prob stuck at 0/1 or invoke_ms way above
 * 100 means something is off (PSRAM disabled, wrong CPU freq, ...).
 *
 * Board: ESP32S3 Dev Module (or your board). Tools→CPU 240 MHz; PSRAM not
 * required by this example (all buffers fit internal RAM).
 * IMPORTANT (boards with native USB, e.g. S3 DevKitC / Dev Module):
 * enable "USB CDC On Boot" — with the default (Disabled) `Serial` maps to
 * UART0 pins and the USB port stays silent. arduino-cli:
 *   --fqbn "esp32:esp32:esp32s3:USBMode=hwcdc,CDCOnBoot=cdc"
 */

#include <VoicuteKWS.h>
#include "model_doubaodoubao.h"

VoicuteKWS kws;

// Synthetic input: 512-sample blocks (32 ms) of low-level noise. This only
// exercises the pipeline — it is NOT real speech and must NOT trigger.
int16_t block[512];
uint32_t frameCount = 0;

void setup() {
    Serial.begin(115200);
    delay(2000);  // USB-Serial-JTAG takes a moment to re-enumerate

    Serial.println("\n=== VoicuteKWS synthetic smoke test ===");
    Serial.printf("model: %u bytes, window: %u samples (%.2f s)\n",
                  (unsigned)DOUBAO_MODEL_LEN,
                  (unsigned)kws.windowSamples(), kws.windowSamples() / 16000.0f);

    if (!kws.beginEmbedded(doubaodoubao_tflite, DOUBAO_MODEL_LEN, "doubaodoubao")) {
        Serial.println("FATAL: begin failed — stopping.");
        while (true) delay(1000);
    }
    kws.onWake(onWake);
}

void onWake(int idx, const char* word, void* user) {
    // Synthetic noise should never get here; if it does, the gates are off.
    Serial.printf(">>> WAKE [%d] %s (prob=%.3f)\n", idx, word, kws.lastProb());
}

void loop() {
    // white-ish noise at modest level (rms ~ 200) — silent-room-like input
    static uint32_t seed = 12345;
    for (int i = 0; i < 512; i++) {
        seed = seed * 1103515245 + 12345;
        block[i] = (int16_t)((seed >> 16) % 400 - 200);
    }
    kws.feed(block, 512);

    frameCount++;
    if (frameCount % 5 == 0) {  // one line per ~160 ms inference
        Serial.printf("frame %6lu: prob=%.4f rms=%.0f mel=%.1fms invoke=%.1fms\n",
                      (unsigned long)frameCount,
                      kws.lastProb(), 200.0f,
                      kws.lastMelMs(), kws.lastInvokeMs());
    }
    delayMicroseconds(20000);  // ~512 samples / 32 ms of "audio time"
}
