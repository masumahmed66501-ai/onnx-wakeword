/*
 * HMI_DevKit_Wake — live wake-word detection on the ESP32-S3-HMI-DevKit,
 * using its onboard microphones (4× MEMS mic through an ES7210 ADC codec).
 *
 * Board wiring (fixed by the board; same config as the ESP-IDF demo in
 * onnx-wakeword/esp32/examples/esp32s3_hmi_devkit):
 *   I2C (ES7210 control): SCL = GPIO 10, SDA = GPIO 11, addr 0x40
 *   I2S  (ES7210 data)  : MCLK = 12, SCLK/BCLK = 13, LRCK/WS = 14, DIN = 15
 *   ES7210 runs as I2S SLAVE — the ESP32 I2S peripheral is the clock master
 *   (MCLK = 16k × 256 = 4.096 MHz), so no codec clock table is needed.
 *
 * Codec init uses the OFFICIAL esp_codec_dev es7210 driver (vendored in
 * this directory), with two ordering rules learned the hard way on this
 * board (see HMI_DevKit_MicDump for the debugging harness):
 *   1. i2s.begin() FIRST, codec init SECOND — the 0x02 DLL clear needs
 *      live BCLK/MCLK or the ADC never locks (random dead capture windows).
 *   2. set_mic_gain AFTER enable(true) — es7210_start → mic_select rewrites
 *      the gain registers from the driver's cached gain (0), wiping any
 *      value set before enable (mics end up deaf at 0 dB).
 *
 * This sketch is board-specific ON PURPOSE: codec init, pins and channel
 * pick live here. The Voicute library still only sees int16 PCM.
 *
 * Requires the esp32 Arduino core 3.x. Native-USB board: enable
 * "USB CDC On Boot" or `Serial` maps to UART0 and the USB port stays silent.
 * Watch prob/rms in the serial monitor (115200) and say the wake word.
 */

#include <Voicute.h>
#include <ESP_I2S.h>
#include <Wire.h>
#include <esp_log.h>
#include <esp_heap_caps.h>
#include "es7210_adc.h"
#include "model_doubaodoubao.h"
#include "wav_data.h"   // selftest: clean TTS replay (from HMI_DevKit_Replay)

// mel debug stats (kws/mel_extractor.c)
extern float g_mel_dbg_max;
extern int   g_mel_dbg_floor_pct;
// arena placement (kws/model_loader.cpp) — DIAG
extern void *g_kws_arena;
extern int   g_kws_arena_psram;

// ESP_LOGx defaults to UART0; route the IDF logger into the USB CDC Serial
// so the library's VOICUTE_DBG diagnostics are visible on the USB port.
static int log_to_serial(const char *fmt, va_list args) {
    char buf[256];
    vsnprintf(buf, sizeof(buf), fmt, args);
    Serial.print(buf);
    return 0;
}

// ---- board-specific pins (ESP32-S3-HMI-DevKit) ----
#define PIN_I2C_SCL   10
#define PIN_I2C_SDA   11
#define PIN_I2S_MCLK  12
#define PIN_I2S_SCLK  13
#define PIN_I2S_LRCK  14
#define PIN_I2S_SDIN  15
#define ES7210_ADDR   0x40

#define MIC_GAIN_DB   30.0f  // 37.5 dB is unstable on this board (per-boot DC
                             // rail / random I2S dead windows — the loop then
                             // spins silently on empty reads); 30 dB + the
                             // ×8 digital boost below is loud and stable

// ---- ES7210 control interface over Wire (I2C addr 0x40) ----
static int ctrl_write_reg(const audio_codec_ctrl_if_t *ctrl, int reg,
                          int reg_len, void *data, int data_len) {
    (void)ctrl; (void)reg_len; (void)data_len;
    Wire.beginTransmission(ES7210_ADDR);
    Wire.write((uint8_t)reg);
    Wire.write(*(uint8_t *)data);
    return Wire.endTransmission() == 0 ? 0 : -1;
}
static int ctrl_read_reg(const audio_codec_ctrl_if_t *ctrl, int reg,
                         int reg_len, void *data, int data_len) {
    (void)ctrl; (void)reg_len; (void)data_len;
    Wire.beginTransmission(ES7210_ADDR);
    Wire.write((uint8_t)reg);
    if (Wire.endTransmission(false) != 0) return -1;
    if (Wire.requestFrom(ES7210_ADDR, (uint8_t)1) != 1) return -1;
    *(uint8_t *)data = Wire.read();
    return 0;
}
static int ctrl_open(const audio_codec_ctrl_if_t *c, void *g, int s) { (void)c;(void)g;(void)s; return 0; }
static bool ctrl_is_open(const audio_codec_ctrl_if_t *c) { (void)c; return true; }
static int ctrl_close(const audio_codec_ctrl_if_t *c) { (void)c; return 0; }

static const audio_codec_if_t *es7210_setup(uint8_t mics, float gain_db) {
    static audio_codec_ctrl_if_t ctrl;
    ctrl.open = ctrl_open;
    ctrl.is_open = ctrl_is_open;
    ctrl.read_reg = ctrl_read_reg;
    ctrl.write_reg = ctrl_write_reg;
    ctrl.close = ctrl_close;
    es7210_codec_cfg_t cfg = {};
    cfg.ctrl_if = &ctrl;
    cfg.master_mode = false;
    cfg.mic_selected = mics;
    const audio_codec_if_t *codec = es7210_codec_new(&cfg);
    if (codec == NULL) { Serial.println("FATAL: es7210 open failed"); return NULL; }
    esp_codec_dev_sample_info_t fs = {};
    fs.sample_rate = 16000;
    fs.channel = 2;
    fs.bits_per_sample = 32;
    codec->set_fs(codec, &fs);
    codec->enable(codec, true);
    // gain MUST be applied AFTER enable: es7210_start -> mic_select
    // rewrites the gain registers from the (unset) cached gain, wiping
    // any value set before enable — this was the deaf-mic root cause
    codec->set_mic_gain(codec, gain_db);

    // Read-back + retry: cheap insurance that the gain write actually
    // stuck (get_db(30.0) = GAIN_30DB = 0x0A, low nibble of the MIC1/MIC2
    // gain regs). NOTE: this was NOT the flat-prob root cause — mic level
    // was identical in good and bad boots; the real bug was the unaligned
    // PROGMEM model array (see model_doubaodoubao.h).
    const int GAIN_REG[2] = { 0x43, 0x44 };  // MIC1 / MIC2
    for (int attempt = 1; attempt <= 10; attempt++) {
        int bad = 0;
        for (int i = 0; i < 2; i++) {
            uint8_t v = 0;
            if (ctrl_read_reg(NULL, GAIN_REG[i], 1, &v, 1) != 0 ||
                (v & 0x0F) != 0x0A) {
                bad++;
                Serial.printf("gain reg 0x%02X = 0x%02X (want low nibble 0x0A), retry %d\n",
                              GAIN_REG[i], v, attempt);
            }
        }
        if (bad == 0) {
            Serial.printf("ES7210 mic gain verified 30 dB (attempt %d)\n", attempt);
            break;
        }
        codec->set_mic_gain(codec, gain_db);
        delay(10);
        if (attempt == 10) Serial.println("FATAL: ES7210 gain never stuck");
    }
    return codec;
}

Voicute kws;
I2SClass i2s;

// ---- audio feeder task (core 0) ----
// mel+invoke stall the loop for ~94ms per 160ms hop, which is past the
// I2S DMA buffer duration (~90ms): every inference window overruns the
// DMA, and the driver eventually degrades into permanent starvation
// (prob frozen, then readBytes returning nothing). A dedicated task
// pinned to the other core services the DMA every few ms regardless of
// inference, and loop() drains this ring instead of touching I2S.
#define AUD_RING_SIZE 8192  // int16 mono samples = 512 ms of slack
static int16_t aud_ring[AUD_RING_SIZE];
static volatile uint32_t aud_head = 0, aud_tail = 0;  // SPSC: core0 writes, loop reads
static volatile uint32_t aud_overruns = 0;

// Static task storage — a heap-allocated task stack collided with TFLite/esp-nn
// heap blocks (inference corrupted deterministically whenever this task ran;
// moving globals or disabling the task changed the collision). .bss removes the
// allocator from the equation.
static StackType_t aud_task_stack[2048];  // 8 KB
static StaticTask_t aud_task_tcb;

static void audio_task(void *arg) {
    static int32_t raw[128];
    for (;;) {
        size_t n = i2s.readBytes((char*)raw, sizeof(raw)) / sizeof(int32_t);
        if (n == 0) { vTaskDelay(pdMS_TO_TICKS(2)); continue; }
        for (size_t j = 0; j < n; j += 2) {
            int32_t s = raw[j] >> 16;  // LEFT slot
            int16_t v = (int16_t)s;
            uint32_t head = aud_head;
            uint32_t next = (head + 1) % AUD_RING_SIZE;
            if (next == aud_tail) {
                // full: drop the NEW sample. The producer must never write
                // aud_tail — the consumer (loop, core 1) owns it; a producer
                // write here races with the consumer's stale-tail store-back.
                aud_overruns++;
                continue;
            }
            aud_ring[head] = v;
            aud_head = next;
        }
    }
}

void onWake(int idx, const char* word, void* user) {
    Serial.printf("\n>>> WAKE [%d] %s (prob=%.3f) <<<\n\n",
                  idx, word, kws.lastProb());
}

void setup() {
    Serial.begin(115200);
    delay(2000);
    esp_log_set_vprintf(log_to_serial);
    esp_log_level_set("*", ESP_LOG_INFO);

    Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL, 400000);
    // probe: the codec must ACK its address (clear error vs driver-internal)
    Wire.beginTransmission(ES7210_ADDR);
    if (Wire.endTransmission() != 0) {
        Serial.println("FATAL: ES7210 not found on I2C (addr 0x40)");
        while (true) delay(1000);
    }

    // kws init FIRST — the 64KB TFLite arena takes the low heap addresses,
    // so the I2S DMA buffers (allocated later in i2s.begin) can never land
    // under it. When the arena was allocated AFTER i2s.begin on some builds,
    // the arena and the DMA region overlapped: invoke intermediates got
    // trashed (prob frozen at 0.055) and the afternoon's wild-DMA crashes.
    if (!kws.beginEmbedded(doubaodoubao_tflite, DOUBAO_MODEL_LEN, "doubaodoubao")) {
        while (true) delay(1000);
    }
    Serial.printf("ARENA %p psram=%d free_int=%u largest_int=%u\n",
                  g_kws_arena, g_kws_arena_psram,
                  (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                  (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));
    Serial.printf("MODEL %p mod16=%d\n", doubaodoubao_tflite,
                  (int)((uintptr_t)doubaodoubao_tflite & 15));

    // SELFTEST: replay the clean TTS wav through the engine before any
    // audio task / codec exists. PEAK ~0.99 = Invoke+mel healthy; PEAK
    // ~0.055 with a byte-correct input = backbone miscomputing (this is
    // how the unaligned model array was caught). Cheap boot-time sanity
    // gate — keep it.
    {
        float peak = 0;
        for (size_t i = 0; i < 27000; i += 64) {
            size_t take = (27000 - i) < 64 ? (27000 - i) : 64;
            kws.feed(kWav + i, take);
            if (kws.lastProb() > peak) peak = kws.lastProb();
        }
        Serial.printf("SELFTEST PEAK=%.4f\n", peak);
    }

    // ESP32 I2S is the master; it generates MCLK for the codec.
    // I2S MUST run before codec init — the ES7210 DLL clear (reg 0x02)
    // needs live BCLK/MCLK or the ADC never locks (root cause of the
    // random dead capture windows seen during bring-up).
    i2s.setPins(PIN_I2S_SCLK, PIN_I2S_LRCK, -1 /*dout unused*/, PIN_I2S_SDIN,
                PIN_I2S_MCLK);
    // 16 kHz / 32-bit / stereo — same slot format as the IDF demo.
    if (!i2s.begin(I2S_MODE_STD, 16000, I2S_DATA_BIT_WIDTH_32BIT,
                   I2S_SLOT_MODE_STEREO)) {
        Serial.println("FATAL: I2S begin failed");
        while (true) delay(1000);
    }

    // mic1+mic2, non-TDM (measured on this board: the TDM/16-bit variant
    // loses the mic signal entirely — flat noise floor on both slots).
    if (!es7210_setup(ES7210_SEL_MIC1 | ES7210_SEL_MIC2, MIC_GAIN_DB)) {
        while (true) delay(1000);
    }

    // start the DMA feeder on core 0 — the loop (core 1) stalls ~94ms per
    // inference window and cannot service the ~90ms I2S DMA buffer itself
    xTaskCreateStaticPinnedToCore(audio_task, "audio", 2048, NULL, 5,
                                  aud_task_stack, &aud_task_tcb, 0);
    kws.onWake(onWake);
    Serial.println("Listening... say the wake word.");
}

void loop() {
    static int16_t pcm[64];
    // DC blocker state: y[n] = a·(y[n-1] + x[n] − x[n−1]), a≈0.9695
    // → one-pole high-pass, −3 dB ≈ 80 Hz @ 16 kHz
    const float HPF_A = 0.9695f;
    static float hpf_xprev = 0, hpf_yprev = 0;
    // drain the audio ring (filled by the core-0 feeder task)
    uint32_t tail = aud_tail;
    size_t n = 0;
    while (n < 64 && tail != aud_head) {
        pcm[n++] = aud_ring[tail];
        tail = (tail + 1) % AUD_RING_SIZE;
    }
    aud_tail = tail;
    if (n == 0) { delay(2); return; }

    // time advances once per stereo frame, so one sample per frame
    static int64_t sq = 0;
    int64_t blk = 0;
    size_t frames = n;
    for (size_t j = 0; j < frames; j++) {
        int16_t v = pcm[j];
        sq += (int64_t)v * v;
        blk += (int64_t)v * v;
        // one-pole DC blocker (~80 Hz @ 16 kHz): the ES7210 at 37.5 dB
        // gain shows a slow DC/VMID wander that rails the input; it
        // pins mel bin 0 as the window max, flooring every speech bin
        // (prob frozen at baseline). Training data has no DC, so this
        // filter is transparent for speech.
        float y = HPF_A * (hpf_yprev + (float)v - hpf_xprev);
        hpf_xprev = (float)v;
        hpf_yprev = y;
        // ×8 digital gain: far-field/speaker playback peaks at only ~±500
        // LSB after the HPF; ×8 lifts it to ~±4k without clipping. (The old
        // ×32 boost clipped LOUD close-TTS at rms ~19k into a square wave.)
        // Clamp the float BEFORE the int16 cast — the DC-wandered HPF state
        // can exceed ±32k and an out-of-range float→int16 cast is UB.
        if (y > 32767.0f) y = 32767.0f; else if (y < -32768.0f) y = -32768.0f;
        int16_t g = (int16_t)y * 8;
        if (g > 32767) g = 32767; else if (g < -32768) g = -32768;
        pcm[j] = g;
    }
    // No digital gain: at 37.5 dB analog the mic is healthy and loud
    // (TTS playback peaks at rms ~19k vs training TTS at several k);
    // the old x32 digital boost now hard-clips speech into a square
    // wave, destroying the spectrum (prob frozen at baseline 0.054).

    float rms = sqrtf((float)(blk / frames));

    kws.feed(pcm, frames);

    static uint32_t cnt = 0;
    if (++cnt % 25 == 0) {  // one line per ~100 ms
        static uint8_t sec = 0;
        if (++sec >= 10) {  // mel stats once per second
            sec = 0;
            Serial.printf("meldbg max=%.1f floor=%d%%\n",
                          g_mel_dbg_max, g_mel_dbg_floor_pct);
        }
        Serial.printf("prob=%.4f mel=%.1fms invoke=%.1fms rms=%.0f ovr=%lu\n",
                      kws.lastProb(), kws.lastMelMs(), kws.lastInvokeMs(),
                      sqrtf((float)(sq / frames)),
                      (unsigned long)aud_overruns);
        sq = 0;
    }
}
