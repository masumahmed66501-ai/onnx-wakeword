/*
 * HMI_DevKit_WakeCmd — wake word + follow-up voice commands on the
 * ESP32-S3-HMI-DevKit, mirroring the ESP-IDF demo in
 * onnx-wakeword/esp32/examples/esp32s3_hmi_devkit.
 *
 *   "Hallo Lumo"                 (hallo_lumo,  mt150) -> command mode, LED blinks blue
 *   "Licht rot/blau/grün/weiß"   (licht_multi, mt150) -> solid LED in the asked color
 *   6 s without a command        -> back to IDLE, LED off
 *
 * Two embedded models, ONE postprocess dispatcher (library contract:
 * kws_postprocess_fn receives the model index and dispatches per model):
 *   idx 0 (wake) -> library kws_postprocess(), bound to this package's WAKE
 *                   head via the aliases at the bottom of voicute_head.h
 *   idx 1 (cmd)  -> kws_postprocess_cmd() fills g_cmd_probs, argmax lands in
 *                   g_cmd_best; returns the best prob so the threshold gate
 *                   only fires events on a real command (IDF demo semantics).
 *
 * The mt150 models need KWS_MEL_TIME 150 — provided by the sketch-local
 * voicute_head.h, which model_head.h picks up via __has_include. If the
 * bundled doubaodoubao head (mt98) were used by mistake instead,
 * model_loader fail-fasts on the mel-window mismatch at boot.
 *
 * Board wiring / codec bring-up: identical to HMI_DevKit_Wake (see the
 * long comment there — i2s.begin BEFORE codec init, mic gain AFTER
 * enable, non-TDM 32-bit, DC blocker + ×8 digital boost, core-0 feeder
 * task). L gates are OFF (threshold-only) to match the IDF demo.
 *
 * Requires the esp32 Arduino core 3.x and "USB CDC On Boot". Serial 115200.
 */
#include <Voicute.h>
#include <ESP_I2S.h>
#include <Wire.h>
#include <esp_log.h>
#include <esp_heap_caps.h>
#include <Adafruit_NeoPixel.h>
#include "es7210_adc.h"
#include "voicute_head.h"          // mt150 heads + KWS_MEL_TIME 150 (sketch-local)
#include "kws/kws_postprocess.h"   // library float head (macros -> WAKE head)
#include "model_hallo_lumo.h"
#include "model_licht_multi.h"

// ---- board-specific pins (ESP32-S3-HMI-DevKit) ----
#define PIN_I2C_SCL   10
#define PIN_I2C_SDA   11
#define PIN_I2S_MCLK  12
#define PIN_I2S_SCLK  13
#define PIN_I2S_LRCK  14
#define PIN_I2S_SDIN  15
#define ES7210_ADDR   0x40
#define MIC_GAIN_DB   30.0f  // 37.5 dB is unstable on this board; 30 dB +
                             // ×8 digital boost below is loud and stable

Voicute kws;
I2SClass i2s;

// ---- demo models (indices = registration order) ----
#define IDX_WAKE 0   // hallo_lumo   — "Hallo Lumo"
#define IDX_CMD  1   // licht_multi  — "Licht rot/blau/grün/weiß"
static const char* CMD_NAMES[KWS_CMD_N] = {
    // u8 literals are char8_t* in C++20; char8_t is byte-sized, the cast
    // passes the UTF-8 bytes through untouched for Serial %s.
    (const char*)KWS_CMD_WORD_0, (const char*)KWS_CMD_WORD_1,
    (const char*)KWS_CMD_WORD_2, (const char*)KWS_CMD_WORD_3
};

// ---- command state machine (mirrors the IDF demo main.cpp) ----
enum state_t { STATE_IDLE, STATE_COMMAND };
static state_t g_state = STATE_IDLE;
#define CMD_TIMEOUT_MS 15000  // bench: extended so the automated speaker test fits; demo default 6000
static uint32_t g_cmd_deadline_ms = 0;
// CMD head outputs of the most recent command-model frame (postprocess
// writes them; onWake — fired later in the same feed() call — reads).
static float g_cmd_probs[KWS_CMD_N];
static int   g_cmd_best = -1;

// ---- LED (WS2812 on GPIO 38, 7 LEDs; R and G are swapped on this board) ----
#define LED_PIN   38
#define LED_COUNT 7
static Adafruit_NeoPixel strip(LED_COUNT, LED_PIN, NEO_GRB + NEO_KHZ800);
enum led_mode_t { LED_OFF, LED_BLINK, LED_SOLID };
static led_mode_t g_led_mode = LED_OFF;
static uint8_t g_led_r, g_led_g, g_led_b;
static bool g_led_on = false;
static uint32_t g_led_last_toggle = 0;
static void led_apply(uint8_t r, uint8_t g, uint8_t b) {
    // R/G swap workaround, same as the IDF demo's
    // led_strip_set_pixel(g_led, i, g, r, b): feed NeoPixel Color(r,g,b)
    // the swapped pair so the physically-wired channel lights up.
    for (int i = 0; i < LED_COUNT; i++)
        strip.setPixelColor(i, strip.Color(g, r, b));
    strip.show();
}
static void led_set_mode(led_mode_t m, uint8_t r, uint8_t g, uint8_t b) {
    g_led_mode = m; g_led_r = r; g_led_g = g; g_led_b = b;
    g_led_last_toggle = millis();
    if (m == LED_OFF) { g_led_on = false; led_apply(0, 0, 0); }
    else              { g_led_on = true;  led_apply(r, g, b); }
}
static void led_tick() {
    if (g_led_mode != LED_BLINK) return;
    if (millis() - g_led_last_toggle < 250) return;
    g_led_last_toggle = millis();
    g_led_on = !g_led_on;
    if (g_led_on) led_apply(g_led_r, g_led_g, g_led_b);
    else          led_apply(0, 0, 0);
}

// ---- postprocess dispatcher: ONE function for both models ----
static float dispatch_postprocess(const int8_t *out_i8, float out_scale,
                                  int out_zero, int model_idx) {
    if (model_idx == IDX_CMD) {
        // kws_postprocess_cmd returns the head COUNT; argmax is ours to take.
        int n = kws_postprocess_cmd(out_i8, out_scale, out_zero, g_cmd_probs);
        int best = 0;
        for (int i = 1; i < n; i++)
            if (g_cmd_probs[i] > g_cmd_probs[best]) best = i;
        g_cmd_best = best;
        if (g_cmd_probs[best] > 0.05f) {
            Serial.printf("[cmd]");
            for (int i = 0; i < n; i++) Serial.printf(" p%d=%.3f", i, g_cmd_probs[i]);
            Serial.printf("\n");
        }
        return g_cmd_probs[best];
    }
    return kws_postprocess(out_i8, out_scale, out_zero);
}

// led_off: timeout reverts to dark; a matched command keeps the requested
// color solid (same behaviour as the IDF demo).
static void leave_cmd(const char *reason, bool led_off) {
    g_state = STATE_IDLE;
    kws.setActive(IDX_CMD, false);
    kws.setActive(IDX_WAKE, true);
    // Clear smoothing so a prob peak near the transition can't re-fire
    // either model right after the switch.
    kws.resetSmoothing();
    if (led_off) led_set_mode(LED_OFF, 0, 0, 0);
    // else: keep whatever the command handler lit up
    Serial.printf("CMD -> IDLE (%s)\n", reason);
}

static void onWake(int idx, const char *word, void *user) {
    (void)word; (void)user;
    if (idx == IDX_WAKE) {
        if (g_state != STATE_IDLE) return;
        g_state = STATE_COMMAND;
        kws.setActive(IDX_WAKE, false);
        kws.setActive(IDX_CMD, true);
        kws.resetSmoothing();
        g_cmd_deadline_ms = millis() + CMD_TIMEOUT_MS;
        led_set_mode(LED_BLINK, 0, 0, 255);
        Serial.printf("\n>>> WAKE -> CMD mode (Licht rot/blau/grun/weis)\n");
    } else if (idx == IDX_CMD) {
        if (g_state != STATE_COMMAND || g_cmd_best < 0) return;
        int id = g_cmd_best;
        switch (id) {
            case 0: led_set_mode(LED_SOLID, 255, 0, 0);     break;  // rot
            case 1: led_set_mode(LED_SOLID, 0, 0, 255);     break;  // blau
            case 2: led_set_mode(LED_SOLID, 0, 255, 0);     break;  // gruen
            case 3: led_set_mode(LED_SOLID, 255, 255, 255); break;  // weiss
        }
        Serial.printf("\n*** CMD #%d: %s (prob=%.3f) ***\n\n",
                      id, CMD_NAMES[id], g_cmd_probs[id]);
        leave_cmd("matched", false);
    }
}

// ---- ES7210 control interface over Wire (I2C addr 0x40) ----
static int ctrl_write_reg(const audio_codec_ctrl_if_t *ctrl, int reg,
                          int reg_len, void *data, int data_len) {
    (void)ctrl; (void)reg_len; (void)data_len;
    Wire.beginTransmission(ES7210_ADDR);
    Wire.write((uint8_t)reg);
    Wire.write(*(uint8_t *)data);
    return Wire.endTransmission();
}
static int ctrl_read_reg(const audio_codec_ctrl_if_t *ctrl, int reg,
                         int reg_len, void *data, int data_len) {
    (void)ctrl; (void)reg_len; (void)data_len;
    if (data_len != 1) return -1;
    Wire.beginTransmission(ES7210_ADDR);
    Wire.write((uint8_t)reg);
    if (Wire.endTransmission(false) != 0) return -1;
    if (Wire.requestFrom(ES7210_ADDR, 1) != 1) return -1;
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
    // stuck (get_db(30.0) = GAIN_30DB = 0x0A, low nibble of MIC1/MIC2).
    const int GAIN_REG[2] = { 0x43, 0x44 };  // MIC1 / MIC2
    for (int attempt = 1; attempt <= 10; attempt++) {
        int bad = 0;
        for (int i = 0; i < 2; i++) {
            uint8_t v = 0;
            if (ctrl_read_reg(NULL, GAIN_REG[i], 1, &v, 1) != 0 ||
                (v & 0x0F) != 0x0A) {
                bad++;
                Serial.printf("gain reg 0x%02X = 0x%02X (want 0x0A), retry %d\n",
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


// ---- audio feeder task (core 0) ----
// mel+invoke stall the loop for ~94ms per 160ms hop, which is past the
// I2S DMA buffer duration (~90ms): a dedicated task pinned to the other
// core services the DMA every few ms regardless of inference, and loop()
// drains this ring instead of touching I2S. (Full story: HMI_DevKit_Wake.)
#define AUD_RING_SIZE 8192  // int16 mono samples = 512 ms of slack
static int16_t aud_ring[AUD_RING_SIZE];
static volatile uint32_t aud_head = 0, aud_tail = 0;  // SPSC: core0 writes, loop reads
static volatile uint32_t aud_overruns = 0;
// Static task storage — a heap-allocated task stack collided with
// TFLite/esp-nn heap blocks. .bss removes the allocator from the equation.
static StackType_t aud_task_stack[2048];  // 8 KB
static StaticTask_t aud_task_tcb;
static void audio_task(void *arg) {
    static int32_t raw[128];
    (void)arg;
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
                // aud_tail — the consumer (loop, core 1) owns it.
                aud_overruns++;
                continue;
            }
            aud_ring[head] = v;
            aud_head = next;
        }
    }
}

// mel debug stats (kws/mel_extractor.c)
extern float g_mel_dbg_max;
extern int   g_mel_dbg_floor_pct;
// arena placement (kws/model_loader.cpp) — DIAG
extern void *g_kws_arena;
extern int   g_kws_arena_psram;

// ESP_LOGx defaults to UART0; route the IDF logger into the USB CDC Serial.
static int log_to_serial(const char *fmt, va_list args) {
    char buf[256];
    vsnprintf(buf, sizeof(buf), fmt, args);
    Serial.print(buf);
    return 0;
}

void setup() {
    Serial.begin(115200);
    delay(2000);
    esp_log_set_vprintf(log_to_serial);
    esp_log_level_set("*", ESP_LOG_INFO);

    strip.begin();
    strip.show();  // all dark

    Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL, 400000);
    Wire.beginTransmission(ES7210_ADDR);
    if (Wire.endTransmission() != 0) {
        Serial.println("FATAL: ES7210 not found on I2C (addr 0x40)");
        while (true) delay(1000);
    }

    // kws init FIRST — the 64KB TFLite arena takes the low heap addresses,
    // so the I2S DMA buffers (allocated later in i2s.begin) can never land
    // under it (see HMI_DevKit_Wake).
    //
    // IDF demo detection config: threshold-only, all L gates bypassed.
    // The FIRST begin* call fixes the Config AND the postprocess function
    // for every subsequent addEmbedded() — so both models share
    // dispatch_postprocess, which switches on the model index.
    Voicute::Config cfg;
    cfg.threshold = 0.70f;
    cfg.l1 = false; cfg.l2 = false; cfg.l3 = false;
    cfg.l4 = false; cfg.l5 = false;
    if (!kws.beginEmbedded(hallo_lumo_tflite, HALLO_LUMO_TFLITE_LEN,
                           "hallo_lumo", dispatch_postprocess, cfg)) {
        Serial.println("FATAL: beginEmbedded(hallo_lumo) failed");
        while (true) delay(1000);
    }
    int n = kws.addEmbedded(licht_multi_tflite, LICHT_MULTI_TFLITE_LEN,
                            "licht_multi");
    if (n < 0) {
        Serial.println("FATAL: addEmbedded(licht_multi) failed");
        while (true) delay(1000);
    }
    Serial.printf("models: %d (wake=%d cmd=%d)\n", n, IDX_WAKE, IDX_CMD);
    Serial.printf("ARENA %p psram=%d free_int=%u largest_int=%u\n",
                  g_kws_arena, g_kws_arena_psram,
                  (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                  (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));
    Serial.printf("MODEL wake %p mod16=%d | cmd %p mod16=%d\n",
                  hallo_lumo_tflite, (int)((uintptr_t)hallo_lumo_tflite & 15),
                  licht_multi_tflite, (int)((uintptr_t)licht_multi_tflite & 15));

    // cmd model only listens after the wake word
    kws.setActive(IDX_CMD, false);
    kws.onWake(onWake);

    // ESP32 I2S is the master; it generates MCLK for the codec.
    // I2S MUST run before codec init — the ES7210 DLL clear (reg 0x02)
    // needs live BCLK/MCLK or the ADC never locks.
    i2s.setPins(PIN_I2S_SCLK, PIN_I2S_LRCK, -1 /*dout unused*/, PIN_I2S_SDIN,
                PIN_I2S_MCLK);
    // 16 kHz / 32-bit / stereo — same slot format as the IDF demo.
    if (!i2s.begin(I2S_MODE_STD, 16000, I2S_DATA_BIT_WIDTH_32BIT,
                   I2S_SLOT_MODE_STEREO)) {
        Serial.println("FATAL: I2S begin failed");
        while (true) delay(1000);
    }
    // mic1+mic2, non-TDM (the TDM/16-bit variant loses the mic signal
    // entirely on this board).
    if (!es7210_setup(ES7210_SEL_MIC1 | ES7210_SEL_MIC2, MIC_GAIN_DB)) {
        while (true) delay(1000);
    }

    // start the DMA feeder on core 0 — the loop (core 1) stalls ~94ms per
    // inference window and cannot service the ~90ms I2S DMA buffer itself
    xTaskCreateStaticPinnedToCore(audio_task, "audio", 2048, NULL, 5,
                                  aud_task_stack, &aud_task_tcb, 0);
    Serial.println("Listening... say Hallo Lumo, then Licht rot/blau/grun/weis.");
}

void loop() {
    static int16_t pcm[64];
    // DC blocker state: y[n] = a·(y[n-1] + x[n] − x[n−1]), a≈0.9695
    const float HPF_A = 0.9695f;
    static float hpf_xprev = 0, hpf_yprev = 0;

    led_tick();
    if (g_state == STATE_COMMAND && millis() > g_cmd_deadline_ms)
        leave_cmd("timeout", true);

    // drain the audio ring (filled by the core-0 feeder task)
    uint32_t tail = aud_tail;
    size_t n = 0;
    while (n < 64 && tail != aud_head) {
        pcm[n++] = aud_ring[tail];
        tail = (tail + 1) % AUD_RING_SIZE;
    }
    aud_tail = tail;
    if (n == 0) { delay(2); return; }

    static int64_t sq = 0;
    int64_t blk = 0;
    size_t frames = n;
    for (size_t j = 0; j < frames; j++) {
        int16_t v = pcm[j];
        sq += (int64_t)v * v;
        blk += (int64_t)v * v;
        // one-pole DC blocker (~80 Hz @ 16 kHz): the ES7210 shows a slow
        // DC/VMID wander that rails the input and floors every speech bin
        float y = HPF_A * (hpf_yprev + (float)v - hpf_xprev);
        hpf_xprev = (float)v;
        hpf_yprev = y;
        // ×8 digital gain; clamp BEFORE the int16 cast (out-of-range
        // float to int16 is UB, and the HPF state can exceed ±32k)
        if (y > 32767.0f) y = 32767.0f; else if (y < -32768.0f) y = -32768.0f;
        int16_t g = (int16_t)y * 8;
        if (g > 32767) g = 32767; else if (g < -32768) g = -32768;
        pcm[j] = g;
    }
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
