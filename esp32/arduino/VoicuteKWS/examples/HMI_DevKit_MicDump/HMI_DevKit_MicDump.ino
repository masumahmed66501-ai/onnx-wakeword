/*
 * HMI_DevKit_MicDump — live mic level monitor.
 * Official es7210 driver (vendored, non-TDM mic1+mic2), max gain 37.5 dB.
 * Prints slot0 rms every 100 ms so a human voice near the mics shows
 * up immediately in the log — separates capture health from playback
 * timing uncertainty.
 */

#include <ESP_I2S.h>
#include <Wire.h>
#include "es7210_adc.h"

#define PIN_I2C_SCL   10
#define PIN_I2C_SDA   11
#define PIN_I2S_MCLK  12
#define PIN_I2S_SCLK  13
#define PIN_I2S_LRCK  14
#define PIN_I2S_SDIN  15

static int ctrl_write_reg(const audio_codec_ctrl_if_t *ctrl, int reg,
                          int reg_len, void *data, int data_len) {
    (void)ctrl; (void)reg_len; (void)data_len;
    Wire.beginTransmission(0x40);
    Wire.write((uint8_t)reg);
    Wire.write(*(uint8_t *)data);
    return Wire.endTransmission() == 0 ? 0 : -1;
}
static int ctrl_read_reg(const audio_codec_ctrl_if_t *ctrl, int reg,
                         int reg_len, void *data, int data_len) {
    (void)ctrl; (void)reg_len; (void)data_len;
    Wire.beginTransmission(0x40);
    Wire.write((uint8_t)reg);
    if (Wire.endTransmission(false) != 0) return -1;
    if (Wire.requestFrom(0x40, (uint8_t)1) != 1) return -1;
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
    return codec;
}

I2SClass i2s;

void setup() {
    Serial.begin(921600);
    delay(2000);
    Serial.println("micmon v1 official driver, mic1+2, 37.5 dB");
    Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL, 400000);

    i2s.setPins(PIN_I2S_SCLK, PIN_I2S_LRCK, -1, PIN_I2S_SDIN, PIN_I2S_MCLK);
    if (!i2s.begin(I2S_MODE_STD, 16000, I2S_DATA_BIT_WIDTH_32BIT,
                   I2S_SLOT_MODE_STEREO)) {
        Serial.println("FATAL: I2S begin failed");
        while (true) delay(1000);
    }
    if (!es7210_setup(ES7210_SEL_MIC1 | ES7210_SEL_MIC2, 37.5)) {
        while (true) delay(1000);
    }
    // read every register back — catch stuck mute/automute/divider bits
    Serial.println("register dump:");
    for (int r = 0; r <= 0x4E; r++) {
        uint8_t v = 0;
        int ret = ctrl_read_reg(NULL, r, 1, &v, 1);
        Serial.printf("%02x=%02x%c", r, ret == 0 ? v : 0xEE,
                      (r % 8) == 7 ? '\n' : ' ');
    }
    Serial.println();
    Serial.println("speak near the mics now");
}

void loop() {
    static int32_t raw[256];
    int64_t sq0 = 0, sq1 = 0, sq2 = 0;
    int32_t n0 = 0;
    uint32_t t0 = millis();
    while (millis() - t0 < 100) {
        size_t n = i2s.readBytes((char*)raw, sizeof(raw)) / sizeof(int32_t);
        if (n < 2) continue;
        for (size_t j = 0; j + 1 < n; j += 2) {
            int16_t s0 = (int16_t)(raw[j] >> 16);      // mic1 (L hi16)
            int16_t s2 = (int16_t)(raw[j + 1] >> 16);  // mic2 (R hi16)
            int16_t s1 = (int16_t)(raw[j]);            // lo16 garbage lane
            sq0 += (int64_t)s0 * s0;
            sq2 += (int64_t)s2 * s2;
            sq1 += (int64_t)s1 * s1;
            n0++;
        }
    }
    if (n0)
        Serial.printf("rms L=%ld R=%ld lo=%ld\n",
                      (long)sqrtf((float)(sq0 / n0)),
                      (long)sqrtf((float)(sq2 / n0)),
                      (long)sqrtf((float)(sq1 / n0)));
}
