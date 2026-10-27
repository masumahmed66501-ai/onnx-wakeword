/*
 * HMI_DevKit_RecDump — record 6 s of mic audio and dump it as raw
 * int16 LE PCM over serial, so a host can wrap it into a wav and let a
 * human hear exactly what the KWS input chain delivers (ES7210 30 dB +
 * DC blocker + x8 digital gain, identical to HMI_DevKit_WakeCmd).
 *
 * Host protocol (921600 8N1, DTR on):
 *   host -> 'r'        start a recording
 *   board -> "REC\n"   recording now (host plays the test wav inside
 *                      the 6 s window)
 *   board -> "DATA\n"  followed by 96000 raw bytes... 3 s at 16 kHz?
 *                      No: REC_SECONDS * 16000 * 2 bytes, binary
 *   board -> "DONE\n"
 */
#include <ESP_I2S.h>
#include <Wire.h>
#include "es7210_adc.h"
#include "esp_heap_caps.h"

#define PIN_I2C_SCL   10
#define PIN_I2C_SDA   11
#define PIN_I2S_MCLK  12
#define PIN_I2S_SCLK  13
#define PIN_I2S_LRCK  14
#define PIN_I2S_SDIN  15
#define MIC_GAIN_DB   30.0f
#define REC_SECONDS   6
#define REC_SAMPLES   (REC_SECONDS * 16000)
#define REC_BYTES     (REC_SAMPLES * 2)

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
  if (codec == NULL) return NULL;
  esp_codec_dev_sample_info_t fs = {};
  fs.sample_rate = 16000;
  fs.channel = 2;
  fs.bits_per_sample = 32;
  codec->set_fs(codec, &fs);
  codec->enable(codec, true);
  // gain AFTER enable — mic_select rewrites gain regs from cache
  codec->set_mic_gain(codec, gain_db);
  const int GAIN_REG[2] = { 0x43, 0x44 };  // MIC1 / MIC2
  for (int attempt = 1; attempt <= 10; attempt++) {
    int bad = 0;
    for (int i = 0; i < 2; i++) {
      uint8_t v = 0;
      if (ctrl_read_reg(NULL, GAIN_REG[i], 1, &v, 1) != 0 ||
          (v & 0x0F) != 0x0A) bad++;
    }
    if (bad == 0) return codec;
    codec->set_mic_gain(codec, gain_db);
    delay(10);
  }
  return NULL;
}

I2SClass i2s;
static int16_t *rec_buf = NULL;

void setup() {
  Serial.begin(921600);
  delay(1500);
  Serial.println("RECDUMP v1");
  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL, 400000);
  i2s.setPins(PIN_I2S_SCLK, PIN_I2S_LRCK, -1, PIN_I2S_SDIN, PIN_I2S_MCLK);
  if (!i2s.begin(I2S_MODE_STD, 16000, I2S_DATA_BIT_WIDTH_32BIT,
                 I2S_SLOT_MODE_STEREO)) {
    Serial.println("FATAL: I2S begin failed");
    while (true) delay(1000);
  }
  if (!es7210_setup(ES7210_SEL_MIC1 | ES7210_SEL_MIC2, MIC_GAIN_DB)) {
    Serial.println("FATAL: es7210 setup failed");
    while (true) delay(1000);
  }
  rec_buf = (int16_t *)heap_caps_malloc(REC_BYTES, MALLOC_CAP_SPIRAM);
  if (rec_buf == NULL)
    rec_buf = (int16_t *)malloc(REC_BYTES);
  if (rec_buf == NULL) {
    Serial.println("FATAL: rec buffer alloc failed");
    while (true) delay(1000);
  }
  Serial.println("READY");
}

void loop() {
  if (!Serial.available()) { delay(5); return; }
  char c = (char)Serial.read();
  if (c != 'r') return;
  Serial.println("REC");
  // DC blocker state, identical to WakeCmd's input chain
  const float HPF_A = 0.9695f;
  float hpf_xprev = 0, hpf_yprev = 0;
  static int32_t raw[256];
  size_t got = 0;
  while (got < (size_t)REC_SAMPLES) {
    size_t n = i2s.readBytes((char *)raw, sizeof(raw)) / sizeof(int32_t);
    if (n < 2) { delayMicroseconds(200); continue; }
    for (size_t j = 0; j + 1 < n && got < (size_t)REC_SAMPLES; j += 2) {
      int16_t v = (int16_t)(raw[j] >> 16);   // LEFT slot, same as WakeCmd
      float y = HPF_A * (hpf_yprev + (float)v - hpf_xprev);
      hpf_xprev = (float)v;
      hpf_yprev = y;
      if (y > 32767.0f) y = 32767.0f; else if (y < -32768.0f) y = -32768.0f;
      int16_t g = (int16_t)y * 8;
      if (g > 32767) g = 32767; else if (g < -32768) g = -32768;
      rec_buf[got++] = g;
    }
  }
  Serial.println("DATA");
  Serial.write((uint8_t *)rec_buf, REC_BYTES);
  Serial.println();
  Serial.println("DONE");
}
