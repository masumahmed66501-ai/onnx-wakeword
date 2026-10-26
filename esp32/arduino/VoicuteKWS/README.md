# VoicuteKWS (Arduino)

Voicute 唤醒词推理库的 Arduino 封装（ESP32-S3）。与 ESP-IDF SDK
（`esp32/sdk/voicute`）同一条推理管线：mel 前端 → INT8 TFLite Micro backbone
→ MultiProto float 头 → L1–L5 检测门控。

## 硬件隔离契约

库只认音频，不认板子：

- **输入**: `feed(pcm, n)` — int16 / 16 kHz / 单声道 / **原始幅度（不要除以 32768）**
- **输出**: `onWake` 回调 `(modelIdx, word, userData)`
- 麦克风、codec、LED、状态机全部留在 sketch 层

## 模型供给（二选一）

1. **`beginEmbedded(model, len, "词名")`** — 模型以 C 数组编进固件（推荐起步，
   无需文件系统）。需配合 `src/kws/head.h`。
2. **`beginSpiffs("/spiffs", post)`** — 扫描已挂载目录下所有 `*.tflite`
   （多词场景）。SPIFFS/LittleFS 由 sketch 负责挂载。

**换模型 = 换两件套**: `src/kws/head.h`（导出包里的 float 头，决定
`KWS_MEL_TIME` 窗长）+ sketch 里的 `.tflite` 数组（或 SPIFFS 文件）。
两者必须来自同一个导出包。

## 多模型：唤醒词 + 命令词（addEmbedded / setActive）

beginEmbedded 之后可继续 addEmbedded(model, len, "词名") 追加模型（上限
MAX_WAKE_WORDS）。所有模型共享同一个 mel 前端（同窗长/同量化），由 Config
里唯一的 postprocess 回调统一分发，参数 model_idx 区分模型：

```cpp
static float dispatch(const int8_t* out, float scale, int zero, int idx) {
    if (idx == IDX_CMD) { /* 多分类头: 填 probs[N], 返回 best prob */ }
    return kws_postprocess(out, scale, zero);   // 单头唤醒
}
kws.beginEmbedded(wake_tflite, WAKE_LEN, "hallo_lumo", dispatch, cfg);
kws.addEmbedded(cmd_tflite, CMD_LEN, "licht_multi");  // 继承同一 dispatch
kws.setActive(IDX_CMD, false);  // 门控: 未激活的模型跳过 Invoke
kws.onWake(onWake);
```

- **setActive(idx, bool)**: 按模型门控推理。唤醒词→命令词两段式状态机靠它在
  两个模型间切换（完整参考 examples/HMI_DevKit_WakeCmd）。
- **resetSmoothing()**: 模型切换后清空后验平滑历史，避免上一状态的残窗误触发。
- postprocess 回调由**第一次 begin*** 固定，addEmbedded 继承之；多分类头的
  proto/FC 常量来自导出包 head.h，在回调里自己算（见示例 voicute_head.h）。
- 命令窗口/超时等状态机逻辑留在 sketch 层（CMD_TIMEOUT_MS），库只管推理与回调。

### sketch 目录覆盖 head（多模型合并头）

多模型场景把合并后的头（如 voicute_head.h）放进 sketch 目录即可覆盖库内
src/kws/head.h：库用 __has_include(<voicute_head.h>) 优先探测 sketch 目录。
**注意**：Windows 下同一头文件经引号/尖括号两种 include 拼写会被视为两个
不同文件，#pragma once 会被穿透 —— 合并头必须同时带 classic guard：
#ifndef VOICUTE_HEAD_H / #define VOICUTE_HEAD_H / #endif。

## 最小示例

```cpp
#include <VoicuteKWS.h>
#include "model_doubaodoubao.h"   // uint8_t doubaodoubao_tflite[] PROGMEM

VoicuteKWS kws;

void setup() {
  Serial.begin(115200);
  kws.beginEmbedded(doubaodoubao_tflite, DOUBAO_MODEL_LEN, "doubaodoubao");
  kws.onWake([](int idx, const char* word, void* u) {
    Serial.printf(">>> WAKE %s\n", word);
  });
}

void loop() {
  int16_t pcm[256];                 // 来自 I2S 麦克风, 16 kHz 单声道 int16
  size_t n = micRead(pcm, 256);
  kws.feed(pcm, n);                 // 库内部管滚动窗口 + 160ms 节流 + RMS
}
```

## 示例

| Sketch | 说明 |
|--------|------|
| `Doubaodoubao_Synthetic` | 无麦冒烟测试：合成噪声走全链路，验证模型加载/mel/Invoke/耗时（第一步先跑这个） |
| `HMI_DevKit_Wake` | **ESP32-S3-HMI-DevKit 板载麦实时唤醒**（ES7210 codec 寄存器级初始化，参数与 IDF demo 一致） |
| HMI_DevKit_WakeCmd | **唤醒词→命令词两段式 Demo**（hallo_lumo 唤醒 + licht_multi 四色命令词，WS2812 LED 变色，命令窗口 15s，板上实测 rot=1.000 / blau=0.972） |
| INMP441_Wake | 外接 I2S 数字麦实时唤醒（INMP441 接线见 sketch 注释） |

> **原生 USB 板卡必读**（ESP32-S3 DevKit 等）：IDE 里把 **USB CDC On Boot**
> 设为 Enabled，否则 `Serial` 走 UART0 引脚，USB 串口口全程静默
> （esptool 能通但看不到任何 sketch 输出）。arduino-cli 烧录用
> `--fqbn "esp32:esp32:esp32s3:USBMode=hwcdc,CDCOnBoot=cdc"`。

## 性能参考

ESP32-S3 @240 MHz：mel + Invoke ≈ 57 ms/窗（98 帧），160 ms 节流下余量充足。
`kws.lastMelMs() / lastInvokeMs()` 可实测你的板子。

## vendored 第三方（版本与 IDF 验证版一致）

- `src/tensorflow/` + `src/third_party/` — esp-tflite-micro 1.4.1
  （放在 src 根：Arduino 只把库 `src/` 根加进 include 路径）
- `src/espnn/` — esp-nn 1.4.1（S3 SIMD 卷积）
- `src/signal/` — TFLM signal 前端（kiss_fft 子集）
- `src/dsp/` — esp-dsp 1.8.2 的 FFT 子集（mel 只用 `dsps_fft2r_fc32` 三个函数）

## vendored 构建适配补丁（相对 IDF 组件原版）

IDF 组件由 CMake 按目标筛选源文件；Arduino 会编译 `src/` 下所有文件，
因此做了以下适配（重装组件版本时需重放）：

1. **include 路径加前缀**: Arduino 只有 `src/` 根在 -I 里，espnn/kissfft
   内部的裸头文件名 include 改为 `espnn/...` / `third_party/kissfft/...` 路径。
2. **删除 RISC-V 专用源**: `espnn` 下 14 个 `*_riscv_pie.*`（ESP32-P4 PIE
   ISA 寄存器 asm），Xtensa 板卡用不到；`esp_nn.h` 对应分支改为走
   generic 优化。
3. **强制 `CONFIG_NN_OPTIMIZED`**: 该宏是 esp-nn 的 Kconfig 默认值，Arduino
   核心 sdkconfig 里不存在——不强制会静默退到 ANSI 慢内核（丢 S3 SIMD，
   57ms→数百 ms）。`VOICUTE_NN_ANSI_ONLY` 可显式退出。
4. **`TF_LITE_REMOVE_VIRTUAL_DELETE` 置空**: IDF 靠 `-fno-exceptions` 跳过
   placement-new 对私有 operator delete 的可达性检查；Arduino esp32 核心
   开着异常。该宏在 public/private 段都有展开点，无法统一翻段，而静态内存
   模式下本就无人调用 delete，置空行为等价。
5. **内核二选一**: 删除通用版 `kernels/{add,conv,depthwise_conv,
   fully_connected,mul,pooling,softmax}.cc` 与 `micro/micro_time.cc`
   （分别由 `kernels/esp_nn/*` 与 `micro/esp/micro_time.cc` 顶替，与组件
   CMake 的 REMOVE_ITEM 清单一致）。
6. **`ESP_NN=1` 注入**: 组件 CMake 用 `-DESP_NN` 启用 esp-nn 胶水层，
   Arduino 构建没有该选项——不定义时全部胶水静默走 TFLM 参考实现
   （板上实测 invoke 1231ms vs esp-nn 路径 63.8ms，差 20 倍）。
   已在各 `kernels/esp_nn/*.cc` 头部无条件 `#define ESP_NN 1`。
7. **kissfft `tools/` 补回**: `kiss_fft_wrappers/*.cc` 以
   `#include "kiss_fft.c"` 方式编译 kissfft 进 namespace，tools/ 不能裁。

## 已知差异（相对 esp32/sdk/voicute 原版）

1. `recognizer_config_t` 新增 `compiled_model / compiled_len / wake_word`
   字段 —— 嵌入式单模型模式（API.md 早已文档化，原代码未实现）。
2. `recognizer_get_last_mel_ms / _invoke_ms` 诊断接口。
3. head.h include 路径修复（原版反向 include 了 example 目录）。
   以上改动验证通过后应双向同步回 `esp32/sdk/voicute`。
