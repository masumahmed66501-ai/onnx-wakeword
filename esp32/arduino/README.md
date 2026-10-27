# Arduino 封装 (Voicute)

ESP32-S3 唤醒词推理的 Arduino 库，封装 `esp32/idf/voicute` 推理 SDK。

- 库本体: [Voicute/](Voicute/)（含 vendored TFLM/esp-nn/esp-dsp）
- 设计与喂料契约: 见 [Voicute/README.md](Voicute/README.md)

## 快速开始（验证顺序）

1. Arduino IDE / arduino-cli 装 esp32 core (3.x)，板型 ESP32S3 Dev Module
2. 打开 `Voicute/examples/Doubaodoubao_Synthetic`，编译烧录 —— 无麦即可，
   串口 115200 看 prob/mel/invoke 耗时，验证全链路
3. 接 INMP441（SCK=4, WS=5, SD=6, L/R=GND），跑 `INMP441_Wake` 实测唤醒

## 换自己的模型

```bash
python Voicute/tools/tflite_to_header.py 导出包/xxx.tflite model_xxx.h
# 把导出包里的 head.h 存到 sketch 目录的 voicute_head.h（库经 __has_include
# 自动优先加载它，未提供时回落内置的 heads/doubaodoubao.h）
# sketch 里换成 beginEmbedded(xxx_tflite, XXX_MODEL_LEN, "词名")
```
