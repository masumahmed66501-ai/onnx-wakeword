#!/usr/bin/env python3
"""tflite_to_header.py — 把导出包里的 .tflite 转成 Arduino sketch 可嵌入的 C 数组头文件。

用法:
    python tflite_to_header.py <model.tflite> [输出头文件路径] [数组名]

输出头文件放进 sketch 目录（和 .ino 同级），sketch 里:
    #include "model_xxx.h"
    kws.beginEmbedded(xxx_tflite, XXX_MODEL_LEN, "词名");
注意: 换模型时 src/kws/head.h 也要同步换成同一导出包里的 float 头。
"""
import sys
import os


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    src = sys.argv[1]
    default_name = os.path.splitext(os.path.basename(src))[0]
    out = sys.argv[2] if len(sys.argv) > 2 else f"model_{default_name}.h"
    arr = sys.argv[3] if len(sys.argv) > 3 else f"{default_name}_tflite"
    guard_name = arr.upper()

    data = open(src, "rb").read()
    lines = [
        "#pragma once",
        f"// Auto-generated from {os.path.basename(src)} ({len(data)} bytes) by tflite_to_header.py",
        "// Pairs with the matching head.h from the same Voicute export package.",
        "#include <pgmspace.h>",
        f"#define {guard_name}_LEN {len(data)}",
        "// aligned(16): TFLite Invoke miscomputes when this byte-aligned",
        "// PROGMEM array lands at unlucky flash addresses (per-binary,",
        "// deterministic backbone garbage with a byte-correct input",
        "// tensor). Keep every generated model array 16-byte aligned.",
        f"static const uint8_t {arr}[{len(data)}] PROGMEM __attribute__((aligned(16))) = {{",
    ]
    for i in range(0, len(data), 20):
        chunk = data[i:i + 20]
        lines.append("  " + ",".join(f"0x{b:02x}" for b in chunk) + ",")
    lines.append("};")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"written {out}: {arr}[{len(data)}] PROGMEM")


if __name__ == "__main__":
    main()
