// Minimal Arduino shim so the vendored official es7210.c compiles here.
// Everything es7210.c actually needs lives in esp_codec_dev_types.h /
// es_common.h; this header only bridges the two names it includes.
#ifndef _ESP_CODEC_DEV_DEFAULTS_SHIM_H_
#define _ESP_CODEC_DEV_DEFAULTS_SHIM_H_
#include "esp_codec_dev_types.h"
#endif
