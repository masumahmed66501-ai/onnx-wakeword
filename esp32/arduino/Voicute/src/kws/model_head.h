// Head resolution for the bundled KWS layer.
//
// The float MultiProto head and the mel window (KWS_MEL_TIME) are per-model
// constants that must come from the same export package as the .tflite
// backbone array. By default the bundled doubaodoubao (mel_time=98) head is
// used. A sketch can swap model families without touching the library by
// dropping its own `voicute_head.h` (copied from the export package) into the
// sketch folder — the angle-bracket include below resolves it there first.
#pragma once
#if __has_include(<voicute_head.h>)
#include <voicute_head.h>
#else
#include "heads/doubaodoubao.h"
#endif
