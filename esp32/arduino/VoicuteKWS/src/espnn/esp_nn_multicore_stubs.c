/* Single-core stubs for the esp-nn multicore hooks.
 *
 * The vendored custom esp32s3 kernels call the dual-core hooks
 * (esp_nn_multicore.c). That file runs a worker on the APP cpu, which this
 * library cannot assume is free (the sketch runs audio/UART tasks there), so
 * it is not linked. These stubs force every dual-core branch to the plain
 * single-core path: active() = false, run() = false (worker never starts),
 * wait() = no-op, scratch() = NULL (caller falls back to its internal path).
 */
#include <stdbool.h>
#include <stddef.h>

#include "esp_nn_ansi_headers.h"
#include "esp_nn_multicore.h"

void esp_nn_dual_core_enable(void) {}

bool esp_nn_dual_core_active(void) { return false; }

bool esp_nn_dual_core_run(void (*fn)(void *), void *arg) {
    (void)fn; (void)arg;
    return false;
}

void esp_nn_dual_core_wait(void) {}

void *esp_nn_dual_core_scratch(int size) {
    (void)size;
    return NULL;
}

/* Grouped-conv repack helper (lives in the unlinked mt_split file). Returning
 * false sends the caller down its plain grouped loop — safe and rare (this
 * model has no grouped conv). */
bool esp_nn_conv_s8_grouped_repack(esp_nn_conv_s8_target_fn_t conv_fn,
                                   const data_dims_t *input_dims,
                                   const int8_t *input_data,
                                   const data_dims_t *filter_dims,
                                   const int8_t *filter_data,
                                   const int32_t *bias,
                                   const data_dims_t *output_dims,
                                   int8_t *out_data,
                                   const conv_params_t *conv_params,
                                   const quant_data_t *quant_data,
                                   void *scratch, int scratch_size,
                                   int inner_scratch_size) {
    (void)conv_fn; (void)input_dims; (void)input_data; (void)filter_dims;
    (void)filter_data; (void)bias; (void)output_dims; (void)out_data;
    (void)conv_params; (void)quant_data; (void)scratch; (void)scratch_size;
    (void)inner_scratch_size;
    return false;
}

/* The mt-split variant is documented as bit-identical to one plain ANSI call. */
void esp_nn_conv_s8_ansi_mt_split(const data_dims_t *input_dims,
                                  const int8_t *input_data,
                                  const data_dims_t *filter_dims,
                                  const int8_t *filter_data,
                                  const int32_t *bias,
                                  const data_dims_t *output_dims,
                                  int8_t *out_data,
                                  const conv_params_t *conv_params,
                                  const quant_data_t *quant_data) {
    esp_nn_conv_s8_ansi(input_dims, input_data, filter_dims, filter_data,
                        bias, output_dims, out_data, conv_params, quant_data);
}
