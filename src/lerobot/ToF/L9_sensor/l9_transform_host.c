/**
 * Host (PC) wrapper of the vl53l9-transform-c library, loaded by l9_visualizer.py through ctypes.
 *
 * The pipeline is configured like vl53l9_app.c in OUTPUT_MODE_TRANSFORM, with all the output streams enabled.
 * l9_visualizer.py builds this file together with the library sources into build/libl9transform.so.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "vl53l9_calib_utils.h"
#include "vl53l9_transform.h"

/* the status line at the end of the raw buffer is copied into this bitfield struct, the layout must match the MCU */
_Static_assert(sizeof(vl53l9_metadata_t) == 100U, "vl53l9_metadata_t layout differs from the target on this host");

#define L9T_CALIB_SIZE    (2332U)
#define L9T_RAW_SIZE_BIN2 (14842U)
#define L9T_RAW_SIZE_BIN4 (3844U)

/* returned by l9t_process() when the status line of the raw buffer is not consistent (e.g. corrupted on the uart) */
#define L9T_ERROR_INVALID_METADATA (-100)
/* consecutive frames with other context / nb_step values before accepting them as the new reference */
#define L9T_REFERENCE_CHANGE_FRAMES (5U)

typedef enum {
    L9T_OUT_DEPTH = 0,
    L9T_OUT_AMPLITUDE,
    L9T_OUT_AMBIENT,
    L9T_OUT_CONFIDENCE,
    L9T_OUT_REFLECTANCE,
    L9T_OUT_STATUS,
    L9T_NB_OUTPUTS
} l9t_output_t;

static const char *const g_output_names[L9T_NB_OUTPUTS] = { "depth",      "amplitude",   "ambient",
                                                            "confidence", "reflectance", "status" };
static const char *const g_output_formats[L9T_NB_OUTPUTS] = { "ZF32", "AF32", "IF32", "CF32", "RF32", "CU32" };

typedef struct {
    transform_t *p_transform;
    uint8_t calib[L9T_CALIB_SIZE];
    uint32_t raw_size;
    uint32_t width;
    uint32_t height;
    uint32_t binning;
    /* dynamic settings of the first valid frame, they do not change during a stream */
    bool has_reference;
    uint32_t ref_context;
    uint32_t ref_nb_step;
    uint32_t mismatch_count;
} l9t_t;

/**
 * The library computes the frame geometry from the status line of each raw buffer and indexes its buffers with it
 * (allocated on the first frame): a corrupted status line would lead to out of bounds accesses.
 */
static bool _is_metadata_valid(l9t_t *p_handle, const uint8_t *p_raw) {
    vl53l9_metadata_t meta;
    memcpy(&meta, &p_raw[p_handle->raw_size - sizeof(meta)], sizeof(meta));

    uint32_t frame_resolution = (uint32_t)meta.frame_width * meta.frame_height;
    if (((frame_resolution * 6U) + (frame_resolution / 2U) + sizeof(meta)) != p_handle->raw_size) {
        return false;
    }
    uint32_t width = meta.crop_enable ? meta.crop_x_size : meta.frame_width;
    uint32_t height = meta.crop_enable ? meta.crop_y_size : meta.frame_height;
    if ((width != p_handle->width) || (height != p_handle->height) || (meta.binning != p_handle->binning)) {
        return false;
    }
    if (meta.crop_enable && (((meta.crop_x_offset + meta.crop_x_size) > meta.frame_width) ||
                             ((meta.crop_y_offset + meta.crop_y_size) > meta.frame_height))) {
        return false;
    }

    if (p_handle->has_reference && ((meta.context != p_handle->ref_context) || (meta.nb_step != p_handle->ref_nb_step))) {
        /* a single mismatch is a corruption, a persistent one means that the reference frame was corrupted */
        if (++p_handle->mismatch_count < L9T_REFERENCE_CHANGE_FRAMES) {
            return false;
        }
        p_handle->has_reference = false;
    }
    if (!p_handle->has_reference) {
        p_handle->ref_context = meta.context;
        p_handle->ref_nb_step = meta.nb_step;
        p_handle->has_reference = true;
    }
    p_handle->mismatch_count = 0;
    return true;
}

static int _set_stream_caps(transform_t *p_transform, const char *name, const char *format, uint32_t width,
                            uint32_t height) {
    property_t format_prop = { "format", { .val.v_string = format, .tid = VTID_STRING } };
    property_t width_prop = { "width", { .val.v_uint32 = width, .tid = VTID_UINT32 } };
    property_t height_prop = { "height", { .val.v_uint32 = height, .tid = VTID_UINT32 } };

    properties_t *p_props = properties_new(3);
    properties_add(p_props, &format_prop);
    properties_add(p_props, &width_prop);
    properties_add(p_props, &height_prop);
    capabilities_t *p_caps = capabilities_new_simple(&p_props);

    int ret = transform_set_stream_capabilities(p_transform, name, p_caps);

    properties_free(p_props, NULL);
    capabilities_free(p_caps, NULL);
    return ret;
}

void l9t_destroy(l9t_t *p_handle) {
    if (p_handle == NULL) {
        return;
    }
    if (p_handle->p_transform != NULL) {
        (void)transform_finalize(p_handle->p_transform);
        (void)transform_release(p_handle->p_transform);
        vl53l9_transform_destroy(p_handle->p_transform);
    }
    free(p_handle);
}

/**
 * Create and prepare a pipeline.
 * @param p_calib calibration data received in the L9CB packet (L9T_CALIB_SIZE bytes), NULL for default calibration
 * @param raw_size size of the raw buffers that will be processed (binning 2 or 4)
 * @param pp_handle created handle
 * @return MEDIA_ERROR_NONE or a MEDIA_ERROR_* code
 */
int l9t_create(const uint8_t *p_calib, uint32_t raw_size, l9t_t **pp_handle) {
    int ret;

    if (pp_handle == NULL) {
        return MEDIA_ERROR_INVALID_PARAMETER;
    }
    *pp_handle = NULL;

    l9t_t *p_handle = calloc(1, sizeof(l9t_t));
    if (p_handle == NULL) {
        return MEDIA_ERROR_UNKNOWN;
    }
    p_handle->raw_size = raw_size;
    if (raw_size == L9T_RAW_SIZE_BIN2) {
        p_handle->width = 54U;
        p_handle->height = 42U;
        p_handle->binning = 2U;
    } else if (raw_size == L9T_RAW_SIZE_BIN4) {
        p_handle->width = 24U;
        p_handle->height = 20U;
        p_handle->binning = 4U;
    } else {
        free(p_handle);
        return MEDIA_ERROR_INVALID_PARAMETER;
    }

    p_handle->p_transform = vl53l9_transform_create();
    if (p_handle->p_transform == NULL) {
        free(p_handle);
        return MEDIA_ERROR_UNKNOWN;
    }

    ret = transform_initialize(p_handle->p_transform);
    if (ret != MEDIA_ERROR_NONE) {
        goto error;
    }

    /* NOTE: the input stream must be configured before the output ones (raw buffers are streamed as 1 line) */
    ret = _set_stream_caps(p_handle->p_transform, "raw", "3DMD", raw_size, 1U);
    if (ret != MEDIA_ERROR_NONE) {
        goto error;
    }
    for (uint32_t i = 0; i < L9T_NB_OUTPUTS; i++) {
        ret = _set_stream_caps(p_handle->p_transform, g_output_names[i], g_output_formats[i], p_handle->width,
                               p_handle->height);
        if (ret != MEDIA_ERROR_NONE) {
            goto error;
        }
    }

    /* without calibration buffer, prepare() applies the default calibration of the library */
    if (p_calib != NULL) {
        memcpy(p_handle->calib, p_calib, L9T_CALIB_SIZE);
        ret = transform_set_control(p_handle->p_transform, "calib-buffer",
                                    (value_t){ .val.v_ptr = p_handle->calib, .tid = VTID_POINTER });
        if (ret != MEDIA_ERROR_NONE) {
            goto error;
        }
    }

    ret = transform_prepare(p_handle->p_transform);
    if (ret != MEDIA_ERROR_NONE) {
        goto error;
    }

    *pp_handle = p_handle;
    return MEDIA_ERROR_NONE;

error:
    l9t_destroy(p_handle);
    return ret;
}

void l9t_get_output_size(const l9t_t *p_handle, uint32_t *p_width, uint32_t *p_height) {
    *p_width = p_handle->width;
    *p_height = p_handle->height;
}

/**
 * Process one raw buffer, frames must be provided in order (temporal noise reduction keeps a state).
 * Float outputs hold width * height values, status holds width * height uint32 (0 = valid pixel).
 * @return MEDIA_ERROR_NONE, L9T_ERROR_INVALID_METADATA (frame skipped, outputs unchanged) or a MEDIA_ERROR_* code
 */
int l9t_process(l9t_t *p_handle, const uint8_t *p_raw, uint32_t raw_size, float *p_depth, float *p_amplitude,
                float *p_ambient, float *p_confidence, float *p_reflectance, uint32_t *p_status) {
    if ((p_handle == NULL) || (p_raw == NULL) || (raw_size != p_handle->raw_size)) {
        return MEDIA_ERROR_INVALID_PARAMETER;
    }
    if (!_is_metadata_valid(p_handle, p_raw)) {
        return L9T_ERROR_INVALID_METADATA;
    }

    size_t resolution = (size_t)p_handle->width * p_handle->height;
    void *p_outputs[L9T_NB_OUTPUTS] = { p_depth, p_amplitude, p_ambient, p_confidence, p_reflectance, p_status };
    size_t output_sizes[L9T_NB_OUTPUTS] = { resolution * sizeof(float),    resolution * sizeof(float),
                                            resolution * sizeof(float),    resolution * sizeof(float),
                                            resolution * sizeof(float),    resolution * sizeof(uint32_t) };

    /* one memory per stream: raw input first, then the outputs */
    memory_t memory[1U + L9T_NB_OUTPUTS];
    memories_t memories[1U + L9T_NB_OUTPUTS];
    stream_buffer_t stream_buffer[1U + L9T_NB_OUTPUTS];

    /* NOTE: the library only reads the input buffer, the const qualifier is dropped by the media-c memory type */
    memory[0] = (memory_t){ .data = (uint8_t *)p_raw, .size = raw_size, .maxsize = raw_size };
    stream_buffer[0].name = "raw";
    for (uint32_t i = 0; i < L9T_NB_OUTPUTS; i++) {
        if (p_outputs[i] == NULL) {
            return MEDIA_ERROR_INVALID_PARAMETER;
        }
        memory[1U + i] = (memory_t){ .data = p_outputs[i], .size = output_sizes[i], .maxsize = output_sizes[i] };
        stream_buffer[1U + i].name = g_output_names[i];
    }
    for (uint32_t i = 0; i < (1U + L9T_NB_OUTPUTS); i++) {
        memories[i] = (memories_t){ .items = &memory[i], .size = 1, .capacity = 1, .item_size = sizeof(memory_t) };
        stream_buffer[i].buffer = (buffer_t){ .memories = &memories[i] };
    }
    stream_buffers_t stream_buffers = { .items = stream_buffer,
                                        .size = 1U + L9T_NB_OUTPUTS,
                                        .capacity = 1U + L9T_NB_OUTPUTS,
                                        .item_size = sizeof(stream_buffer_t) };

    return transform_process_stream(p_handle->p_transform, &stream_buffers);
}
