// Wire format of $XDG_RUNTIME_DIR/mrhand/cam.sock (VK_LAYER_MRHAND -> handtrack).
// On connect the layer sends one MrhandBufInfo per imported buffer, each with
// its dup'd dma-buf fd attached (SCM_RIGHTS), then a terminator with id == 0.
#pragma once
#include <stdint.h>

#define MRHAND_MAGIC 0x4d52484eu   // "MRHN"
#define MRHAND_MAX_PLANES 4

struct MrhandBufInfo {
    uint32_t magic;
    uint32_t generation;     // bumps whenever the registry changes
    uint32_t id;             // 0 = end of list
    uint32_t handleType;     // VkExternalMemoryHandleTypeFlagBits used for the import
    uint64_t size;           // dma-buf size in bytes
    uint64_t bindOffset;
    uint32_t hasImage;       // 1 if bound to a known external VkImage
    uint32_t format;         // VkFormat
    uint32_t width, height, arrayLayers;
    uint32_t tiling;         // VkImageTiling
    uint32_t usage;
    uint32_t hasModifier;
    uint64_t modifier;       // DRM format modifier
    uint32_t planeCount;
    uint32_t pad;
    uint64_t planeOffset[MRHAND_MAX_PLANES];
    uint64_t planeRowPitch[MRHAND_MAX_PLANES];
};

// /dev/shm/mrhand-pt (ptgrab stream -> handtrack): header, then `cams` luma
// frames of width*height bytes.  Per-slot seqlock: seq odd while writing.
#define MRHAND_PT_MAGIC 0x4d525054u   // "MRPT"
struct MrhandPtSlot {
    uint64_t seq;
    uint64_t tDetectNs;      // CLOCK_MONOTONIC when ptgrab saw the buffer change
    uint32_t bufferId;
    uint32_t pad;
    uint64_t frameCount;     // new frames seen for this camera (incl. skipped)
};
struct MrhandPtHeader {
    uint32_t magic, width, height, cams;
    MrhandPtSlot slot[2];
    uint64_t stat[2];        // frames seen in the current stats window
};
