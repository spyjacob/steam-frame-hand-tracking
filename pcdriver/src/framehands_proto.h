// Wire format between the Frame (handtrack bridge) and the PC driver.
// All little-endian.  Hand poses are in the HMD/head frame (OpenVR convention:
// +x right, +y up, -z forward, metres).
//
// v3: the PC sends FhPing every ~100 ms (it also subscribes); the headset answers
// each with FhPong carrying its own CLOCK_MONOTONIC receive time, so the driver
// can estimate the headset->PC clock offset and its uncertainty (ping/pong, lowest
// round trips).  Hand packets carry the camera capture time in headset clock.
#pragma once
#include <stdint.h>

#define FH_MAGIC 0x444e4846u   // "FHND"
#define FH_VERSION 3u
#define FH_PORT 47811
#define FH_KIND_PING 1u        // PC -> headset: subscribe + clock ping
#define FH_KIND_PONG 2u        // headset -> PC
#define FH_MODE_HANDS 1u

#pragma pack(push, 1)
struct FhPing {
    uint32_t magic, version, kind, pad;
    uint64_t pcSendNs;    // PC steady clock
};
struct FhPong {
    uint32_t magic, version, kind, pad;
    uint64_t pcSendNs;    // echoed
    uint64_t hsRecvNs;    // headset CLOCK_MONOTONIC when the ping arrived
};
struct FhHand {
    uint32_t valid;       // 1 = pose below is current
    float confidence;     // 0..1
    float pos[3];         // palm centre, head frame, metres
    float quat[4];        // w, x, y, z: head-from-palm rotation (controller convention: -z along fingers, +y back of hand)
    float ageMs;          // capture -> send (headset clock)
    float curl[5];        // thumb, index, middle, ring, pinky: 0 open .. 1 fully curled
    float splay[5];       // -1..1 (not computed yet: 0)
};
struct FhHandPacket {
    uint32_t magic, version;
    uint64_t seq;
    uint64_t tCapNs;      // camera capture time, headset CLOCK_MONOTONIC (snapped to the 60 Hz frame grid;
                          // an estimate: ptgrab only sees a frame after XRService finished writing it)
    uint32_t mode;
    uint32_t pad;
    FhHand hand[2];       // [0] left, [1] right
};
#pragma pack(pop)
