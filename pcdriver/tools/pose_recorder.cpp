// pose_recorder — records what SteamVR presents for the framehands virtual controllers (and the
// HMD) once per display vsync, as an application would query it (predicted to photons).
// Loads SteamVR's own openvr_api.dll; no install needed.
//
//   pose_recorder.exe OUT.csv SECONDS
// CSV: t_ns,frame,device,valid,x,y,z,qw,qx,qy,qz   (device: hmd | left | right)
#include <openvr.h>
#include <windows.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

using namespace vr;

typedef intptr_t (*PFN_Init)(EVRInitError *, EVRApplicationType, const char *);
typedef void *(*PFN_GetIf)(const char *, EVRInitError *);
typedef void (*PFN_Shutdown)();

static void quatOf(const HmdMatrix34_t &m, double q[4]) {
    double tr = m.m[0][0] + m.m[1][1] + m.m[2][2];
    if (tr > 0) {
        double s = sqrt(tr + 1.0) * 2;
        q[0] = 0.25 * s; q[1] = (m.m[2][1] - m.m[1][2]) / s; q[2] = (m.m[0][2] - m.m[2][0]) / s; q[3] = (m.m[1][0] - m.m[0][1]) / s;
    } else if (m.m[0][0] > m.m[1][1] && m.m[0][0] > m.m[2][2]) {
        double s = sqrt(1.0 + m.m[0][0] - m.m[1][1] - m.m[2][2]) * 2;
        q[0] = (m.m[2][1] - m.m[1][2]) / s; q[1] = 0.25 * s; q[2] = (m.m[0][1] + m.m[1][0]) / s; q[3] = (m.m[0][2] + m.m[2][0]) / s;
    } else if (m.m[1][1] > m.m[2][2]) {
        double s = sqrt(1.0 + m.m[1][1] - m.m[0][0] - m.m[2][2]) * 2;
        q[0] = (m.m[0][2] - m.m[2][0]) / s; q[1] = (m.m[0][1] + m.m[1][0]) / s; q[2] = 0.25 * s; q[3] = (m.m[1][2] + m.m[2][1]) / s;
    } else {
        double s = sqrt(1.0 + m.m[2][2] - m.m[0][0] - m.m[1][1]) * 2;
        q[0] = (m.m[1][0] - m.m[0][1]) / s; q[1] = (m.m[0][2] + m.m[2][0]) / s; q[2] = (m.m[1][2] + m.m[2][1]) / s; q[3] = 0.25 * s;
    }
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: pose_recorder OUT.csv SECONDS\n"); return 2; }
    HMODULE lib = LoadLibraryA("C:\\Program Files (x86)\\Steam\\steamapps\\common\\SteamVR\\bin\\win64\\openvr_api.dll");
    if (!lib) { fprintf(stderr, "cannot load openvr_api.dll\n"); return 1; }
    auto init = (PFN_Init)GetProcAddress(lib, "VR_InitInternal2");
    auto getIf = (PFN_GetIf)GetProcAddress(lib, "VR_GetGenericInterface");
    auto shut = (PFN_Shutdown)GetProcAddress(lib, "VR_ShutdownInternal");
    EVRInitError e;
    init(&e, VRApplication_Background, nullptr);
    if (e != VRInitError_None) { fprintf(stderr, "VR init failed %d\n", e); return 1; }
    auto *sys = (IVRSystem *)getIf(IVRSystem_Version, &e);
    if (!sys) { fprintf(stderr, "no IVRSystem %d\n", e); return 1; }

    int dev[2] = {-1, -1};
    for (uint32_t i = 0; i < k_unMaxTrackedDeviceCount; i++) {
        char ser[64] = {};
        sys->GetStringTrackedDeviceProperty(i, Prop_SerialNumber_String, ser, sizeof ser);
        if (!strcmp(ser, "framehands-left")) dev[0] = i;
        if (!strcmp(ser, "framehands-right")) dev[1] = i;
    }
    float hz = sys->GetFloatTrackedDeviceProperty(k_unTrackedDeviceIndex_Hmd, Prop_DisplayFrequency_Float);
    float v2p = sys->GetFloatTrackedDeviceProperty(k_unTrackedDeviceIndex_Hmd, Prop_SecondsFromVsyncToPhotons_Float);
    printf("framehands devices: left %d right %d, display %.1f Hz, vsync->photons %.1f ms\n", dev[0], dev[1], hz,
           v2p * 1000);
    FILE *f = fopen(argv[1], "w");
    fprintf(f, "t_ns,frame,device,valid,x,y,z,qw,qx,qy,qz\n");
    auto t0 = std::chrono::steady_clock::now();
    uint64_t lastFrame = 0, rows = 0;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < atof(argv[2])) {
        float since;
        uint64_t frame;
        if (!sys->GetTimeSinceLastVsync(&since, &frame) || frame == lastFrame) {
            std::this_thread::sleep_for(std::chrono::microseconds(500));
            continue;
        }
        lastFrame = frame;
        // what an application rendering this frame would ask for: pose at photon time
        float predict = (hz > 0 ? 1.0f / hz : 0.011f) - since + v2p;
        TrackedDevicePose_t poses[k_unMaxTrackedDeviceCount];
        sys->GetDeviceToAbsoluteTrackingPose(TrackingUniverseStanding, predict, poses, k_unMaxTrackedDeviceCount);
        int64_t t = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        const char *names[3] = {"hmd", "left", "right"};
        int ids[3] = {0, dev[0], dev[1]};
        for (int k = 0; k < 3; k++) {
            if (ids[k] < 0) continue;
            const TrackedDevicePose_t &p = poses[ids[k]];
            double q[4];
            quatOf(p.mDeviceToAbsoluteTracking, q);
            fprintf(f, "%lld,%llu,%s,%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n", (long long)t, (unsigned long long)frame,
                    names[k], p.bPoseIsValid ? 1 : 0, p.mDeviceToAbsoluteTracking.m[0][3],
                    p.mDeviceToAbsoluteTracking.m[1][3], p.mDeviceToAbsoluteTracking.m[2][3], q[0], q[1], q[2], q[3]);
            rows++;
        }
    }
    fclose(f);
    printf("wrote %llu rows\n", (unsigned long long)rows);
    shut();
    return 0;
}
