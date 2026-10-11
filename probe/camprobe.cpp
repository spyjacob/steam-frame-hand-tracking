// camprobe — Arcturus camera probe via the public OpenVR IVRTrackedCamera API.
// Read-only: VR_Init as a background app, query camera properties, optionally
// stream N frames and dump one to PPM.  Touches nothing in the compositor.
//
//   camprobe [frames=0] [frametype=0|1|2] [dump.ppm]
#include <openvr.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <thread>
#include <vector>

using namespace vr;
static const uint32_t kMaxCams = 16;

static double nowS() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    int nFrames = argc > 1 ? atoi(argv[1]) : 0;
    auto ftype = (EVRTrackedCameraFrameType)(argc > 2 ? atoi(argv[2]) : 0);
    const char *dump = argc > 3 ? argv[3] : nullptr;

    EVRInitError ie;
    IVRSystem *sys = VR_Init(&ie, VRApplication_Background);
    if (!sys) { printf("VR_Init failed: %s\n", VR_GetVRInitErrorAsEnglishDescription(ie)); return 1; }
    IVRTrackedCamera *cam = VRTrackedCamera();
    if (!cam) { printf("no IVRTrackedCamera interface\n"); VR_Shutdown(); return 1; }

    const TrackedDeviceIndex_t hmd = k_unTrackedDeviceIndex_Hmd;
    char buf[512];
    sys->GetStringTrackedDeviceProperty(hmd, Prop_ModelNumber_String, buf, sizeof buf);
    printf("HMD model: %s\n", buf);
    bool has = false;
    EVRTrackedCameraError ce = cam->HasCamera(hmd, &has);
    printf("HasCamera: %d (%s)\n", has, cam->GetCameraErrorNameFromEnum(ce));

    ETrackedPropertyError pe;
    int32_t nCams = sys->GetInt32TrackedDeviceProperty(hmd, Prop_NumCameras_Int32, &pe);
    printf("NumCameras: %d (pe=%d)\n", nCams, pe);
    printf("FrameLayout: 0x%x (pe=%d)\n", sys->GetInt32TrackedDeviceProperty(hmd, Prop_CameraFrameLayout_Int32, &pe), pe);
    printf("StreamFormat: %d (pe=%d)\n", sys->GetInt32TrackedDeviceProperty(hmd, Prop_CameraStreamFormat_Int32, &pe), pe);
    sys->GetStringTrackedDeviceProperty(hmd, Prop_CameraFirmwareDescription_String, buf, sizeof buf, &pe);
    printf("CameraFirmware: %s (pe=%d)\n", buf, pe);

    HmdMatrix34_t xf[kMaxCams];
    uint32_t n = sys->GetArrayTrackedDeviceProperty(hmd, Prop_CameraToHeadTransforms_Matrix34_Array,
                                                    k_unHmdMatrix34PropertyTag, xf, sizeof xf, &pe);
    printf("CameraToHead: %u bytes (pe=%d)\n", n, pe);
    for (uint32_t c = 0; c < n / sizeof(HmdMatrix34_t); c++) {
        printf("  cam%u:", c);
        for (int r = 0; r < 3; r++) printf(" [%.4f %.4f %.4f %.4f]", xf[c].m[r][0], xf[c].m[r][1], xf[c].m[r][2], xf[c].m[r][3]);
        printf("\n");
    }
    int32_t dfn[kMaxCams];
    n = sys->GetArrayTrackedDeviceProperty(hmd, Prop_CameraDistortionFunction_Int32_Array, k_unInt32PropertyTag, dfn, sizeof dfn, &pe);
    printf("DistortionFunction: %u bytes (pe=%d):", n, pe);
    for (uint32_t c = 0; c < n / 4; c++) printf(" %d", dfn[c]);
    printf("\n");
    double dco[kMaxCams][k_unMaxDistortionFunctionParameters];
    n = sys->GetArrayTrackedDeviceProperty(hmd, Prop_CameraDistortionCoefficients_Float_Array, k_unDoublePropertyTag, dco, sizeof dco, &pe);
    printf("DistortionCoeffs: %u bytes (pe=%d)\n", n, pe);
    for (uint32_t c = 0; c < n / sizeof dco[0]; c++) {
        printf("  cam%u:", c);
        for (uint32_t k = 0; k < k_unMaxDistortionFunctionParameters; k++) printf(" %.6g", dco[c][k]);
        printf("\n");
    }

    for (int t = 0; t < MAX_CAMERA_FRAME_TYPES; t++) {
        uint32_t w = 0, h = 0, sz = 0;
        ce = cam->GetCameraFrameSize(hmd, (EVRTrackedCameraFrameType)t, &w, &h, &sz);
        printf("FrameType %d: %ux%u buf=%u (%s)\n", t, w, h, sz, cam->GetCameraErrorNameFromEnum(ce));
        for (int c = 0; c < (nCams > 0 ? nCams : 2); c++) {
            HmdVector2_t f{}, cc{};
            ce = cam->GetCameraIntrinsics(hmd, c, (EVRTrackedCameraFrameType)t, &f, &cc);
            printf("  intr cam%d: f=(%.3f,%.3f) c=(%.3f,%.3f) (%s)\n", c, f.v[0], f.v[1], cc.v[0], cc.v[1], cam->GetCameraErrorNameFromEnum(ce));
        }
    }

    if (nFrames > 0) {
        uint32_t w = 0, h = 0, sz = 0;
        cam->GetCameraFrameSize(hmd, ftype, &w, &h, &sz);
        TrackedCameraHandle_t hc = INVALID_TRACKED_CAMERA_HANDLE;
        ce = cam->AcquireVideoStreamingService(hmd, &hc);
        printf("AcquireVideoStreamingService: %s\n", cam->GetCameraErrorNameFromEnum(ce));
        if (ce == VRTrackedCameraError_None && sz) {
            std::vector<uint8_t> fb(sz);
            uint32_t lastSeq = 0, got = 0;
            double t0 = nowS(), tFirst = 0, tLast = 0;
            while (got < (uint32_t)nFrames && nowS() - t0 < 15) {
                CameraVideoStreamFrameHeader_t hdr{};
                ce = cam->GetVideoStreamFrameBuffer(hc, ftype, fb.data(), sz, &hdr, sizeof hdr);
                if (ce == VRTrackedCameraError_None && hdr.nFrameSequence != lastSeq) {
                    double t = nowS();
                    if (!got) tFirst = t;
                    tLast = t;
                    if (got < 5 || got % 30 == 0)
                        printf("  frame seq=%u %ux%u bpp=%u type=%d exposureTime=%llu poseValid=%d t=%.3f\n",
                               hdr.nFrameSequence, hdr.nWidth, hdr.nHeight, hdr.nBytesPerPixel, hdr.eFrameType,
                               (unsigned long long)hdr.ulFrameExposureTime, hdr.trackedDevicePose.bPoseIsValid, t - t0);
                    lastSeq = hdr.nFrameSequence;
                    got++;
                    if (dump && got == 10) {
                        FILE *f = fopen(dump, "wb");
                        fprintf(f, "P6\n%u %u\n255\n", hdr.nWidth, hdr.nHeight);
                        for (uint32_t i = 0; i < hdr.nWidth * hdr.nHeight; i++) fwrite(&fb[i * hdr.nBytesPerPixel], 1, 3, f);
                        fclose(f);
                        printf("  dumped %s\n", dump);
                    }
                } else if (ce != VRTrackedCameraError_None && ce != VRTrackedCameraError_NoFrameAvailable) {
                    printf("  GetVideoStreamFrameBuffer: %s\n", cam->GetCameraErrorNameFromEnum(ce));
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            if (got > 1) printf("received %u frames, %.2f fps\n", got, (got - 1) / (tLast - tFirst));
            else printf("received %u frames\n", got);
            cam->ReleaseVideoStreamingService(hc);
        }
    }
    VR_Shutdown();
    return 0;
}
