// driver_framehands v3 — SteamVR driver (PC side) exposing two virtual Frame
// controllers driven by the optical hand tracking running on the Steam Frame.
// Optical hands produce POSE and SKELETON only: no button/trigger/grip/joystick
// input is ever created or pressed.
//
// Data path: headset UDP (FhHandPacket v3, hand poses in head frame + camera
// capture time in headset clock) -> clock mapping (ping/pong) -> world pose using
// the HMD pose at capture time from our own HMD pose history -> per-hand sample
// history -> presentation at every RunFrame (interpolated / limited extrapolation).
//
// Everything that affects motion is configurable at runtime from
// <driver dir>\framehands_live.ini (re-read every second; no SteamVR restart):
//   predictLinear=0|1, predictAngular=0|1, maxPredictMs, interpDelayMs, velWindowMs,
//   holdMs, maxVelocity, maxAngularVelocity, maxJumpCm, stillSeconds, stillCm, stillDeg,
//   handAwayCm, pickupCm, pickupDeg, forceHandMode=0|1, headsetHost, port, statsSeconds
//
// Mode switching, per hand, only for hands enabled on the headset panel (packet flags;
// a disabled hand always stays on its real controller): set the real controller down
// (within stillCm / stillDeg for stillSeconds) with the tracked palm seen > handAwayCm away
// from it -> that hand switches to optical tracking (virtual controller connects with high
// hand-selection priority); pick it up and move it > pickupCm / pickupDeg -> the virtual
// controller disconnects again.  The defaults are loose on purpose: an enabled hand is
// expected to be used.
#include <openvr_driver.h>

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#pragma comment(lib, "ws2_32.lib")

using namespace vr;

#include "framehands_proto.h"
#include "hand_simulation.h"   // Valve's OpenVR SDK sample (third_party/openvr_handsim, BSD-3)

static void Log(const char *fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    VRDriverLog()->Log(buf);
}

using Clock = std::chrono::steady_clock;
using Quat = HmdQuaternion_t;

static int64_t nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------- math
static Quat mul(const Quat &a, const Quat &b) {
    return {a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z, a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x, a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
}
static Quat conj(const Quat &q) { return {q.w, -q.x, -q.y, -q.z}; }
static void rotate(const Quat &q, const double v[3], double out[3]) {
    Quat r = mul(mul(q, Quat{0, v[0], v[1], v[2]}), conj(q));
    out[0] = r.x;
    out[1] = r.y;
    out[2] = r.z;
}
static double qdot(const Quat &a, const Quat &b) { return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z; }
static Quat qnorm(Quat q) {
    double n = sqrt(qdot(q, q));
    return {q.w / n, q.x / n, q.y / n, q.z / n};
}
static Quat slerp(Quat a, Quat b, double t) {
    double d = qdot(a, b);
    if (d < 0) {
        b = {-b.w, -b.x, -b.y, -b.z};
        d = -d;
    }
    if (d > 0.9995) return qnorm({a.w + t * (b.w - a.w), a.x + t * (b.x - a.x), a.y + t * (b.y - a.y), a.z + t * (b.z - a.z)});
    double th = acos(d), s = sin(th);
    double wa = sin((1 - t) * th) / s, wb = sin(t * th) / s;
    return {wa * a.w + wb * b.w, wa * a.x + wb * b.x, wa * a.y + wb * b.y, wa * a.z + wb * b.z};
}
static HmdQuaternion_t quatFromMatrix(const HmdMatrix34_t &m) {
    HmdQuaternion_t q;
    double tr = m.m[0][0] + m.m[1][1] + m.m[2][2];
    if (tr > 0) {
        double s = sqrt(tr + 1.0) * 2;
        q.w = 0.25 * s;
        q.x = (m.m[2][1] - m.m[1][2]) / s;
        q.y = (m.m[0][2] - m.m[2][0]) / s;
        q.z = (m.m[1][0] - m.m[0][1]) / s;
    } else if (m.m[0][0] > m.m[1][1] && m.m[0][0] > m.m[2][2]) {
        double s = sqrt(1.0 + m.m[0][0] - m.m[1][1] - m.m[2][2]) * 2;
        q.w = (m.m[2][1] - m.m[1][2]) / s;
        q.x = 0.25 * s;
        q.y = (m.m[0][1] + m.m[1][0]) / s;
        q.z = (m.m[0][2] + m.m[2][0]) / s;
    } else if (m.m[1][1] > m.m[2][2]) {
        double s = sqrt(1.0 + m.m[1][1] - m.m[0][0] - m.m[2][2]) * 2;
        q.w = (m.m[0][2] - m.m[2][0]) / s;
        q.x = (m.m[0][1] + m.m[1][0]) / s;
        q.y = 0.25 * s;
        q.z = (m.m[1][2] + m.m[2][1]) / s;
    } else {
        double s = sqrt(1.0 + m.m[2][2] - m.m[0][0] - m.m[1][1]) * 2;
        q.w = (m.m[1][0] - m.m[0][1]) / s;
        q.x = (m.m[0][2] + m.m[2][0]) / s;
        q.y = (m.m[1][2] + m.m[2][1]) / s;
        q.z = 0.25 * s;
    }
    return q;
}

// ---------------------------------------------------------------- live configuration
struct Config {
    bool predictLinear = true;      // report linear velocity so SteamVR extrapolates position
    bool predictAngular = false;    // report angular velocity (off: rotation measurements are 5-10 Hz)
    double maxPredictMs = 50;       // cap on how far SteamVR may extrapolate a sample
    double interpDelayMs = 0;       // present the hand state this far in the past (interpolated)
    double velWindowMs = 50;        // velocity = regression over samples within this window
    double holdMs = 250;            // newest sample older than this: hold, out of range, no velocity
    double maxVelocity = 3.0, maxAngularVelocity = 15.0, maxJumpCm = 15;
    double stillSeconds = 0.8, stillCm = 2.0, stillDeg = 6.0, handAwayCm = 10;
    double pickupCm = 4.0, pickupDeg = 15.0;
    bool forceHandMode = false;
    std::string headsetHost = "192.168.8.180";
    int port = FH_PORT;
    double statsSeconds = 10;
};
static std::mutex g_cfgM;
static Config g_cfg;
static Config cfg() {
    std::lock_guard<std::mutex> l(g_cfgM);
    return g_cfg;
}

static std::string driverDir() {
    HMODULE h = nullptr;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       (LPCSTR)&driverDir, &h);
    char p[MAX_PATH] = {};
    GetModuleFileNameA(h, p, MAX_PATH);
    std::string s(p);
    for (int i = 0; i < 3; i++) s = s.substr(0, s.find_last_of("\\/"));   // ...\framehands\bin\win64\x.dll
    return s;
}

static void loadLiveConfig(const std::string &path, bool log) {
    FILE *f = fopen(path.c_str(), "r");
    if (!f) return;
    Config c = cfg();
    char line[256];
    while (fgets(line, sizeof line, f)) {
        char k[64], v[128];
        if (line[0] == '#' || sscanf(line, " %63[^= ] = %127s", k, v) != 2) continue;
        std::string key(k);
        double d = atof(v);
        if (key == "predictLinear") c.predictLinear = d != 0;
        else if (key == "predictAngular") c.predictAngular = d != 0;
        else if (key == "maxPredictMs") c.maxPredictMs = d;
        else if (key == "interpDelayMs") c.interpDelayMs = d;
        else if (key == "velWindowMs") c.velWindowMs = d;
        else if (key == "holdMs") c.holdMs = d;
        else if (key == "maxVelocity") c.maxVelocity = d;
        else if (key == "maxAngularVelocity") c.maxAngularVelocity = d;
        else if (key == "maxJumpCm") c.maxJumpCm = d;
        else if (key == "stillSeconds") c.stillSeconds = d;
        else if (key == "handAwayCm") c.handAwayCm = d;
        else if (key == "stillCm") c.stillCm = d;
        else if (key == "stillDeg") c.stillDeg = d;
        else if (key == "pickupCm") c.pickupCm = d;
        else if (key == "pickupDeg") c.pickupDeg = d;
        else if (key == "forceHandMode") c.forceHandMode = d != 0;
        else if (key == "headsetHost") c.headsetHost = v;
        else if (key == "port") c.port = (int)d;
        else if (key == "statsSeconds") c.statsSeconds = d;
    }
    fclose(f);
    {
        std::lock_guard<std::mutex> l(g_cfgM);
        g_cfg = c;
    }
    if (log)
        Log("framehands: config predictLinear=%d predictAngular=%d maxPredictMs=%.0f interpDelayMs=%.0f velWindowMs=%.0f "
            "holdMs=%.0f maxVel=%.1f maxAngVel=%.1f maxJumpCm=%.0f still=%.1fs/%.1fcm/%.0fdeg handAway=%.0fcm "
            "pickup=%.1fcm/%.0fdeg force=%d host=%s:%d",
            c.predictLinear, c.predictAngular, c.maxPredictMs, c.interpDelayMs, c.velWindowMs, c.holdMs, c.maxVelocity,
            c.maxAngularVelocity, c.maxJumpCm, c.stillSeconds, c.stillCm, c.stillDeg, c.handAwayCm, c.pickupCm,
            c.pickupDeg, c.forceHandMode, c.headsetHost.c_str(), c.port);
}

// ---------------------------------------------------------------- clock mapping (ping/pong)
// offset = headset_clock - pc_clock.  Each pong gives offset = hsRecv - (pcSend + rtt/2), with an
// uncertainty of rtt/2 (network asymmetry is unknown).  We keep ~5 s of samples and use the median
// offset of the lowest-RTT quarter.
struct ClockSync {
    std::mutex m;
    std::deque<std::pair<int64_t, std::pair<int64_t, int64_t>>> samples;   // pcRecv, (rtt, offset)
    int64_t offset = 0, rttMin = 0;
    bool valid = false;

    void add(int64_t pcSend, int64_t pcRecv, int64_t hsRecv) {
        int64_t rtt = pcRecv - pcSend;
        if (rtt <= 0 || rtt > 200000000) return;
        std::lock_guard<std::mutex> l(m);
        samples.push_back({pcRecv, {rtt, hsRecv - (pcSend + rtt / 2)}});
        while (!samples.empty() && pcRecv - samples.front().first > 5000000000LL) samples.pop_front();
        std::vector<std::pair<int64_t, int64_t>> v;
        for (auto &s : samples) v.push_back(s.second);
        std::sort(v.begin(), v.end());
        size_t n = std::max<size_t>(1, v.size() / 4);
        std::vector<int64_t> offs;
        for (size_t i = 0; i < n; i++) offs.push_back(v[i].second);
        std::sort(offs.begin(), offs.end());
        offset = offs[offs.size() / 2];
        rttMin = v[0].first;
        valid = true;
    }
    bool hsToPc(int64_t hsNs, int64_t &pcNs) {
        std::lock_guard<std::mutex> l(m);
        if (!valid) return false;
        pcNs = hsNs - offset;
        return true;
    }
} g_clock;

// ---------------------------------------------------------------- HMD pose history
// The HMD pose at each RunFrame (GetRawTrackedDevicePoses(0)), so a hand sample can be
// placed in the world with the head pose of its own capture time.
struct HmdHistory {
    struct E { int64_t t; double p[3]; Quat q; };
    std::deque<E> h;
    void push(int64_t t, const HmdMatrix34_t &m) {
        E e{t, {m.m[0][3], m.m[1][3], m.m[2][3]}, quatFromMatrix(m)};
        h.push_back(e);
        while (h.size() > 600) h.pop_front();   // ~6 s at 90 Hz
    }
    bool at(int64_t t, double p[3], Quat &q) const {
        if (h.size() < 2 || t < h.front().t || t > h.back().t + 20000000) return false;
        if (t >= h.back().t) {   // up to 20 ms beyond the newest entry: use the newest
            for (int i = 0; i < 3; i++) p[i] = h.back().p[i];
            q = h.back().q;
            return true;
        }
        auto it = std::lower_bound(h.begin(), h.end(), t, [](const E &e, int64_t tt) { return e.t < tt; });
        const E &b = *it, &a = *(it - 1);
        double f = b.t > a.t ? double(t - a.t) / double(b.t - a.t) : 0;
        for (int i = 0; i < 3; i++) p[i] = a.p[i] + f * (b.p[i] - a.p[i]);
        q = slerp(a.q, b.q, f);
        return true;
    }
} g_hmd;

// ---------------------------------------------------------------- shared receive state
struct Shared {
    std::mutex m;
    FhHandPacket last{};
    int64_t lastRxNs = 0;
    bool have = false;
    uint64_t packets = 0, pongs = 0, badSize = 0;
} g_state;

// ---------------------------------------------------------------- statistics
struct Stats {
    std::vector<double> presentAgeMs, sampleGapMs;
    uint64_t samples = 0, presents = 0, extrap = 0, interp = 0, held = 0, hmdMiss = 0;
    void clear() { *this = Stats(); }
};
static double pctl(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, size_t(p / 100.0 * v.size()))];
}

// ---------------------------------------------------------------- device
class HandController : public ITrackedDeviceServerDriver {
public:
    explicit HandController(int hand) : hand_(hand) {}

    // Presents as a Frame controller (same controller type, input profile and render model) so
    // every app's existing Frame-controller bindings and hand rendering apply.  The only inputs
    // it drives are the pose and the hand skeleton.
    EVRInitError Activate(uint32_t id) override {
        id_ = id;
        PropertyContainerHandle_t c = VRProperties()->TrackedDeviceToPropertyContainer(id);
        auto *p = VRProperties();
        p->SetStringProperty(c, Prop_ModelNumber_String, hand_ ? "FrameHands Right" : "FrameHands Left");
        p->SetStringProperty(c, Prop_SerialNumber_String, hand_ ? "framehands-right" : "framehands-left");
        p->SetStringProperty(c, Prop_ManufacturerName_String, "framehands");
        p->SetStringProperty(c, Prop_TrackingSystemName_String, "framehands");
        p->SetStringProperty(c, Prop_ControllerType_String, "frame_controller");
        p->SetStringProperty(c, Prop_InputProfilePath_String, "{frame_controller}/input/frame_controller_profile.json");
        p->SetInt32Property(c, Prop_ControllerRoleHint_Int32,
                            hand_ ? TrackedControllerRole_RightHand : TrackedControllerRole_LeftHand);
        p->SetInt32Property(c, Prop_DeviceClass_Int32, TrackedDeviceClass_Controller);
        p->SetBoolProperty(c, Prop_DeviceIsWireless_Bool, true);
        p->SetBoolProperty(c, Prop_DeviceProvidesBatteryStatus_Bool, false);
        p->SetStringProperty(c, Prop_RenderModelName_String,
                             hand_ ? "{frame_controller}frame_controller_right" : "{frame_controller}frame_controller_left");
        setPriority(false);
        role_ = hand_ ? TrackedControllerRole_RightHand : TrackedControllerRole_LeftHand;
        EVRInputError e = VRDriverInput()->CreateSkeletonComponent(
            c, hand_ ? "/input/skeleton/right" : "/input/skeleton/left",
            hand_ ? "/skeleton/hand/right" : "/skeleton/hand/left", "/pose/raw", VRSkeletalTracking_Full, nullptr, 0,
            &skeleton_);
        VRDriverInput()->CreateHapticComponent(c, "/output/haptic", &haptic_);
        computePalmOffset();
        updateSkeleton();   // SteamVR wants skeletal data immediately
        updateSkeleton();
        Log("framehands: activated %s as device %u (skeleton err %d), palm offset in /pose/raw: t=(%.3f %.3f %.3f)",
            hand_ ? "right" : "left", id, e, palmT_[0], palmT_[1], palmT_[2]);
        return VRInitError_None;
    }

    // Palm frame of the open-hand reference skeleton in /pose/raw space, built the same way the
    // headset builds it from landmarks.  Controller pose = measured palm pose x inverse(this).
    void computePalmOffset() {
        VRBoneTransform_t b[eBone_Count];
        MyFingerCurls curls{};
        MyFingerSplays splays{};
        sim_.ComputeSkeletonTransforms(role_, curls, splays, b);
        static const int parent[eBone_Count] = {-1, 0, 1, 2, 3, 4, 1, 6, 7, 8, 9, 1, 11, 12, 13, 14,
                                                1, 16, 17, 18, 19, 1, 21, 22, 23, 24, 0, 0, 0, 0, 0};
        Quat q[eBone_Count];
        double t[eBone_Count][3];
        for (int i = 0; i < eBone_Count; i++) {
            Quat lq{b[i].orientation.w, b[i].orientation.x, b[i].orientation.y, b[i].orientation.z};
            double lt[3] = {b[i].position.v[0], b[i].position.v[1], b[i].position.v[2]};
            if (parent[i] < 0) {
                q[i] = lq;
                for (int k = 0; k < 3; k++) t[i][k] = lt[k];
            } else {
                double r[3];
                rotate(q[parent[i]], lt, r);
                for (int k = 0; k < 3; k++) t[i][k] = t[parent[i]][k] + r[k];
                q[i] = mul(q[parent[i]], lq);
            }
        }
        const double *wrist = t[eBone_Wrist], *idx = t[eBone_IndexFinger1], *mid = t[eBone_MiddleFinger1],
                     *ring = t[eBone_RingFinger1], *pinky = t[eBone_PinkyFinger1];
        double centre[3], z[3], x[3], y[3];
        for (int k = 0; k < 3; k++) {
            centre[k] = (wrist[k] + idx[k] + mid[k] + pinky[k]) / 4;
            z[k] = -((idx[k] + mid[k] + ring[k] + pinky[k]) / 4 - wrist[k]);
            x[k] = hand_ ? -(idx[k] - pinky[k]) : (idx[k] - pinky[k]);
        }
        auto norm = [](double *v) {
            double n = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            for (int k = 0; k < 3; k++) v[k] /= n;
        };
        norm(z);
        double d = x[0] * z[0] + x[1] * z[1] + x[2] * z[2];
        for (int k = 0; k < 3; k++) x[k] -= d * z[k];
        norm(x);
        y[0] = z[1] * x[2] - z[2] * x[1];
        y[1] = z[2] * x[0] - z[0] * x[2];
        y[2] = z[0] * x[1] - z[1] * x[0];
        HmdMatrix34_t m{};
        for (int r = 0; r < 3; r++) {
            m.m[r][0] = (float)x[r];
            m.m[r][1] = (float)y[r];
            m.m[r][2] = (float)z[r];
        }
        palmQ_ = quatFromMatrix(m);
        for (int k = 0; k < 3; k++) palmT_[k] = centre[k];
    }

    void updateSkeleton() {
        if (skeleton_ == k_ulInvalidInputComponentHandle) return;
        VRBoneTransform_t b[eBone_Count];
        MyFingerSplays splays{};
        sim_.ComputeSkeletonTransforms(role_, curls_, splays, b);
        VRDriverInput()->UpdateSkeletonComponent(skeleton_, VRSkeletalMotionRange_WithController, b, eBone_Count);
        VRDriverInput()->UpdateSkeletonComponent(skeleton_, VRSkeletalMotionRange_WithoutController, b, eBone_Count);
    }
    void Deactivate() override { id_ = k_unTrackedDeviceIndexInvalid; }
    void EnterStandby() override {}
    void *GetComponent(const char *) override { return nullptr; }
    void DebugRequest(const char *, char *resp, uint32_t n) override { if (n) resp[0] = 0; }
    DriverPose_t GetPose() override { return pose_; }
    uint32_t id() const { return id_; }

    void setPriority(bool active) {
        if (active == prioActive_ && prioSet_) return;
        prioSet_ = true;
        prioActive_ = active;
        PropertyContainerHandle_t c = VRProperties()->TrackedDeviceToPropertyContainer(id_);
        VRProperties()->SetInt32Property(c, Prop_ControllerHandSelectionPriority_Int32, active ? 10000 : -10000);
        Log("framehands: %s priority -> %s", hand_ ? "right" : "left", active ? "HIGH" : "low");
    }

    bool trackedWithin(double seconds) const {
        return !samples_.empty() && (nowNs() - samples_.back().t) / 1e9 < seconds;
    }
    double palmDistance(const double w[3]) const {
        double d = 0;
        for (int i = 0; i < 3; i++) d += (palmWorld_[i] - w[i]) * (palmWorld_[i] - w[i]);
        return sqrt(d);
    }

    // A new hand sample (head frame, capture time already mapped to PC clock) -> world sample.
    void ingest(const FhHand &h, int64_t tCapPc, Stats &st) {
        double hp[3], q0p[3];
        Quat qh;
        if (!g_hmd.at(tCapPc, hp, qh)) {
            st.hmdMiss++;
            return;
        }
        double pin[3] = {h.pos[0], h.pos[1], h.pos[2]}, wp[3];
        rotate(qh, pin, wp);
        for (int i = 0; i < 3; i++) wp[i] += hp[i];
        for (int i = 0; i < 3; i++) palmWorld_[i] = wp[i];
        Quat wq = mul(qh, Quat{h.quat[0], h.quat[1], h.quat[2], h.quat[3]});
        // palm pose -> controller (/pose/raw) pose: ctrl = palm x inverse(palm-in-raw)
        wq = mul(wq, conj(palmQ_));
        rotate(wq, palmT_, q0p);
        for (int i = 0; i < 3; i++) wp[i] -= q0p[i];
        if (!samples_.empty()) {
            const Sample &l = samples_.back();
            if (qdot(wq, l.q) < 0) wq = {-wq.w, -wq.x, -wq.y, -wq.z};
            if (tCapPc <= l.t) return;   // duplicate / out of order
            st.sampleGapMs.push_back((tCapPc - l.t) / 1e6);
        }
        samples_.push_back({tCapPc, {wp[0], wp[1], wp[2]}, wq});
        while (samples_.size() > 64) samples_.pop_front();
        curls_ = {h.curl[0], h.curl[1], h.curl[2], h.curl[3], h.curl[4]};
        st.samples++;
    }

    // Presentation, every RunFrame.
    void present(bool handMode, const Config &c, Stats &st) {
        setPriority(handMode);
        if (handMode != connected_) {
            connected_ = handMode;
            Log("framehands: %s virtual controller %s", hand_ ? "right" : "left",
                handMode ? "CONNECTED (hand mode)" : "disconnected (controller mode)");
        }
        DriverPose_t p{};
        p.qWorldFromDriverRotation = {1, 0, 0, 0};
        p.qDriverFromHeadRotation = {1, 0, 0, 0};
        p.qRotation = {1, 0, 0, 0};
        p.deviceIsConnected = handMode;
        p.result = TrackingResult_Running_OutOfRange;
        if (handMode && !samples_.empty()) {
            int64_t now = nowNs();
            int64_t target = now - int64_t(c.interpDelayMs * 1e6);
            const Sample &newest = samples_.back();
            double pos[3];
            Quat q;
            int64_t tState;
            if (target <= newest.t && samples_.size() >= 2 && c.interpDelayMs > 0) {
                // interpolate between the two samples bracketing the target time
                size_t i = samples_.size() - 1;
                while (i > 0 && samples_[i - 1].t > target) i--;
                const Sample &b = samples_[i], &a = samples_[i > 0 ? i - 1 : 0];
                double f = b.t > a.t ? std::clamp(double(target - a.t) / double(b.t - a.t), 0.0, 1.0) : 1.0;
                for (int k = 0; k < 3; k++) pos[k] = a.p[k] + f * (b.p[k] - a.p[k]);
                q = slerp(a.q, b.q, f);
                tState = target;
                st.interp++;
            } else {
                for (int k = 0; k < 3; k++) pos[k] = newest.p[k];
                q = newest.q;
                tState = newest.t;
                st.extrap++;
            }
            double age = (now - tState) / 1e9;
            st.presentAgeMs.push_back((now - newest.t) / 1e6);
            for (int k = 0; k < 3; k++) p.vecPosition[k] = pos[k];
            p.qRotation = q;
            p.poseIsValid = true;
            if ((now - newest.t) / 1e6 < c.holdMs) {
                p.result = TrackingResult_Running_OK;
                double v[3], w[3];
                velocities(c, v, w);
                // SteamVR predicts from (state time) to display time with these velocities; cap how far
                p.poseTimeOffset = -std::min(age, c.maxPredictMs / 1000.0);
                for (int k = 0; k < 3; k++) {
                    p.vecVelocity[k] = c.predictLinear ? v[k] : 0;
                    p.vecAngularVelocity[k] = c.predictAngular ? w[k] : 0;
                }
            } else {
                st.held++;   // stale: hold, out of range, no velocity
            }
            st.presents++;
        }
        pose_ = p;
        VRServerDriverHost()->TrackedDevicePoseUpdated(id_, pose_, sizeof pose_);
        updateSkeleton();
    }

    // Linear velocity: least-squares slope over samples within velWindowMs of the newest.
    // Angular velocity: rotation between the newest sample and one >= velWindowMs/2 older.
    void velocities(const Config &c, double v[3], double w[3]) {
        for (int k = 0; k < 3; k++) v[k] = w[k] = 0;
        if (samples_.size() < 2) return;
        const Sample &n = samples_.back();
        int64_t win = int64_t(c.velWindowMs * 1e6);
        double st = 0, stt = 0, sp[3] = {}, stp[3] = {};
        int cnt = 0;
        for (auto it = samples_.rbegin(); it != samples_.rend() && n.t - it->t <= win; ++it) {
            double t = (it->t - n.t) / 1e9;
            st += t;
            stt += t * t;
            for (int k = 0; k < 3; k++) {
                sp[k] += it->p[k];
                stp[k] += t * it->p[k];
            }
            cnt++;
        }
        double den = cnt * stt - st * st;
        if (cnt >= 2 && den > 1e-12)
            for (int k = 0; k < 3; k++) v[k] = (cnt * stp[k] - st * sp[k]) / den;
        // a jump is a tracking glitch, never something to extrapolate
        const Sample &pv = samples_[samples_.size() - 2];
        double jump = 0;
        for (int k = 0; k < 3; k++) jump += (n.p[k] - pv.p[k]) * (n.p[k] - pv.p[k]);
        if (sqrt(jump) > c.maxJumpCm / 100.0) {
            for (int k = 0; k < 3; k++) v[k] = 0;
            return;
        }
        double vn = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        if (vn > c.maxVelocity)
            for (int k = 0; k < 3; k++) v[k] *= c.maxVelocity / vn;
        for (auto it = samples_.rbegin(); it != samples_.rend(); ++it) {
            double dt = (n.t - it->t) / 1e9;
            if (dt * 2000 < c.velWindowMs) continue;
            Quat d = mul(n.q, conj(it->q));
            if (d.w < 0) d = {-d.w, -d.x, -d.y, -d.z};
            double s = sqrt(d.x * d.x + d.y * d.y + d.z * d.z), ang = 2 * atan2(s, d.w);
            if (s > 1e-9) {
                w[0] = d.x / s * ang / dt;
                w[1] = d.y / s * ang / dt;
                w[2] = d.z / s * ang / dt;
            }
            double wn = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
            if (wn > c.maxAngularVelocity)
                for (int k = 0; k < 3; k++) w[k] *= c.maxAngularVelocity / wn;
            break;
        }
    }

private:
    struct Sample {
        int64_t t;
        double p[3];
        Quat q;
    };
    int hand_;
    uint32_t id_ = k_unTrackedDeviceIndexInvalid;
    DriverPose_t pose_{};
    std::deque<Sample> samples_;
    bool prioSet_ = false, prioActive_ = false, connected_ = false;
    ETrackedControllerRole role_ = TrackedControllerRole_LeftHand;
    VRInputComponentHandle_t skeleton_ = k_ulInvalidInputComponentHandle;
    VRInputComponentHandle_t haptic_ = k_ulInvalidInputComponentHandle;
    MyHandSimulation sim_;
    MyFingerCurls curls_{};
    Quat palmQ_{1, 0, 0, 0};
    double palmT_[3] = {};
    double palmWorld_[3] = {};
};

// ---------------------------------------------------------------- set-down detection
// Hand mode needs BOTH: the real controller stayed within stillCm / stillDeg of where it
// settled for stillSeconds, AND the tracked palm is currently seen > handAwayCm from it (a
// controller held still in the hand must not switch).  Moving it > pickupCm / pickupDeg from
// where it lay -> controller mode.
struct SetDownDetector {
    bool handMode = false, anchored = false;
    HmdMatrix34_t anchor{};
    Clock::time_point since{};

    static double dist(const HmdMatrix34_t &a, const HmdMatrix34_t &b) {
        double d = 0;
        for (int i = 0; i < 3; i++) d += (a.m[i][3] - b.m[i][3]) * (a.m[i][3] - b.m[i][3]);
        return sqrt(d);
    }
    static double angleDeg(const HmdMatrix34_t &a, const HmdMatrix34_t &b) {
        double tr = 0;
        for (int i = 0; i < 3; i++)
            for (int k = 0; k < 3; k++) tr += a.m[k][i] * b.m[k][i];
        return acos(std::max(-1.0, std::min(1.0, (tr - 1) / 2))) * 180.0 / 3.14159265358979;
    }
    void reset() {
        handMode = false;
        anchored = false;
    }
    bool feed(const TrackedDevicePose_t &p, const Config &c, bool handAway) {
        if (!p.bDeviceIsConnected || !p.bPoseIsValid) return false;
        auto now = Clock::now();
        const HmdMatrix34_t &m = p.mDeviceToAbsoluteTracking;
        if (!anchored) {
            anchor = m;
            since = now;
            anchored = true;
            return false;
        }
        double d = dist(anchor, m), a = angleDeg(anchor, m);
        if (handMode) {
            if (d > c.pickupCm / 100.0 || a > c.pickupDeg) {
                handMode = false;
                anchor = m;
                since = now;
                return true;
            }
            return false;
        }
        if (d > c.stillCm / 100.0 || a > c.stillDeg) {
            anchor = m;
            since = now;
            return false;
        }
        if (handAway && std::chrono::duration<double>(now - since).count() >= c.stillSeconds) {
            handMode = true;
            return true;
        }
        return false;
    }
};

// ---------------------------------------------------------------- provider
class Provider : public IServerTrackedDeviceProvider {
public:
    EVRInitError Init(IVRDriverContext *ctx) override {
        VR_INIT_SERVER_DRIVER_CONTEXT(ctx);
        livePath_ = driverDir() + "\\framehands_live.ini";
        loadLiveConfig(livePath_, true);
        Log("framehands v3: live config %s", livePath_.c_str());
        WSADATA w;
        WSAStartup(MAKEWORD(2, 2), &w);
        for (int i = 0; i < 2; i++) {
            hands_[i] = new HandController(i);
            VRServerDriverHost()->TrackedDeviceAdded(i ? "framehands-right" : "framehands-left",
                                                     TrackedDeviceClass_Controller, hands_[i]);
        }
        running_ = true;
        net_ = std::thread([this] { netLoop(); });
        return VRInitError_None;
    }
    void Cleanup() override {
        running_ = false;
        if (sock_ != INVALID_SOCKET) closesocket(sock_);
        if (net_.joinable()) net_.join();
        WSACleanup();
    }
    const char *const *GetInterfaceVersions() override { return k_InterfaceVersions; }

    void RunFrame() override {
        int64_t now = nowNs();
        // live config, once a second
        if (now - lastCfgCheck_ > 1000000000LL) {
            lastCfgCheck_ = now;
            WIN32_FILE_ATTRIBUTE_DATA fa;
            if (GetFileAttributesExA(livePath_.c_str(), GetFileExInfoStandard, &fa)) {
                uint64_t mt = (uint64_t(fa.ftLastWriteTime.dwHighDateTime) << 32) | fa.ftLastWriteTime.dwLowDateTime;
                if (mt != cfgMtime_) {
                    cfgMtime_ = mt;
                    loadLiveConfig(livePath_, true);
                }
            }
        }
        Config c = cfg();

        // all device poses: HMD history + real controllers for set-down detection
        TrackedDevicePose_t poses[k_unMaxTrackedDeviceCount];
        VRServerDriverHost()->GetRawTrackedDevicePoses(0, poses, k_unMaxTrackedDeviceCount);
        if (poses[0].bPoseIsValid) g_hmd.push(now, poses[0].mDeviceToAbsoluteTracking);

        // new hand packet?
        FhHandPacket pk;
        int64_t rx;
        bool fresh;
        {
            std::lock_guard<std::mutex> l(g_state.m);
            pk = g_state.last;
            rx = g_state.lastRxNs;
            fresh = g_state.have && now - g_state.lastRxNs < 500000000LL;
        }
        // hands enabled on the headset panel (v3 pad: bit 31 present, bit 0 left, bit 1 right;
        // older trackers send 0 = both)
        bool enabled[2] = {true, true};
        if (fresh && (pk.pad & 0x80000000u))
            for (int i = 0; i < 2; i++) enabled[i] = (pk.pad >> i) & 1;
        for (int i = 0; i < 2; i++)
            if (enabled[i] != enabled_[i]) {
                enabled_[i] = enabled[i];
                Log("framehands: %s hand %s on the headset", i ? "right" : "left", enabled[i] ? "enabled" : "disabled");
                if (!enabled[i] && detect_[i].handMode) Log("framehands: %s -> CONTROLLER mode (hand disabled)", i ? "right" : "left");
                detect_[i].reset();
            }
        if (fresh && pk.seq != lastSeq_) {
            lastSeq_ = pk.seq;
            int64_t tCapPc;
            if (g_clock.hsToPc((int64_t)pk.tCapNs, tCapPc))
                for (int i = 0; i < 2; i++)
                    if (pk.hand[i].valid) hands_[i]->ingest(pk.hand[i], tCapPc, stats_[i]);
        }

        for (uint32_t i = 1; i < k_unMaxTrackedDeviceCount; i++) {
            if (i == hands_[0]->id() || i == hands_[1]->id() || !poses[i].bDeviceIsConnected) continue;
            PropertyContainerHandle_t pc = VRProperties()->TrackedDeviceToPropertyContainer(i);
            ETrackedPropertyError e;
            if (VRProperties()->GetInt32Property(pc, Prop_DeviceClass_Int32, &e) != TrackedDeviceClass_Controller)
                continue;
            int32_t role = VRProperties()->GetInt32Property(pc, Prop_ControllerRoleHint_Int32, &e);
            int hand = role == TrackedControllerRole_LeftHand ? 0 : role == TrackedControllerRole_RightHand ? 1 : -1;
            if (hand < 0 || !enabled[hand]) continue;
            const HmdMatrix34_t &m = poses[i].mDeviceToAbsoluteTracking;
            double cpos[3] = {m.m[0][3], m.m[1][3], m.m[2][3]};
            bool seen = hands_[hand]->trackedWithin(0.3);
            bool away = seen && hands_[hand]->palmDistance(cpos) > c.handAwayCm / 100.0;
            if (detect_[hand].feed(poses[i], c, away))
                Log("framehands: %s controller %s -> %s mode (palm %.2f m from controller)", hand ? "right" : "left",
                    detect_[hand].handMode ? "set down" : "picked up", detect_[hand].handMode ? "HAND" : "CONTROLLER",
                    seen ? hands_[hand]->palmDistance(cpos) : -1.0);
        }
        for (int i = 0; i < 2; i++)
            hands_[i]->present(fresh && enabled[i] && (detect_[i].handMode || c.forceHandMode), c, stats_[i]);

        if (c.statsSeconds > 0 && now - lastStats_ > int64_t(c.statsSeconds * 1e9)) {
            double dt = (now - lastStats_) / 1e9;
            lastStats_ = now;
            uint64_t pk_, pg_;
            {
                std::lock_guard<std::mutex> l(g_state.m);
                pk_ = g_state.packets;
                pg_ = g_state.pongs;
                g_state.packets = g_state.pongs = 0;
            }
            Log("framehands stats: %.1f packets/s, %.1f pongs/s, clock offset %.3f ms (min rtt %.2f ms), hmd history %zu",
                pk_ / dt, pg_ / dt, g_clock.offset / 1e6, g_clock.rttMin / 1e6, g_hmd.h.size());
            for (int i = 0; i < 2; i++) {
                Stats &s = stats_[i];
                if (s.presents || s.samples)
                    Log("framehands stats %s: samples %.1f/s (gap ms p50 %.1f p95 %.1f), presents %.1f/s "
                        "(interp %llu extrap %llu held %llu), newest-sample age at present ms p50 %.1f p95 %.1f p99 %.1f, "
                        "hmd-history misses %llu",
                        i ? "right" : "left", s.samples / dt, pctl(s.sampleGapMs, 50), pctl(s.sampleGapMs, 95),
                        s.presents / dt, (unsigned long long)s.interp, (unsigned long long)s.extrap,
                        (unsigned long long)s.held, pctl(s.presentAgeMs, 50), pctl(s.presentAgeMs, 95),
                        pctl(s.presentAgeMs, 99), (unsigned long long)s.hmdMiss);
                s.clear();
            }
        }
        VREvent_t ev;
        while (VRServerDriverHost()->PollNextEvent(&ev, sizeof ev)) {}
    }
    bool ShouldBlockStandbyMode() override { return false; }
    void EnterStandby() override {}
    void LeaveStandby() override {}

private:
    void netLoop() {
        sock_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        DWORD tmo = 20;
        setsockopt(sock_, SOL_SOCKET, SO_RCVTIMEO, (const char *)&tmo, sizeof tmo);
        int64_t lastPing = 0;
        bool logged = false;
        while (running_) {
            Config c = cfg();
            sockaddr_in hs{};
            hs.sin_family = AF_INET;
            hs.sin_port = htons((u_short)c.port);
            inet_pton(AF_INET, c.headsetHost.c_str(), &hs.sin_addr);
            int64_t now = nowNs();
            if (now - lastPing > 100000000LL) {   // ping (and subscribe) every 100 ms
                FhPing ping{FH_MAGIC, FH_VERSION, FH_KIND_PING, 0, (uint64_t)now};
                sendto(sock_, (const char *)&ping, sizeof ping, 0, (sockaddr *)&hs, sizeof hs);
                lastPing = now;
            }
            char buf[512];
            sockaddr_in from{};
            int fl = sizeof from;
            int n = recvfrom(sock_, buf, sizeof buf, 0, (sockaddr *)&from, &fl);
            int64_t rxNs = nowNs();
            if (n < 12 || from.sin_addr.s_addr != hs.sin_addr.s_addr) continue;   // only the configured headset
            uint32_t magic, version, kind;
            memcpy(&magic, buf, 4);
            memcpy(&version, buf + 4, 4);
            memcpy(&kind, buf + 8, 4);
            if (magic != FH_MAGIC || version != FH_VERSION) continue;
            if (n == (int)sizeof(FhPong) && kind == FH_KIND_PONG) {
                FhPong pg;
                memcpy(&pg, buf, sizeof pg);
                g_clock.add((int64_t)pg.pcSendNs, rxNs, (int64_t)pg.hsRecvNs);
                std::lock_guard<std::mutex> l(g_state.m);
                g_state.pongs++;
                continue;
            }
            if (n != (int)sizeof(FhHandPacket)) {
                std::lock_guard<std::mutex> l(g_state.m);
                g_state.badSize++;
                continue;
            }
            FhHandPacket pk;
            memcpy(&pk, buf, sizeof pk);
            std::lock_guard<std::mutex> l(g_state.m);
            g_state.last = pk;
            g_state.lastRxNs = rxNs;
            g_state.have = true;
            g_state.packets++;
            if (!logged) {
                Log("framehands: receiving hand data (v3) from headset");
                logged = true;
            }
        }
    }

    HandController *hands_[2] = {};
    SetDownDetector detect_[2];
    bool enabled_[2] = {true, true};
    Stats stats_[2];
    std::thread net_;
    std::atomic<bool> running_{false};
    SOCKET sock_ = INVALID_SOCKET;
    std::string livePath_;
    uint64_t cfgMtime_ = 0;
    int64_t lastCfgCheck_ = 0, lastStats_ = 0;
    uint64_t lastSeq_ = 0;
};

static Provider g_provider;

extern "C" __declspec(dllexport) void *HmdDriverFactory(const char *name, int *ret) {
    if (!strcmp(name, IServerTrackedDeviceProvider_Version)) return &g_provider;
    if (ret) *ret = VRInitError_Init_InterfaceNotFound;
    return nullptr;
}
