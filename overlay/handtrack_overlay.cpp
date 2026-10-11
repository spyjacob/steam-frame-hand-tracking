// handtrack_overlay — SteamVR dashboard panel "Hand Tracking": LEFT / RIGHT toggles for the
// optical hand-tracking system on the headset (ptgrab + tracker + watchdog).  The enabled hands
// are written to ~/.config/mrhand/hands (read live by the tracker, which skips a disabled hand
// and tells the PC driver); the first hand on runs ~/handtrack/start.sh, the last hand off runs
// ~/handtrack/stop.sh.  Public OpenVR overlay API only; modelled on the mr-chroma "Chroma Key"
// panel but a separate program (the chroma panel is untouched).
#include <openvr.h>
#include <ft2build.h>
#include FT_FREETYPE_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <signal.h>
#include <string>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace vr;
using Clock = std::chrono::steady_clock;

static const int W = 1024, H = 480;
static std::string g_home;

// ---------------------------------------------------------------- status
static bool pidAlive(const std::string &pidFile) {
    FILE *f = fopen(pidFile.c_str(), "r");
    if (!f) return false;
    int pid = 0;
    bool ok = fscanf(f, "%d", &pid) == 1 && pid > 0;
    fclose(f);
    return ok && kill(pid, 0) == 0;
}
static std::string runtimeDir() {
    const char *r = getenv("XDG_RUNTIME_DIR");
    return std::string(r ? r : "/tmp") + "/mrhand";
}
static bool trackerOn() { return pidAlive(runtimeDir() + "/live.pid"); }

// enabled hands: bit 0 left, bit 1 right.  Missing file = both (what the tracker assumes too).
static std::string handsPath() { return g_home + "/.config/mrhand/hands"; }
static int readHands() {
    FILE *f = fopen(handsPath().c_str(), "r");
    if (!f) return 3;
    char buf[64] = {};
    size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[n] = 0;
    return (strstr(buf, "left") ? 1 : 0) | (strstr(buf, "right") ? 2 : 0);
}
static void writeHands(int m) {
    std::string tmp = handsPath() + ".tmp";
    if (FILE *f = fopen(tmp.c_str(), "w")) {
        fprintf(f, "%s%s%s\n", m & 1 ? "left" : "", m == 3 ? " " : "", m & 2 ? "right" : "");
        fclose(f);
        rename(tmp.c_str(), handsPath().c_str());
    }
}
static std::string lastLine(const std::string &path, const char *contains) {
    FILE *f = fopen(path.c_str(), "r");
    if (!f) return "";
    char buf[512];
    std::string last;
    while (fgets(buf, sizeof buf, f))
        if (!contains || strstr(buf, contains)) last = buf;
    fclose(f);
    while (!last.empty() && (last.back() == '\n' || last.back() == '\r')) last.pop_back();
    return last;
}

// ---------------------------------------------------------------- drawing
struct Canvas {
    std::vector<uint8_t> px = std::vector<uint8_t>(W * H * 4);
    FT_Library ft = nullptr;
    FT_Face bold = nullptr, reg = nullptr;
    void blend(int x, int y, uint32_t rgb, float a) {
        if (x < 0 || y < 0 || x >= W || y >= H || a <= 0) return;
        uint8_t *p = &px[(y * W + x) * 4];
        float c[3] = {float(rgb >> 16 & 255), float(rgb >> 8 & 255), float(rgb & 255)};
        float da = p[3] / 255.f, oa = a + da * (1 - a);
        for (int i = 0; i < 3; i++) p[i] = uint8_t((c[i] * a + p[i] * da * (1 - a)) / std::max(oa, 1e-4f));
        p[3] = uint8_t(oa * 255);
    }
    void clear() { std::fill(px.begin(), px.end(), 0); }
    void rrect(float x0, float y0, float x1, float y1, float rad, uint32_t rgb, float a = 1) {
        for (int y = int(y0); y < int(ceil(y1)); y++)
            for (int x = int(x0); x < int(ceil(x1)); x++) {
                float cx = std::clamp(x + .5f, x0 + rad, x1 - rad), cy = std::clamp(y + .5f, y0 + rad, y1 - rad);
                blend(x, y, rgb, a * std::clamp(0.5f - (hypotf(x + .5f - cx, y + .5f - cy) - rad), 0.f, 1.f));
            }
    }
    void ring(float x0, float y0, float x1, float y1, float rad, float w, uint32_t rgb) {
        for (int y = int(y0 - w); y < int(y1 + w); y++)
            for (int x = int(x0 - w); x < int(x1 + w); x++) {
                float cx = std::clamp(x + .5f, x0 + rad, x1 - rad), cy = std::clamp(y + .5f, y0 + rad, y1 - rad);
                float d = fabsf(hypotf(x + .5f - cx, y + .5f - cy) - rad) - w / 2;
                blend(x, y, rgb, std::clamp(0.5f - d, 0.f, 1.f));
            }
    }
    void text(const std::string &s, int x, int y, int size, uint32_t rgb, bool b = false, int align = 0) {
        FT_Face f = b ? bold : reg;
        if (!f) return;
        FT_Set_Pixel_Sizes(f, 0, size);
        int w = 0;
        for (char c : s)
            if (!FT_Load_Char(f, c, FT_LOAD_DEFAULT)) w += f->glyph->advance.x >> 6;
        int pen = align == 1 ? x - w / 2 : align == 2 ? x - w : x;
        for (char c : s) {
            if (FT_Load_Char(f, c, FT_LOAD_RENDER)) continue;
            FT_Bitmap &bm = f->glyph->bitmap;
            for (unsigned yy = 0; yy < bm.rows; yy++)
                for (unsigned xx = 0; xx < bm.width; xx++)
                    blend(pen + f->glyph->bitmap_left + xx, y - f->glyph->bitmap_top + yy, rgb,
                          bm.buffer[yy * bm.pitch + xx] / 255.f);
            pen += f->glyph->advance.x >> 6;
        }
    }
};

struct Rect { float x0, y0, x1, y1; bool in(float x, float y) const { return x >= x0 && x < x1 && y >= y0 && y < y1; } };
static const Rect kHand[2] = {{60, 130, 492, 340}, {532, 130, 964, 340}};

// hands: enabled mask; running: tracker process alive
static void render(Canvas &cv, int hands, bool running, bool busy, const std::string &status) {
    cv.clear();
    cv.rrect(0, 0, W, H, 40, 0x1c1d22, 0.94f);
    cv.ring(4, 4, W - 4, H - 4, 36, 4, 0x4a4d57);
    cv.text("HAND TRACKING", W / 2, 92, 60, 0xe8e8e8, true, 1);
    for (int i = 0; i < 2; i++) {
        const Rect &r = kHand[i];
        bool en = hands >> i & 1, stopped = en && !running;
        // green = tracking, amber = enabled but the tracker is not running (e.g. thermal watchdog), red = off
        uint32_t tc = busy ? 0x9a9a9a : stopped ? 0xe0b050 : en ? 0x5ae678 : 0xeb5f55;
        cv.rrect(r.x0, r.y0, r.x1, r.y1, 50, busy ? 0x34343a : stopped ? 0x3a3018 : en ? 0x173a22 : 0x3d1d1d);
        cv.ring(r.x0, r.y0, r.x1, r.y1, 50, 8, tc);
        float cx = (r.x0 + r.x1) / 2;
        cv.text(i ? "RIGHT" : "LEFT", int(cx), int(r.y0) + 70, 44, 0xe8e8e8, true, 1);
        cv.text(busy ? "..." : stopped ? "STOPPED" : en ? "ON" : "OFF", int(cx), int(r.y0) + 170, stopped ? 64 : 90, tc,
                true, 1);
    }
    std::string s = status.size() > 70 ? status.substr(0, 70) : status;
    cv.text(s, W / 2, 392, 26, 0xb0b0b0, false, 1);
    cv.text("one hand = less heat;  set its controller down to switch", W / 2, 440, 26, 0x80838c, false, 1);
}

int main(int argc, char **argv) {
    const char *home = getenv("HOME");
    g_home = home ? home : "/home/steamos";
    std::string self = argv[0];
    char real[4096];
    if (realpath("/proc/self/exe", real)) self = real;
    std::string resDir = self.substr(0, self.rfind('/'));

    Canvas cv;
    FT_Init_FreeType(&cv.ft);
    FT_New_Face(cv.ft, "/usr/share/fonts/noto/NotoSans-Bold.ttf", 0, &cv.bold);
    FT_New_Face(cv.ft, "/usr/share/fonts/noto/NotoSans-Regular.ttf", 0, &cv.reg);
    if (!cv.reg) cv.reg = cv.bold;

    if (argc > 2 && !strcmp(argv[1], "--render")) {   // layout preview without SteamVR
        render(cv, 2, true, false, "right hand on, 74 C");
        FILE *f = fopen(argv[2], "wb");
        fprintf(f, "P6 %d %d 255\n", W, H);
        for (int i = 0; i < W * H; i++) {
            float a = cv.px[i * 4 + 3] / 255.f;
            for (int c = 0; c < 3; c++) fputc(int(cv.px[i * 4 + c] * a + 96 * (1 - a)), f);
        }
        fclose(f);
        return 0;
    }

    EVRInitError ie;
    VR_Init(&ie, VRApplication_Overlay);
    if (ie != VRInitError_None) {
        fprintf(stderr, "VR_Init: %s\n", VR_GetVRInitErrorAsEnglishDescription(ie));
        return 1;
    }
    IVROverlay *ov = VROverlay();
    VROverlayHandle_t main = k_ulOverlayHandleInvalid, thumb = k_ulOverlayHandleInvalid;
    if (ov->CreateDashboardOverlay("handtrack.toggle", "Hand Tracking", &main, &thumb) != VROverlayError_None) {
        fprintf(stderr, "CreateDashboardOverlay failed\n");
        VR_Shutdown();
        return 1;
    }
    ov->SetOverlayWidthInMeters(main, 2.0f);
    ov->SetOverlayInputMethod(main, VROverlayInputMethod_Mouse);
    HmdVector2_t scale = {{float(W), float(H)}};
    ov->SetOverlayMouseScale(main, &scale);
    ov->SetOverlayFromFile(thumb, (resDir + "/icon.png").c_str());

    std::atomic<bool> busy{false};
    std::thread worker;
    bool on = trackerOn(), shownOn = !on, shownBusy = true;
    int hands = readHands(), shownHands = -1;
    std::string status, shownStatus = "-";
    auto lastPoll = Clock::time_point();
    bool quit = false;
    while (!quit) {
        VREvent_t ev;
        while (ov->PollNextOverlayEvent(main, &ev, sizeof ev)) {
            if (ev.eventType != VREvent_MouseButtonUp || busy) continue;
            float x = ev.data.mouse.x, y = H - ev.data.mouse.y;   // OpenVR mouse origin is bottom-left
            int i = kHand[0].in(x, y) ? 0 : kHand[1].in(x, y) ? 1 : -1;
            if (i < 0) continue;
            bool en = hands >> i & 1;
            // an enabled hand whose tracker stopped (watchdog) restarts instead of switching off
            int want = en && !on ? hands : hands ^ (1 << i);
            writeHands(want);
            hands = want;
            std::string cmd;
            if (want && !on) cmd = g_home + "/handtrack/start.sh";
            else if (!want && on) cmd = g_home + "/handtrack/stop.sh";
            if (cmd.empty()) continue;           // the running tracker picks the change up itself
            busy = true;
            if (worker.joinable()) worker.join();
            worker = std::thread([cmd, &busy] {
                if (system((cmd + " >/dev/null 2>&1").c_str()) != 0) fprintf(stderr, "command failed: %s\n", cmd.c_str());
                std::this_thread::sleep_for(std::chrono::milliseconds(1500));
                busy = false;
            });
        }
        while (VRSystem()->PollNextEvent(&ev, sizeof ev))
            if (ev.eventType == VREvent_Quit) {
                VRSystem()->AcknowledgeQuit_Exiting();
                quit = true;
            }
        auto now = Clock::now();
        if (now - lastPoll > std::chrono::milliseconds(700)) {
            lastPoll = now;
            on = trackerOn();
            hands = readHands();
            std::string wd = lastLine("/tmp/watchdog.log", "WATCHDOG:");
            const char *which = hands == 3 ? "both hands" : hands == 1 ? "left hand" : "right hand";
            if (!on && hands && !wd.empty()) status = "stopped by thermal watchdog - tap a hand to restart";
            else if (on) {
                std::string st = lastLine("/tmp/live.log", "fast:");
                // e.g. "[77 C] fast: ..." -> show temperature and hand pose rate
                std::string temp = st.size() > 1 && st[0] == '[' ? st.substr(1, st.find(']') - 1) : "";
                status = std::string(hands ? which : "no hands") + " on" + (temp.empty() ? std::string() : ", " + temp);
            } else
                status = "off";
        }
        if (on != shownOn || busy != shownBusy || status != shownStatus || hands != shownHands) {
            render(cv, hands, on, busy, status);
            ov->SetOverlayRaw(main, cv.px.data(), W, H, 4);
            shownOn = on;
            shownHands = hands;
            shownBusy = busy;
            shownStatus = status;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    if (worker.joinable()) worker.join();
    VR_Shutdown();
    return 0;
}
