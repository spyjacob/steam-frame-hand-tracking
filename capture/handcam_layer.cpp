// VK_LAYER_MRHAND — passive camera-buffer registry for vrcompositor.
//
// Independent of the mr-chroma project.  Loads only into vrcompositor, only
// while ~/.config/mrhand/enable-capture exists, and stops loading after the
// compositor starts 3x within 120 s (crash guard).
//
// It records no commands and touches no queues.  When the compositor imports
// external memory (the dma-bufs XRService shares for passthrough/tracking
// camera frames), the layer dup()s the fd *before* passing the import down,
// remembers the image it is bound to (format, extent, tiling, DRM modifier and
// plane layout), and serves {metadata + dup'd fd} over a private unix socket
// ($XDG_RUNTIME_DIR/mrhand/cam.sock, same-uid peers only).  The consumer
// (handtrack) mmaps those buffers read-only on its own time.

#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <mutex>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "handcam_proto.h"

#define EXPORT extern "C" __attribute__((visibility("default")))

// ---------------------------------------------------------------- logging
static FILE *g_log;
static std::mutex g_logM;
static int g_logLines;
static double monoMs() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}
static void LOG(const char *fmt, ...) {
    std::lock_guard<std::mutex> l(g_logM);
    if (g_logLines > 20000) return;
    if (!g_log) {
        char p[128];
        snprintf(p, sizeof p, "/tmp/mrhandcam-%d.log", getpid());
        g_log = fopen(p, "w");
        if (!g_log) return;
        setvbuf(g_log, nullptr, _IOLBF, 0);
    }
    fprintf(g_log, "[%.3f] ", monoMs());
    va_list ap;
    va_start(ap, fmt);
    vfprintf(g_log, fmt, ap);
    va_end(ap);
    fputc('\n', g_log);
    g_logLines++;
}
static std::string home(const char *rel) {
    const char *h = getenv("HOME");
    return std::string(h ? h : "/tmp") + rel;
}

// ---------------------------------------------------------------- load gating
static bool fileExists(const std::string &p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0;
}
static bool compositorAllowed() {
    if (!fileExists(home("/.config/mrhand/enable-capture"))) return false;
    std::string dir = home("/.local/share/mrhand");
    mkdir(dir.c_str(), 0755);
    if (fileExists(dir + "/guard-tripped")) return false;
    // crash guard: remember recent start times
    std::vector<long> starts;
    long now = time(nullptr);
    if (FILE *f = fopen((dir + "/compositor-starts").c_str(), "r")) {
        long t;
        while (fscanf(f, "%ld", &t) == 1)
            if (now - t < 120) starts.push_back(t);
        fclose(f);
    }
    starts.push_back(now);
    if (starts.size() >= 3) {
        if (FILE *f = fopen((dir + "/guard-tripped").c_str(), "w")) {
            fprintf(f, "vrcompositor started %zu times in 120 s with VK_LAYER_MRHAND; layer disabled.\n"
                       "Delete this file to re-arm.\n", starts.size());
            fclose(f);
        }
        return false;
    }
    if (FILE *f = fopen((dir + "/compositor-starts").c_str(), "w")) {
        for (long t : starts) fprintf(f, "%ld\n", t);
        fclose(f);
    }
    return true;
}
static bool processAllowed() {
    char exe[512];
    ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
    if (n <= 0) return false;
    exe[n] = 0;
    const char *base = strrchr(exe, '/');
    base = base ? base + 1 : exe;
    if (strcmp(base, "vrcompositor")) return false;
    static int decided = -1;   // evaluate (and count the start) once per process
    if (decided < 0) decided = compositorAllowed() ? 1 : 0;
    return decided == 1;
}

// ---------------------------------------------------------------- registry
struct ImageRec {
    VkImageCreateInfo ci{};
    uint64_t modifier = 0;
    bool hasModifier = false;
    uint32_t planeCount = 0;
    VkSubresourceLayout planes[MRHAND_MAX_PLANES]{};
    VkExternalMemoryHandleTypeFlags extTypes = 0;
};
struct MemRec {
    int fd = -1;              // our dup
    uint64_t size = 0;        // dma-buf size (lseek) or allocationSize
    uint32_t handleType = 0;
    VkImage image = VK_NULL_HANDLE;
    uint64_t bindOffset = 0;
    uint32_t id = 0;
};

template <typename T> static void *key(T h) { return *(void **)h; }

struct InstanceData {
    PFN_vkGetInstanceProcAddr gipa;
    PFN_vkDestroyInstance DestroyInstance;
};
struct DeviceData {
    VkDevice device;
    PFN_vkGetDeviceProcAddr gdpa;
    PFN_vkDestroyDevice DestroyDevice;
    PFN_vkCreateImage CreateImage;
    PFN_vkDestroyImage DestroyImage;
    PFN_vkAllocateMemory AllocateMemory;
    PFN_vkFreeMemory FreeMemory;
    PFN_vkBindImageMemory BindImageMemory;
    PFN_vkBindImageMemory2 BindImageMemory2;
};

static std::mutex g_mapM;
static std::unordered_map<void *, InstanceData *> g_inst;
static std::unordered_map<void *, DeviceData *> g_dev;

static std::mutex g_regM;
static std::unordered_map<VkImage, ImageRec> g_images;
static std::unordered_map<VkDeviceMemory, MemRec> g_mems;
static uint32_t g_nextId = 1;
static std::atomic<uint32_t> g_generation{1};

static const void *findPNext(const void *p, VkStructureType t) {
    for (auto *b = (const VkBaseInStructure *)p; b; b = b->pNext)
        if (b->sType == t) return b;
    return nullptr;
}

// ---------------------------------------------------------------- socket server
static void serveClient(int c) {
    ucred cr{};
    socklen_t cl = sizeof cr;
    if (getsockopt(c, SOL_SOCKET, SO_PEERCRED, &cr, &cl) != 0 || cr.uid != getuid()) {
        close(c);
        return;
    }
    std::vector<std::pair<MrhandBufInfo, int>> out;
    {
        std::lock_guard<std::mutex> l(g_regM);
        for (auto &kv : g_mems) {
            const MemRec &m = kv.second;
            if (m.fd < 0) continue;
            MrhandBufInfo bi{};
            bi.magic = MRHAND_MAGIC;
            bi.generation = g_generation;
            bi.id = m.id;
            bi.size = m.size;
            bi.handleType = m.handleType;
            bi.bindOffset = m.bindOffset;
            auto it = g_images.find(m.image);
            if (m.image && it != g_images.end()) {
                const ImageRec &ir = it->second;
                bi.hasImage = 1;
                bi.format = ir.ci.format;
                bi.width = ir.ci.extent.width;
                bi.height = ir.ci.extent.height;
                bi.arrayLayers = ir.ci.arrayLayers;
                bi.tiling = ir.ci.tiling;
                bi.usage = ir.ci.usage;
                bi.modifier = ir.modifier;
                bi.hasModifier = ir.hasModifier;
                bi.planeCount = ir.planeCount;
                for (uint32_t p = 0; p < ir.planeCount; p++) {
                    bi.planeOffset[p] = ir.planes[p].offset;
                    bi.planeRowPitch[p] = ir.planes[p].rowPitch;
                }
            }
            out.push_back({bi, m.fd});
        }
    }
    for (auto &e : out) {
        iovec iov{&e.first, sizeof e.first};
        char ctrl[CMSG_SPACE(sizeof(int))] = {};
        msghdr mh{};
        mh.msg_iov = &iov;
        mh.msg_iovlen = 1;
        mh.msg_control = ctrl;
        mh.msg_controllen = sizeof ctrl;
        cmsghdr *cm = CMSG_FIRSTHDR(&mh);
        cm->cmsg_level = SOL_SOCKET;
        cm->cmsg_type = SCM_RIGHTS;
        cm->cmsg_len = CMSG_LEN(sizeof(int));
        memcpy(CMSG_DATA(cm), &e.second, sizeof(int));
        if (sendmsg(c, &mh, MSG_NOSIGNAL) < 0) break;
    }
    MrhandBufInfo end{};
    end.magic = MRHAND_MAGIC;
    end.id = 0;   // terminator
    end.generation = g_generation;
    send(c, &end, sizeof end, MSG_NOSIGNAL);
    close(c);
}

static void serverThread() {
    const char *rt = getenv("XDG_RUNTIME_DIR");
    std::string dir = std::string(rt ? rt : "/tmp") + "/mrhand";
    mkdir(dir.c_str(), 0700);
    chmod(dir.c_str(), 0700);
    std::string path = dir + "/cam.sock";
    unlink(path.c_str());
    int s = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    sockaddr_un a{};
    a.sun_family = AF_UNIX;
    snprintf(a.sun_path, sizeof a.sun_path, "%s", path.c_str());
    if (s < 0 || bind(s, (sockaddr *)&a, sizeof a) != 0 || listen(s, 4) != 0) {
        LOG("socket server failed: %s", strerror(errno));
        if (s >= 0) close(s);
        return;
    }
    chmod(path.c_str(), 0600);
    LOG("serving %s", path.c_str());
    while (true) {
        int c = accept4(s, nullptr, nullptr, SOCK_CLOEXEC);
        if (c < 0) {
            if (errno == EINTR) continue;
            LOG("accept: %s", strerror(errno));
            return;
        }
        serveClient(c);
    }
}
static std::once_flag g_serverOnce;

// ---------------------------------------------------------------- hooks
static VKAPI_ATTR VkResult VKAPI_CALL CreateImage(VkDevice dev, const VkImageCreateInfo *ci,
                                                  const VkAllocationCallbacks *a, VkImage *out) {
    DeviceData *d;
    { std::lock_guard<std::mutex> l(g_mapM); d = g_dev[key(dev)]; }
    VkResult r = d->CreateImage(dev, ci, a, out);
    if (r != VK_SUCCESS) return r;
    auto *ext = (const VkExternalMemoryImageCreateInfo *)findPNext(
        ci->pNext, VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO);
    if (!ext) return r;   // only external images are interesting
    ImageRec ir;
    ir.ci = *ci;
    ir.ci.pNext = nullptr;
    ir.ci.pQueueFamilyIndices = nullptr;
    ir.extTypes = ext->handleTypes;
    if (auto *mx = (const VkImageDrmFormatModifierExplicitCreateInfoEXT *)findPNext(
            ci->pNext, VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT)) {
        ir.hasModifier = true;
        ir.modifier = mx->drmFormatModifier;
        ir.planeCount = std::min<uint32_t>(mx->drmFormatModifierPlaneCount, MRHAND_MAX_PLANES);
        for (uint32_t p = 0; p < ir.planeCount; p++) ir.planes[p] = mx->pPlaneLayouts[p];
    }
    LOG("CreateImage %p ext=0x%x fmt=%d %ux%ux%u layers=%u tiling=%d usage=0x%x mod=%s0x%llx planes=%u",
        (void *)*out, ext->handleTypes, ci->format, ci->extent.width, ci->extent.height, ci->extent.depth,
        ci->arrayLayers, ci->tiling, ci->usage, ir.hasModifier ? "" : "(none)", (unsigned long long)ir.modifier,
        ir.planeCount);
    for (uint32_t p = 0; p < ir.planeCount; p++)
        LOG("    plane%u offset=%llu rowPitch=%llu size=%llu", p, (unsigned long long)ir.planes[p].offset,
            (unsigned long long)ir.planes[p].rowPitch, (unsigned long long)ir.planes[p].size);
    std::lock_guard<std::mutex> l(g_regM);
    g_images[*out] = ir;
    return r;
}

static VKAPI_ATTR void VKAPI_CALL DestroyImage(VkDevice dev, VkImage img, const VkAllocationCallbacks *a) {
    DeviceData *d;
    { std::lock_guard<std::mutex> l(g_mapM); d = g_dev[key(dev)]; }
    {
        std::lock_guard<std::mutex> l(g_regM);
        g_images.erase(img);
    }
    d->DestroyImage(dev, img, a);
}

static VKAPI_ATTR VkResult VKAPI_CALL AllocateMemory(VkDevice dev, const VkMemoryAllocateInfo *ai,
                                                     const VkAllocationCallbacks *a, VkDeviceMemory *out) {
    DeviceData *d;
    { std::lock_guard<std::mutex> l(g_mapM); d = g_dev[key(dev)]; }
    auto *imp = (const VkImportMemoryFdInfoKHR *)findPNext(ai->pNext, VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR);
    int dupfd = -1;
    if (imp && imp->fd >= 0) dupfd = fcntl(imp->fd, F_DUPFD_CLOEXEC, 0);   // before the driver takes ownership
    VkResult r = d->AllocateMemory(dev, ai, a, out);
    if (r != VK_SUCCESS || !imp) {
        if (dupfd >= 0) close(dupfd);
        return r;
    }
    auto *ded = (const VkMemoryDedicatedAllocateInfo *)findPNext(ai->pNext,
                                                                 VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO);
    MemRec m;
    m.fd = dupfd;
    m.handleType = imp->handleType;
    off_t sz = dupfd >= 0 ? lseek(dupfd, 0, SEEK_END) : -1;
    if (dupfd >= 0) lseek(dupfd, 0, SEEK_SET);
    m.size = sz > 0 ? (uint64_t)sz : ai->allocationSize;
    if (ded && ded->image) m.image = ded->image;
    {
        std::lock_guard<std::mutex> l(g_regM);
        m.id = g_nextId++;
        g_mems[*out] = m;
        g_generation++;
    }
    LOG("ImportMemory #%u mem=%p handleType=0x%x size=%llu alloc=%llu dedicatedImage=%p dup=%d", m.id, (void *)*out,
        imp->handleType, (unsigned long long)m.size, (unsigned long long)ai->allocationSize, (void *)m.image, dupfd);
    std::call_once(g_serverOnce, [] { std::thread(serverThread).detach(); });
    return r;
}

static VKAPI_ATTR void VKAPI_CALL FreeMemory(VkDevice dev, VkDeviceMemory mem, const VkAllocationCallbacks *a) {
    DeviceData *d;
    { std::lock_guard<std::mutex> l(g_mapM); d = g_dev[key(dev)]; }
    {
        std::lock_guard<std::mutex> l(g_regM);
        auto it = g_mems.find(mem);
        if (it != g_mems.end()) {
            LOG("FreeMemory #%u", it->second.id);
            if (it->second.fd >= 0) close(it->second.fd);
            g_mems.erase(it);
            g_generation++;
        }
    }
    d->FreeMemory(dev, mem, a);
}

static void noteBind(VkImage img, VkDeviceMemory mem, VkDeviceSize off) {
    std::lock_guard<std::mutex> l(g_regM);
    auto it = g_mems.find(mem);
    if (it == g_mems.end()) return;
    it->second.image = img;
    it->second.bindOffset = off;
    g_generation++;
    auto ii = g_images.find(img);
    if (ii != g_images.end())
        LOG("Bind #%u -> image %p fmt=%d %ux%u off=%llu", it->second.id, (void *)img, ii->second.ci.format,
            ii->second.ci.extent.width, ii->second.ci.extent.height, (unsigned long long)off);
}

static VKAPI_ATTR VkResult VKAPI_CALL BindImageMemory(VkDevice dev, VkImage img, VkDeviceMemory mem,
                                                      VkDeviceSize off) {
    DeviceData *d;
    { std::lock_guard<std::mutex> l(g_mapM); d = g_dev[key(dev)]; }
    VkResult r = d->BindImageMemory(dev, img, mem, off);
    if (r == VK_SUCCESS) noteBind(img, mem, off);
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL BindImageMemory2(VkDevice dev, uint32_t n, const VkBindImageMemoryInfo *b) {
    DeviceData *d;
    { std::lock_guard<std::mutex> l(g_mapM); d = g_dev[key(dev)]; }
    VkResult r = d->BindImageMemory2(dev, n, b);
    if (r == VK_SUCCESS)
        for (uint32_t i = 0; i < n; i++) noteBind(b[i].image, b[i].memory, b[i].memoryOffset);
    return r;
}

// ---------------------------------------------------------------- instance / device plumbing
EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL mrhand_GetDeviceProcAddr(VkDevice dev, const char *name);
EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL mrhand_GetInstanceProcAddr(VkInstance inst, const char *name);

static VKAPI_ATTR VkResult VKAPI_CALL CreateInstance(const VkInstanceCreateInfo *ci, const VkAllocationCallbacks *a,
                                                     VkInstance *out) {
    auto *chain = (VkLayerInstanceCreateInfo *)ci->pNext;
    while (chain && !(chain->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO &&
                      chain->function == VK_LAYER_LINK_INFO))
        chain = (VkLayerInstanceCreateInfo *)chain->pNext;
    if (!chain) return VK_ERROR_INITIALIZATION_FAILED;
    PFN_vkGetInstanceProcAddr gipa = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;
    auto create = (PFN_vkCreateInstance)gipa(VK_NULL_HANDLE, "vkCreateInstance");
    VkResult r = create(ci, a, out);
    if (r != VK_SUCCESS) return r;
    auto *id = new InstanceData{gipa, (PFN_vkDestroyInstance)gipa(*out, "vkDestroyInstance")};
    std::lock_guard<std::mutex> l(g_mapM);
    g_inst[key(*out)] = id;
    LOG("CreateInstance");
    return r;
}

static VKAPI_ATTR void VKAPI_CALL DestroyInstance(VkInstance inst, const VkAllocationCallbacks *a) {
    InstanceData *id;
    {
        std::lock_guard<std::mutex> l(g_mapM);
        id = g_inst[key(inst)];
        g_inst.erase(key(inst));
    }
    id->DestroyInstance(inst, a);
    delete id;
}

static VKAPI_ATTR VkResult VKAPI_CALL CreateDevice(VkPhysicalDevice phys, const VkDeviceCreateInfo *ci,
                                                   const VkAllocationCallbacks *a, VkDevice *out) {
    auto *chain = (VkLayerDeviceCreateInfo *)ci->pNext;
    while (chain && !(chain->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO &&
                      chain->function == VK_LAYER_LINK_INFO))
        chain = (VkLayerDeviceCreateInfo *)chain->pNext;
    if (!chain) return VK_ERROR_INITIALIZATION_FAILED;
    PFN_vkGetInstanceProcAddr gipa = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr gdpa = chain->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;
    auto create = (PFN_vkCreateDevice)gipa(VK_NULL_HANDLE, "vkCreateDevice");
    VkResult r = create(phys, ci, a, out);
    if (r != VK_SUCCESS) return r;
    auto *d = new DeviceData{};
    d->device = *out;
    d->gdpa = gdpa;
#define DF(n) d->n = (PFN_vk##n)gdpa(*out, "vk" #n);
    DF(DestroyDevice) DF(CreateImage) DF(DestroyImage) DF(AllocateMemory) DF(FreeMemory) DF(BindImageMemory)
    DF(BindImageMemory2)
#undef DF
    if (!d->BindImageMemory2) d->BindImageMemory2 = (PFN_vkBindImageMemory2)gdpa(*out, "vkBindImageMemory2KHR");
    std::lock_guard<std::mutex> l(g_mapM);
    g_dev[key(*out)] = d;
    LOG("CreateDevice");
    return r;
}

static VKAPI_ATTR void VKAPI_CALL DestroyDevice(VkDevice dev, const VkAllocationCallbacks *a) {
    DeviceData *d;
    {
        std::lock_guard<std::mutex> l(g_mapM);
        d = g_dev[key(dev)];
        g_dev.erase(key(dev));
    }
    d->DestroyDevice(dev, a);
    delete d;
}

static PFN_vkVoidFunction deviceHook(const char *n) {
#define H(x) if (!strcmp(n, "vk" #x)) return (PFN_vkVoidFunction)x;
    H(DestroyDevice) H(CreateImage) H(DestroyImage) H(AllocateMemory) H(FreeMemory) H(BindImageMemory)
    H(BindImageMemory2)
    if (!strcmp(n, "vkBindImageMemory2KHR")) return (PFN_vkVoidFunction)BindImageMemory2;
#undef H
    return nullptr;
}

EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL mrhand_GetDeviceProcAddr(VkDevice dev, const char *name) {
    if (!strcmp(name, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)mrhand_GetDeviceProcAddr;
    DeviceData *d;
    { std::lock_guard<std::mutex> l(g_mapM); d = g_dev[key(dev)]; }
    PFN_vkVoidFunction next = d->gdpa(dev, name);
    if (!next) return nullptr;
    if (auto h = deviceHook(name)) return h;
    return next;
}

EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL mrhand_GetInstanceProcAddr(VkInstance inst, const char *name) {
    if (!strcmp(name, "vkGetInstanceProcAddr")) return (PFN_vkVoidFunction)mrhand_GetInstanceProcAddr;
    if (!strcmp(name, "vkCreateInstance")) return (PFN_vkVoidFunction)CreateInstance;
    if (!strcmp(name, "vkDestroyInstance")) return (PFN_vkVoidFunction)DestroyInstance;
    if (!strcmp(name, "vkCreateDevice")) return (PFN_vkVoidFunction)CreateDevice;
    if (!strcmp(name, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)mrhand_GetDeviceProcAddr;
    if (!inst) return nullptr;
    InstanceData *id;
    { std::lock_guard<std::mutex> l(g_mapM); id = g_inst[key(inst)]; }
    if (auto h = deviceHook(name)) {   // device functions queried through the instance
        if (id->gipa(inst, name)) return h;
        return nullptr;
    }
    return id->gipa(inst, name);
}

EXPORT VKAPI_ATTR VkResult VKAPI_CALL mrhand_NegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *v) {
    if (!processAllowed()) return VK_ERROR_INITIALIZATION_FAILED;
    if (v->loaderLayerInterfaceVersion < 2) return VK_ERROR_INITIALIZATION_FAILED;
    v->loaderLayerInterfaceVersion = 2;
    v->pfnGetInstanceProcAddr = mrhand_GetInstanceProcAddr;
    v->pfnGetDeviceProcAddr = mrhand_GetDeviceProcAddr;
    v->pfnGetPhysicalDeviceProcAddr = nullptr;
    return VK_SUCCESS;
}
