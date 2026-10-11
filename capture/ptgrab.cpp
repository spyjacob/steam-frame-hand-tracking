// ptgrab — read the live passthrough camera buffers through our own GPU queue.
//
// The buffers XRService shares with vrcompositor are NV12 in Qualcomm's
// compressed tile format (DRM modifier 0x0500000000000001), which the CPU
// cannot read.  ptgrab gets the dup'd dma-buf fds from VK_LAYER_MRHAND,
// imports them into *its own* Vulkan device as images with that modifier, and
// copies the luma plane into host-visible memory with vkCmdCopyImageToBuffer
// (the driver decompresses).  Nothing is submitted on the compositor's queues.
//
//   ptgrab identify OUTDIR      dump every passthrough buffer's Y plane twice (PGM)
#include <vulkan/vulkan.h>

#include <algorithm>
#include <ctime>
#include <fcntl.h>
#include <sys/mman.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "handcam_proto.h"

#define CHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s:%d %s -> %d\n", __FILE__, __LINE__, #x, r_); exit(1); } } while (0)

struct Buf { MrhandBufInfo info; int fd; };

static std::vector<Buf> fetchBuffers() {
    const char *rt = getenv("XDG_RUNTIME_DIR");
    std::string path = std::string(rt ? rt : "/tmp") + "/mrhand/cam.sock";
    int s = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    sockaddr_un a{};
    a.sun_family = AF_UNIX;
    snprintf(a.sun_path, sizeof a.sun_path, "%s", path.c_str());
    if (connect(s, (sockaddr *)&a, sizeof a) != 0) { perror("connect cam.sock"); exit(1); }
    std::vector<Buf> out;
    while (true) {
        MrhandBufInfo bi;
        char ctrl[CMSG_SPACE(sizeof(int))];
        iovec iov{&bi, sizeof bi};
        msghdr mh{};
        mh.msg_iov = &iov;
        mh.msg_iovlen = 1;
        mh.msg_control = ctrl;
        mh.msg_controllen = sizeof ctrl;
        ssize_t n = recvmsg(s, &mh, MSG_WAITALL | MSG_CMSG_CLOEXEC);
        if (n != (ssize_t)sizeof bi || bi.magic != MRHAND_MAGIC || bi.id == 0) break;
        int fd = -1;
        for (cmsghdr *c = CMSG_FIRSTHDR(&mh); c; c = CMSG_NXTHDR(&mh, c))
            if (c->cmsg_type == SCM_RIGHTS) memcpy(&fd, CMSG_DATA(c), sizeof fd);
        out.push_back({bi, fd});
    }
    close(s);
    return out;
}

struct Gpu {
    VkInstance inst;
    VkPhysicalDevice phys;
    VkDevice dev;
    VkQueue queue;
    uint32_t qfam = 0;
    VkCommandPool pool;
    VkPhysicalDeviceMemoryProperties mp;
    PFN_vkGetMemoryFdPropertiesKHR getFdProps;

    void init() {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "ptgrab";
        app.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        CHECK(vkCreateInstance(&ici, nullptr, &inst));
        uint32_t n = 1;
        vkEnumeratePhysicalDevices(inst, &n, &phys);
        vkGetPhysicalDeviceMemoryProperties(phys, &mp);
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qp(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &qn, qp.data());
        for (uint32_t i = 0; i < qn; i++)
            if (qp[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) { qfam = i; break; }
        float prio = 0.0f;   // lowest priority: never compete with the compositor
        VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        q.queueFamilyIndex = qfam;
        q.queueCount = 1;
        q.pQueuePriorities = &prio;
        const char *exts[] = {"VK_KHR_external_memory_fd", "VK_EXT_external_memory_dma_buf",
                              "VK_EXT_image_drm_format_modifier", "VK_EXT_queue_family_foreign"};
        VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        f11.samplerYcbcrConversion = VK_TRUE;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f11};
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &q;
        dci.enabledExtensionCount = 4;
        dci.ppEnabledExtensionNames = exts;
        CHECK(vkCreateDevice(phys, &dci, nullptr, &dev));
        vkGetDeviceQueue(dev, qfam, 0, &queue);
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = qfam;
        CHECK(vkCreateCommandPool(dev, &pci, nullptr, &pool));
        getFdProps = (PFN_vkGetMemoryFdPropertiesKHR)vkGetDeviceProcAddr(dev, "vkGetMemoryFdPropertiesKHR");
    }
    uint32_t memType(uint32_t bits, VkMemoryPropertyFlags want) {
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
        fprintf(stderr, "no memory type for bits 0x%x\n", bits);
        exit(1);
    }
};

struct CamImage {
    MrhandBufInfo info;
    VkImage img;
    VkDeviceMemory mem;
};

static CamImage importImage(Gpu &g, const Buf &b) {
    CamImage ci{b.info};
    VkSubresourceLayout planes[MRHAND_MAX_PLANES]{};
    for (uint32_t p = 0; p < b.info.planeCount; p++) {
        planes[p].offset = b.info.planeOffset[p];
        planes[p].rowPitch = b.info.planeRowPitch[p];
    }
    VkImageDrmFormatModifierExplicitCreateInfoEXT mod{VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT};
    mod.drmFormatModifier = b.info.modifier;
    mod.drmFormatModifierPlaneCount = b.info.planeCount;
    mod.pPlaneLayouts = planes;
    VkExternalMemoryImageCreateInfo ext{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO, &mod,
                                        VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
    VkImageCreateInfo ic{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, &ext};
    ic.imageType = VK_IMAGE_TYPE_2D;
    ic.format = (VkFormat)b.info.format;
    ic.extent = {b.info.width, b.info.height, 1};
    ic.mipLevels = 1;
    ic.arrayLayers = b.info.arrayLayers ? b.info.arrayLayers : 1;
    ic.samples = VK_SAMPLE_COUNT_1_BIT;
    ic.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
    ic.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    CHECK(vkCreateImage(g.dev, &ic, nullptr, &ci.img));
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(g.dev, ci.img, &req);
    int fd = dup(b.fd);   // Vulkan takes ownership of the imported fd
    VkMemoryFdPropertiesKHR fp{VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
    CHECK(g.getFdProps(g.dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, fd, &fp));
    VkMemoryDedicatedAllocateInfo ded{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, nullptr, ci.img};
    VkImportMemoryFdInfoKHR imp{VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR, &ded,
                                VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, fd};
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &imp, std::max<VkDeviceSize>(req.size, b.info.size),
                            g.memType(req.memoryTypeBits & fp.memoryTypeBits, 0)};
    CHECK(vkAllocateMemory(g.dev, &ai, nullptr, &ci.mem));
    CHECK(vkBindImageMemory(g.dev, ci.img, ci.mem, 0));
    return ci;
}

struct HostBuf { VkBuffer buf; VkDeviceMemory mem; void *ptr; VkDeviceSize size; };
static HostBuf hostBuffer(Gpu &g, VkDeviceSize size) {
    HostBuf h{};
    h.size = size;
    VkBufferCreateInfo bc{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bc.size = size;
    bc.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    CHECK(vkCreateBuffer(g.dev, &bc, nullptr, &h.buf));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(g.dev, h.buf, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, req.size,
                            g.memType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                              VK_MEMORY_PROPERTY_HOST_CACHED_BIT)};
    CHECK(vkAllocateMemory(g.dev, &ai, nullptr, &h.mem));
    CHECK(vkBindBufferMemory(g.dev, h.buf, h.mem, 0));
    CHECK(vkMapMemory(g.dev, h.mem, 0, VK_WHOLE_SIZE, 0, &h.ptr));
    return h;
}

// PTGRAB_QFOT=0: no queue-family ownership transfer from FOREIGN (plain read barrier) — banding A/B test
static bool useQfot() {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("PTGRAB_QFOT");
        v = (e && e[0] == '0') ? 0 : 1;
    }
    return v == 1;
}
static uint32_t srcQ() { return useQfot() ? VK_QUEUE_FAMILY_FOREIGN_EXT : VK_QUEUE_FAMILY_IGNORED; }

struct Region { size_t image; uint32_t y0, rows; VkDeviceSize dstOffset; };

// Copy luma rows [y0, y0+rows) of the given images into dst (tightly packed).
static void copyRegions(Gpu &g, std::vector<CamImage> &ims, const std::vector<Region> &regs, HostBuf &dst) {
    static VkCommandBuffer cb = VK_NULL_HANDLE;
    static VkFence fence = VK_NULL_HANDLE;
    if (!cb) {
        VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, g.pool,
                                       VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        CHECK(vkAllocateCommandBuffers(g.dev, &ca, &cb));
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        CHECK(vkCreateFence(g.dev, &fci, nullptr, &fence));
    }
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(cb, 0);
    vkBeginCommandBuffer(cb, &bi);
    for (auto &rg : regs) {
        CamImage &ci = ims[rg.image];
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        b.oldLayout = b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = srcQ();
        b.dstQueueFamilyIndex = useQfot() ? g.qfam : VK_QUEUE_FAMILY_IGNORED;
        b.image = ci.img;
        b.subresourceRange = {VK_IMAGE_ASPECT_PLANE_0_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                             nullptr, 1, &b);
        VkBufferImageCopy r{};
        r.bufferOffset = rg.dstOffset;
        r.imageSubresource = {VK_IMAGE_ASPECT_PLANE_0_BIT, 0, 0, 1};
        r.imageOffset = {0, (int32_t)rg.y0, 0};
        r.imageExtent = {ci.info.width, rg.rows, 1};
        vkCmdCopyImageToBuffer(cb, ci.img, VK_IMAGE_LAYOUT_GENERAL, dst.buf, 1, &r);
    }
    vkEndCommandBuffer(cb);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    vkResetFences(g.dev, 1, &fence);
    CHECK(vkQueueSubmit(g.queue, 1, &si, fence));
    CHECK(vkWaitForFences(g.dev, 1, &fence, VK_TRUE, 2000000000ull));
    VkMappedMemoryRange mr{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, dst.mem, 0, VK_WHOLE_SIZE};
    vkInvalidateMappedMemoryRanges(g.dev, 1, &mr);
}

// Copy plane 0 (luma) of every image into consecutive regions of dst.
static void copyLuma(Gpu &g, std::vector<CamImage> &ims, HostBuf &dst) {
    VkCommandBuffer cb;
    VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, g.pool,
                                   VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    CHECK(vkAllocateCommandBuffers(g.dev, &ca, &cb));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);
    VkDeviceSize off = 0;
    for (auto &ci : ims) {
        // acquire from the foreign (XRService/compositor) owner, read-only
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.srcAccessMask = 0;
        b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = srcQ();
        b.dstQueueFamilyIndex = useQfot() ? g.qfam : VK_QUEUE_FAMILY_IGNORED;
        b.image = ci.img;
        b.subresourceRange = {VK_IMAGE_ASPECT_PLANE_0_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                             nullptr, 1, &b);
        VkBufferImageCopy r{};
        r.bufferOffset = off;
        r.imageSubresource = {VK_IMAGE_ASPECT_PLANE_0_BIT, 0, 0, 1};
        r.imageExtent = {ci.info.width, ci.info.height, 1};
        vkCmdCopyImageToBuffer(cb, ci.img, VK_IMAGE_LAYOUT_GENERAL, dst.buf, 1, &r);
        off += (VkDeviceSize)ci.info.width * ci.info.height;
    }
    vkEndCommandBuffer(cb);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    VkFence fence;
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    CHECK(vkCreateFence(g.dev, &fci, nullptr, &fence));
    CHECK(vkQueueSubmit(g.queue, 1, &si, fence));
    CHECK(vkWaitForFences(g.dev, 1, &fence, VK_TRUE, 2000000000ull));
    vkDestroyFence(g.dev, fence, nullptr);
    vkFreeCommandBuffers(g.dev, g.pool, 1, &cb);
    VkMappedMemoryRange mr{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, dst.mem, 0, VK_WHOLE_SIZE};
    vkInvalidateMappedMemoryRanges(g.dev, 1, &mr);
}

// GPU downscale of one camera's luma plane: plane0 -> R8 full -> R8 half-ish -> R8 out
// (two linear-filtered blits, the last one an exact 2x so it averages 2x2), then
// a small copy to host memory.  Runs on our own low-priority queue.
struct Downscaler {
    uint32_t outSize = 640;
    VkImage full = VK_NULL_HANDLE, mid = VK_NULL_HANDLE, out = VK_NULL_HANDLE;
    VkDeviceMemory mem[3] = {};
    HostBuf host{};
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    VkImage makeImage(Gpu &g, uint32_t w, uint32_t h, VkDeviceMemory &m) {
        VkImageCreateInfo ic{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ic.imageType = VK_IMAGE_TYPE_2D;
        ic.format = VK_FORMAT_R8_UNORM;
        ic.extent = {w, h, 1};
        ic.mipLevels = ic.arrayLayers = 1;
        ic.samples = VK_SAMPLE_COUNT_1_BIT;
        ic.tiling = VK_IMAGE_TILING_OPTIMAL;
        ic.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        VkImage img;
        CHECK(vkCreateImage(g.dev, &ic, nullptr, &img));
        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(g.dev, img, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, req.size,
                                g.memType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
        CHECK(vkAllocateMemory(g.dev, &ai, nullptr, &m));
        CHECK(vkBindImageMemory(g.dev, img, m, 0));
        return img;
    }
    void init(Gpu &g, uint32_t W, uint32_t size) {
        outSize = size;
        full = makeImage(g, W, W, mem[0]);
        mid = makeImage(g, size * 2, size * 2, mem[1]);
        out = makeImage(g, size, size, mem[2]);
        host = hostBuffer(g, (VkDeviceSize)size * size);
        VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, g.pool,
                                       VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        CHECK(vkAllocateCommandBuffers(g.dev, &ca, &cb));
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        CHECK(vkCreateFence(g.dev, &fci, nullptr, &fence));
    }
    static void barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to, VkAccessFlags src,
                        VkAccessFlags dst, VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                        uint32_t srcQ = VK_QUEUE_FAMILY_IGNORED, uint32_t dstQ = VK_QUEUE_FAMILY_IGNORED) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.srcAccessMask = src;
        b.dstAccessMask = dst;
        b.oldLayout = from;
        b.newLayout = to;
        b.srcQueueFamilyIndex = srcQ;
        b.dstQueueFamilyIndex = dstQ;
        b.image = img;
        b.subresourceRange = {aspect, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr,
                             0, nullptr, 1, &b);
    }
    static void blit(VkCommandBuffer cb, VkImage src, uint32_t sw, VkImage dst, uint32_t dw) {
        VkImageBlit r{};
        r.srcSubresource = r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        r.srcOffsets[1] = {(int32_t)sw, (int32_t)sw, 1};
        r.dstOffsets[1] = {(int32_t)dw, (int32_t)dw, 1};
        vkCmdBlitImage(cb, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r,
                       VK_FILTER_LINEAR);
    }
    // returns pointer to outSize*outSize luma bytes (valid until the next call)
    const uint8_t *run(Gpu &g, CamImage &src) {
        const uint32_t W = src.info.width, M = outSize * 2, S = outSize;
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkResetCommandBuffer(cb, 0);
        vkBeginCommandBuffer(cb, &bi);
        barrier(cb, src.img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_TRANSFER_READ_BIT,
                VK_IMAGE_ASPECT_PLANE_0_BIT, srcQ(), useQfot() ? g.qfam : VK_QUEUE_FAMILY_IGNORED);
        barrier(cb, full, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
        VkImageCopy c{};
        c.srcSubresource = {VK_IMAGE_ASPECT_PLANE_0_BIT, 0, 0, 1};
        c.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c.extent = {W, W, 1};
        vkCmdCopyImage(cb, src.img, VK_IMAGE_LAYOUT_GENERAL, full, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
        barrier(cb, full, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        barrier(cb, mid, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
        blit(cb, full, W, mid, M);
        barrier(cb, mid, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        barrier(cb, out, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
        blit(cb, mid, M, out, S);
        barrier(cb, out, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferImageCopy r{};
        r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        r.imageExtent = {S, S, 1};
        vkCmdCopyImageToBuffer(cb, out, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, host.buf, 1, &r);
        vkEndCommandBuffer(cb);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        vkResetFences(g.dev, 1, &fence);
        CHECK(vkQueueSubmit(g.queue, 1, &si, fence));
        CHECK(vkWaitForFences(g.dev, 1, &fence, VK_TRUE, 2000000000ull));
        VkMappedMemoryRange mr{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, host.mem, 0, VK_WHOLE_SIZE};
        vkInvalidateMappedMemoryRanges(g.dev, 1, &mr);
        return (const uint8_t *)host.ptr;
    }
};

static uint64_t monoNs() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

// stream: detect which buffer XRService just refreshed (strip compare), copy
// that frame's luma into the shm slot of its camera.  Cameras = the two halves
// of the id-sorted buffer list (XRService allocates each camera's ring together).
static int stream(Gpu &g, std::vector<CamImage> &ims, int maxFps, uint32_t outSize) {
    std::sort(ims.begin(), ims.end(), [](const CamImage &a, const CamImage &b) { return a.info.id < b.info.id; });
    const uint32_t W = ims[0].info.width, H = ims[0].info.height;
    const uint32_t S0 = H / 2 - 8, SR = 16;   // probe strip
    const VkDeviceSize strip = (VkDeviceSize)W * SR;
    const uint32_t OW = outSize ? outSize : W, OH = outSize ? outSize : H;
    const VkDeviceSize frame = (VkDeviceSize)OW * OH;
    HostBuf probe = hostBuffer(g, strip * ims.size());
    HostBuf full{};
    Downscaler ds;
    if (outSize) ds.init(g, W, outSize);
    else full = hostBuffer(g, frame);
    std::vector<std::vector<uint8_t>> last(ims.size(), std::vector<uint8_t>(strip));
    std::vector<int> camOf(ims.size());
    for (size_t i = 0; i < ims.size(); i++) camOf[i] = i < ims.size() / 2 ? 0 : 1;

    std::string shmPath = "/dev/shm/mrhand-pt";
    size_t total = sizeof(MrhandPtHeader) + 2 * frame;
    int fd = open(shmPath.c_str(), O_RDWR | O_CREAT, 0600);
    if (fd < 0 || ftruncate(fd, total) != 0) { perror("shm"); return 1; }
    auto *base = (uint8_t *)mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    auto *hdr = (MrhandPtHeader *)base;
    memset(hdr, 0, sizeof *hdr);
    hdr->magic = MRHAND_PT_MAGIC;
    hdr->width = OW;   // published frame size (downscaled on the GPU when outSize > 0)
    hdr->height = OH;
    hdr->cams = 2;
    uint8_t *slots = base + sizeof(MrhandPtHeader);
    printf("streaming %zu buffers (%ux%u -> %ux%u) -> %s\n", ims.size(), W, H, OW, OH, shmPath.c_str());
    fflush(stdout);

    std::vector<Region> probeRegs;
    for (size_t i = 0; i < ims.size(); i++) probeRegs.push_back({i, S0, SR, i * strip});
    uint64_t frames[2] = {0, 0}, lastPub[2] = {0, 0}, statT = monoNs(), copyNsSum = 0, copies = 0;
    const uint64_t minGap = maxFps > 0 ? 1000000000ull / maxFps : 0;
    bool primed = false;
    // A buffer that changed is published once a later probe shows it stable
    // (XRService writes take a few ms; copying mid-write would tear).
    std::vector<uint64_t> pendingSince(ims.size(), 0);
    while (true) {
        copyRegions(g, ims, probeRegs, probe);
        uint64_t tNow = monoNs();
        for (size_t i = 0; i < ims.size(); i++) {
            const uint8_t *p = (const uint8_t *)probe.ptr + i * strip;
            bool changed = memcmp(p, last[i].data(), strip) != 0;
            memcpy(last[i].data(), p, strip);
            if (!primed) continue;
            if (changed) {
                if (!pendingSince[i]) pendingSince[i] = tNow;
                continue;
            }
            if (!pendingSince[i]) continue;
            uint64_t tDetect = pendingSince[i];
            pendingSince[i] = 0;
            int cam = camOf[i];
            if (getenv("PTGRAB_TRACE")) {
                static uint64_t tPrev[2] = {0, 0};
                fprintf(stderr, "t=%.3f cam%d buf#%u dt=%.2fms\n", tDetect / 1e9, cam, ims[i].info.id,
                        tPrev[cam] ? (tDetect - tPrev[cam]) / 1e6 : 0.0);
                tPrev[cam] = tDetect;
            }
            frames[cam]++;
            if (tDetect - lastPub[cam] < minGap) continue;
            lastPub[cam] = tDetect;
            uint64_t t0 = monoNs();
            const uint8_t *px;
            if (outSize) {
                px = ds.run(g, ims[i]);
            } else {
                copyRegions(g, ims, {{i, 0, H, 0}}, full);
                px = (const uint8_t *)full.ptr;
            }
            copyNsSum += monoNs() - t0;
            copies++;
            MrhandPtSlot &s = hdr->slot[cam];
            __atomic_store_n(&s.seq, s.seq + 1, __ATOMIC_RELEASE);   // odd: writing
            memcpy(slots + cam * frame, px, frame);
            s.tDetectNs = tDetect;
            s.bufferId = ims[i].info.id;
            s.frameCount = frames[cam];
            __atomic_store_n(&s.seq, s.seq + 1, __ATOMIC_RELEASE);   // even: done
        }
        primed = true;
        uint64_t now = monoNs();
        if (now - statT > 5000000000ull) {
            double dt = (now - statT) / 1e9;
            printf("cam0 %.1f fps  cam1 %.1f fps  full-copy avg %.2f ms\n", hdr->stat[0] / dt, hdr->stat[1] / dt,
                   copies ? copyNsSum / 1e6 / copies : 0.0);
            fflush(stdout);
            hdr->stat[0] = hdr->stat[1] = 0;
            copyNsSum = copies = 0;
            statT = now;
        }
        static uint64_t prev[2] = {0, 0};
        for (int c = 0; c < 2; c++) { hdr->stat[c] += frames[c] - prev[c]; prev[c] = frames[c]; }
        static const int pollUs = getenv("PTGRAB_POLL_US") ? atoi(getenv("PTGRAB_POLL_US")) : 4000;
        std::this_thread::sleep_for(std::chrono::microseconds(pollUs));
    }
}

int main(int argc, char **argv) {
    if (argc < 2 || (strcmp(argv[1], "identify") && strcmp(argv[1], "stream")) ||
        (!strcmp(argv[1], "identify") && argc < 3)) {
        fprintf(stderr, "usage: ptgrab identify OUTDIR | ptgrab stream [maxfps] [outsize=640]\n");
        return 2;
    }
    auto bufs = fetchBuffers();
    Gpu g;
    g.init();
    std::vector<CamImage> ims;
    for (auto &b : bufs)
        if (b.info.hasImage && b.info.width == 2464 && b.info.height == 2464) {
            ims.push_back(importImage(g, b));
            printf("imported #%u %ux%u mod=0x%llx\n", b.info.id, b.info.width, b.info.height,
                   (unsigned long long)b.info.modifier);
        }
    if (ims.empty()) { fprintf(stderr, "no passthrough buffers\n"); return 1; }
    if (!strcmp(argv[1], "stream"))   // ptgrab stream [maxfps] [outsize]  (outsize 0 = full resolution)
        return stream(g, ims, argc > 2 ? atoi(argv[2]) : 0, argc > 3 ? atoi(argv[3]) : 640);
    std::string out = argv[2];
    VkDeviceSize one = 2464ull * 2464;
    HostBuf hb = hostBuffer(g, one * ims.size());
    std::vector<uint8_t> first(one * ims.size());
    for (int pass = 0; pass < 2; pass++) {
        auto t0 = std::chrono::steady_clock::now();
        copyLuma(g, ims, hb);
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        printf("pass %d: copied %zu luma planes in %.2f ms\n", pass, ims.size(), ms);
        const uint8_t *p = (const uint8_t *)hb.ptr;
        for (size_t i = 0; i < ims.size(); i++) {
            const uint8_t *y = p + i * one;
            double sum = 0;
            for (VkDeviceSize k = 0; k < one; k += 97) sum += y[k];
            bool changed = pass == 1 && memcmp(first.data() + i * one, y, one) != 0;
            printf("  #%u mean %.1f%s\n", ims[i].info.id, sum / (one / 97.0), pass == 1 ? (changed ? " CHANGED" : " same") : "");
            char fn[512];
            snprintf(fn, sizeof fn, "%s/pt%u_pass%d.pgm", out.c_str(), ims[i].info.id, pass);
            if (FILE *f = fopen(fn, "wb")) {
                fprintf(f, "P5 2464 2464 255\n");
                fwrite(y, 1, one, f);
                fclose(f);
            }
        }
        memcpy(first.data(), p, one * ims.size());
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }
    return 0;
}
