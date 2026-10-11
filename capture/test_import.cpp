// test_import — stand-in for vrcompositor's camera import, for testing VK_LAYER_MRHAND
// without touching SteamVR.  Build as a binary named "vrcompositor" in a scratch dir.
//
// Creates a memfd-backed udmabuf holding a 1056x1024 NV12 test pattern (row
// pitch 1152, like the real SLAM buffers), imports it into Vulkan as an
// external LINEAR-modifier G8_B8R8_2PLANE_420 image, binds it, then idles for
// N seconds so a consumer can connect to the layer's socket.
#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <linux/udmabuf.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

#define CHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s -> %d\n", #x, r_); return 1; } } while (0)

int main(int argc, char **argv) {
    int secs = argc > 1 ? atoi(argv[1]) : 20;
    const uint32_t W = 1056, H = 1024, P = 1152;
    size_t ySize = (size_t)P * H, size = (ySize * 3 / 2 + 4095) & ~4095ull;

    int mfd = memfd_create("mrhand-test", MFD_ALLOW_SEALING);
    if (ftruncate(mfd, size) != 0) return 1;
    auto *px = (uint8_t *)mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, mfd, 0);
    for (uint32_t y = 0; y < H; y++)
        for (uint32_t x = 0; x < P; x++) px[y * P + x] = (uint8_t)((x / 32 + y / 32) % 2 ? 200 : (x ^ y) & 0x3f);
    memset(px + ySize, 128, size - ySize);
    munmap(px, size);
    fcntl(mfd, F_ADD_SEALS, F_SEAL_SHRINK);
    int ud = open("/dev/udmabuf", O_RDWR);
    udmabuf_create c{};
    c.memfd = mfd;
    c.size = size;
    int dbuf = ioctl(ud, UDMABUF_CREATE, &c);
    if (dbuf < 0) { perror("UDMABUF_CREATE"); return 1; }

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst;
    CHECK(vkCreateInstance(&ici, nullptr, &inst));
    uint32_t n = 1;
    VkPhysicalDevice phys;
    vkEnumeratePhysicalDevices(inst, &n, &phys);
    float prio = 1;
    VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    q.queueCount = 1;
    q.pQueuePriorities = &prio;
    const char *exts[] = {"VK_KHR_external_memory_fd", "VK_EXT_external_memory_dma_buf",
                          "VK_EXT_image_drm_format_modifier"};
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &q;
    dci.enabledExtensionCount = 3;
    dci.ppEnabledExtensionNames = exts;
    VkDevice dev;
    CHECK(vkCreateDevice(phys, &dci, nullptr, &dev));

    VkSubresourceLayout planes[2] = {{0, ySize, P, 0, 0}, {ySize, ySize / 2, P, 0, 0}};
    VkImageDrmFormatModifierExplicitCreateInfoEXT mod{VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT};
    mod.drmFormatModifier = 0;   // DRM_FORMAT_MOD_LINEAR
    mod.drmFormatModifierPlaneCount = 2;
    mod.pPlaneLayouts = planes;
    VkExternalMemoryImageCreateInfo ext{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO, &mod,
                                        VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, &ext};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
    ci.extent = {W, H, 1};
    ci.mipLevels = ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
    ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    VkImage img;
    CHECK(vkCreateImage(dev, &ci, nullptr, &img));

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(dev, img, &req);
    auto getProps = (PFN_vkGetMemoryFdPropertiesKHR)vkGetDeviceProcAddr(dev, "vkGetMemoryFdPropertiesKHR");
    VkMemoryFdPropertiesKHR fp{VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
    CHECK(getProps(dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, dbuf, &fp));
    uint32_t bits = req.memoryTypeBits & fp.memoryTypeBits, mt = 0;
    while (bits && !(bits & 1)) { bits >>= 1; mt++; }
    VkMemoryDedicatedAllocateInfo ded{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, nullptr, img};
    VkImportMemoryFdInfoKHR imp{VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR, &ded,
                                VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, dbuf};
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &imp, std::max<VkDeviceSize>(req.size, size), mt};
    VkDeviceMemory mem;
    CHECK(vkAllocateMemory(dev, &ai, nullptr, &mem));
    CHECK(vkBindImageMemory(dev, img, mem, 0));
    printf("imported test NV12 buffer (%zu bytes); idling %d s\n", size, secs);
    fflush(stdout);
    sleep(secs);
    vkDestroyImage(dev, img, nullptr);
    vkFreeMemory(dev, mem, nullptr);
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
    return 0;
}
