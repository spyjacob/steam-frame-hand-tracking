// vrsetting — get/set a SteamVR bool setting through the public IVRSettings API.
//   vrsetting <section> <key>            print
//   vrsetting <section> <key> 0|1        set
//   vrsetting <section> <key> -          remove (fall back to driver/default value)
//   vrsetting -i <section> <key> [N]     int get/set
#include <openvr.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char **argv) {
    bool isInt = argc > 1 && !strcmp(argv[1], "-i");
    if (isInt) { argv++; argc--; }
    if (argc < 3) { fprintf(stderr, "usage: vrsetting section key [0|1]\n"); return 2; }
    vr::EVRInitError ie;
    if (!vr::VR_Init(&ie, vr::VRApplication_Background)) { fprintf(stderr, "VR_Init: %d\n", ie); return 1; }
    vr::EVRSettingsError se;
    if (isInt) {
        if (argc > 3) {
            vr::VRSettings()->SetInt32(argv[1], argv[2], atoi(argv[3]), &se);
            printf("set %s/%s=%d err=%d\n", argv[1], argv[2], atoi(argv[3]), se);
        }
        printf("%s/%s=%d\n", argv[1], argv[2], vr::VRSettings()->GetInt32(argv[1], argv[2], &se));
        vr::VR_Shutdown();
        return 0;
    }
    if (argc > 3 && argv[3][0] == '-') {
        vr::VRSettings()->RemoveKeyInSection(argv[1], argv[2], &se);
        printf("removed %s/%s err=%d\n", argv[1], argv[2], se);
    } else if (argc > 3) {
        vr::VRSettings()->SetBool(argv[1], argv[2], atoi(argv[3]) != 0, &se);
        printf("set %s/%s=%d err=%d\n", argv[1], argv[2], atoi(argv[3]), se);
    }
    bool v = vr::VRSettings()->GetBool(argv[1], argv[2], &se);
    printf("%s/%s=%d err=%d\n", argv[1], argv[2], v, se);
    vr::VR_Shutdown();
    return 0;
}
