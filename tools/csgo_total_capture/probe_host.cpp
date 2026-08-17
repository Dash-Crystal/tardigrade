#include <dlfcn.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

using CreateInterfaceFn = void *(*)(const char *, int *);
class ProbeCallbacks {
public:
    virtual bool Load(CreateInterfaceFn, CreateInterfaceFn) = 0;
};

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;
    const char *plugin = getenv("TARDIGRADE_PROBE_PLUGIN");
    const char *output = getenv("TARDIGRADE_TOTAL_CAPTURE_OUTPUT");
    if (plugin == nullptr || output == nullptr) return 10;
    void *handle = dlopen(plugin, RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        std::fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 11;
    }
    auto create = reinterpret_cast<CreateInterfaceFn>(
        dlsym(handle, "CreateInterface"));
    if (create == nullptr) return 12;
    int status = -1;
    if (create("ISERVERPLUGINCALLBACKS003", &status) != nullptr) return 13;
    auto *callbacks = static_cast<ProbeCallbacks *>(
        create("ISERVERPLUGINCALLBACKS004", &status));
    if (callbacks == nullptr || status != 0) return 14;
    if (callbacks->Load(nullptr, nullptr)) {
        std::fprintf(stderr, "plugin activated in non-CSGO probe host\n");
        return 15;
    }
    if (access(output, F_OK) == 0) {
        std::fprintf(stderr, "refused plugin created a capture artifact\n");
        return 16;
    }
    dlclose(handle);
    std::puts("probe_host_refusal=ok");
    return 0;
}
