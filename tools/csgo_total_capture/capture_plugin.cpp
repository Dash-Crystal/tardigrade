// Offline-only, fail-closed Source 1 server-plugin capture boundary.
//
// This intentionally uses only the public IServerPluginCallbacks004 ABI. It
// does not patch, detour, scan, or alter game code. Unavailable client/render/
// physics faculties are emitted as hard-refused capabilities.
#include <mach-o/dyld.h>
#include <mach/mach_time.h>
#include <crt_externs.h>
#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>
#include <fcntl.h>
#include <limits.h>
#include <sys/stat.h>
#include <signal.h>

#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

struct edict_t;
class CCommand;
using CreateInterfaceFn = void *(*)(const char *, int *);

enum PLUGIN_RESULT {
    PLUGIN_CONTINUE = 0,
    PLUGIN_OVERRIDE,
    PLUGIN_STOP,
};

enum EQueryCvarValueStatus {
    eQueryCvarValueStatus_ValueIntact = 0,
    eQueryCvarValueStatus_CvarNotFound,
    eQueryCvarValueStatus_NotACvar,
    eQueryCvarValueStatus_CvarProtected,
};
using QueryCvarCookie_t = int;

class IServerPluginCallbacks004 {
public:
    virtual bool Load(CreateInterfaceFn interface_factory,
                      CreateInterfaceFn game_server_factory) = 0;
    virtual void Unload() = 0;
    virtual void Pause() = 0;
    virtual void UnPause() = 0;
    virtual const char *GetPluginDescription() = 0;
    virtual void LevelInit(const char *map_name) = 0;
    virtual void ServerActivate(edict_t *edict_list, int edict_count,
                                int client_max) = 0;
    virtual void GameFrame(bool simulating) = 0;
    virtual void LevelShutdown() = 0;
    virtual void ClientActive(edict_t *entity) = 0;
    virtual void ClientFullyConnect(edict_t *entity) = 0;
    virtual void ClientDisconnect(edict_t *entity) = 0;
    virtual void ClientPutInServer(edict_t *entity, const char *player_name) = 0;
    virtual void SetCommandClient(int index) = 0;
    virtual void ClientSettingsChanged(edict_t *entity) = 0;
    virtual PLUGIN_RESULT ClientConnect(bool *allow_connect, edict_t *entity,
                                        const char *name, const char *address,
                                        char *reject, int reject_size) = 0;
    virtual PLUGIN_RESULT ClientCommand(edict_t *entity,
                                        const CCommand &args) = 0;
    virtual PLUGIN_RESULT NetworkIDValidated(const char *user_name,
                                             const char *network_id) = 0;
    virtual void OnQueryCvarValueFinished(QueryCvarCookie_t cookie,
                                          edict_t *player,
                                          EQueryCvarValueStatus status,
                                          const char *cvar_name,
                                          const char *cvar_value) = 0;
    virtual void OnEdictAllocated(edict_t *edict) = 0;
    virtual void OnEdictFreed(const edict_t *edict) = 0;
    virtual bool BNetworkCryptKeyCheckRequired(
        uint32_t from_ip, uint16_t from_port, uint32_t account_id,
        bool client_wants_crypt_key) = 0;
    virtual bool BNetworkCryptKeyValidate(
        uint32_t from_ip, uint16_t from_port, uint32_t account_id,
        int encryption_key_index, int encrypted_byte_count,
        uint8_t *encrypted_buffer, uint8_t *plain_text_key) = 0;
};

namespace {

constexpr const char *kSchema =
    "tardigrade/source1-native-capture-diagnostic/v1";
constexpr const char *kExpectedRoot =
    "/Users/mdot/Library/Application Support/Steam/steamapps/common/"
    "Counter-Strike Global Offensive";
constexpr const char *kRepository =
    "/Users/mdot/dox/runs/tardigrade-v21-source";

struct ExpectedFile {
    const char *label;
    const char *relative_path;
    const char *sha256;
};

constexpr ExpectedFile kExpectedFiles[] = {
    {"executable", "csgo_osx64",
     "6a42c62aca022b8f66a45ef826d6b33b199a54c7de20f50f9cc4aa22726aae3b"},
    {"engine", "bin/osx64/engine.dylib",
     "1b0ac82e0e95b87d9f5c58550757df53a285bfac9a393a3545a936414038f9e6"},
    {"client", "csgo/bin/osx64/client.dylib",
     "3623246285d7411429e04222e53f26e9d31b4616707e0bd447445cef22e03872"},
    {"server", "csgo/bin/osx64/server.dylib",
     "8642ea4a18abae3bc2c2af5929bfb7499b11449b71680a58ec0b9edce0e077c2"},
    {"vphysics", "bin/osx64/vphysics.dylib",
     "0e63a7e033b33d7aebb5aaf9cf5eef745a3800887220596af4efa98d5fde3164"},
    {"studiorender", "bin/osx64/studiorender.dylib",
     "c234269052f8653f4aac67ffea39c2cb2a13ad1850969ee1a6134c0beee803fa"},
};

// Small standalone SHA-256 implementation so the module validates its exact
// host without subprocesses or an extra crypto-library dependency.
class Sha256 {
public:
    Sha256() { reset(); }

    void update(const uint8_t *data, size_t length) {
        for (size_t index = 0; index < length; ++index) {
            block_[block_length_++] = data[index];
            if (block_length_ == 64) {
                transform();
                bit_length_ += 512;
                block_length_ = 0;
            }
        }
    }

    std::array<uint8_t, 32> finish() {
        size_t index = block_length_;
        block_[index++] = 0x80;
        if (index > 56) {
            while (index < 64) block_[index++] = 0;
            transform();
            index = 0;
        }
        while (index < 56) block_[index++] = 0;
        bit_length_ += static_cast<uint64_t>(block_length_) * 8;
        for (int shift = 56; shift >= 0; shift -= 8) {
            block_[index++] = static_cast<uint8_t>(bit_length_ >> shift);
        }
        transform();
        std::array<uint8_t, 32> digest{};
        for (size_t word = 0; word < 8; ++word) {
            digest[word * 4] = static_cast<uint8_t>(state_[word] >> 24);
            digest[word * 4 + 1] = static_cast<uint8_t>(state_[word] >> 16);
            digest[word * 4 + 2] = static_cast<uint8_t>(state_[word] >> 8);
            digest[word * 4 + 3] = static_cast<uint8_t>(state_[word]);
        }
        return digest;
    }

private:
    static uint32_t rotate(uint32_t value, uint32_t amount) {
        return (value >> amount) | (value << (32 - amount));
    }
    void reset() {
        state_ = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                  0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
        block_.fill(0);
        block_length_ = 0;
        bit_length_ = 0;
    }
    void transform() {
        static constexpr uint32_t constants[64] = {
            0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
            0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
            0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
            0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
            0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
            0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
            0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
            0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U};
        uint32_t words[64]{};
        for (size_t index = 0; index < 16; ++index) {
            const size_t base = index * 4;
            words[index] = (static_cast<uint32_t>(block_[base]) << 24) |
                           (static_cast<uint32_t>(block_[base + 1]) << 16) |
                           (static_cast<uint32_t>(block_[base + 2]) << 8) |
                           static_cast<uint32_t>(block_[base + 3]);
        }
        for (size_t index = 16; index < 64; ++index) {
            const uint32_t s0 = rotate(words[index - 15], 7) ^
                                rotate(words[index - 15], 18) ^
                                (words[index - 15] >> 3);
            const uint32_t s1 = rotate(words[index - 2], 17) ^
                                rotate(words[index - 2], 19) ^
                                (words[index - 2] >> 10);
            words[index] = words[index - 16] + s0 + words[index - 7] + s1;
        }
        uint32_t a=state_[0], b=state_[1], c=state_[2], d=state_[3];
        uint32_t e=state_[4], f=state_[5], g=state_[6], h=state_[7];
        for (size_t index = 0; index < 64; ++index) {
            const uint32_t s1 = rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25);
            const uint32_t choice = (e & f) ^ ((~e) & g);
            const uint32_t temp1 = h + s1 + choice + constants[index] + words[index];
            const uint32_t s0 = rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = s0 + majority;
            h=g; g=f; f=e; e=d+temp1; d=c; c=b; b=a; a=temp1+temp2;
        }
        state_[0]+=a; state_[1]+=b; state_[2]+=c; state_[3]+=d;
        state_[4]+=e; state_[5]+=f; state_[6]+=g; state_[7]+=h;
    }
    std::array<uint32_t, 8> state_{};
    std::array<uint8_t, 64> block_{};
    size_t block_length_ = 0;
    uint64_t bit_length_ = 0;
};

std::string hex_digest(const std::array<uint8_t, 32> &digest) {
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (uint8_t value : digest) output << std::setw(2) << static_cast<int>(value);
    return output.str();
}

bool hash_file(const std::string &path, std::string *result) {
    const int descriptor = open(path.c_str(), O_RDONLY);
    if (descriptor < 0) return false;
    Sha256 sha;
    std::array<uint8_t, 1024 * 1024> buffer{};
    bool ok = true;
    while (true) {
        const ssize_t count = read(descriptor, buffer.data(), buffer.size());
        if (count == 0) break;
        if (count < 0) { ok = false; break; }
        sha.update(buffer.data(), static_cast<size_t>(count));
    }
    close(descriptor);
    if (ok) *result = hex_digest(sha.finish());
    return ok;
}

std::string json_string(const char *value) {
    if (value == nullptr) return "null";
    std::ostringstream output;
    output << '"';
    for (const unsigned char character : std::string(value)) {
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20) {
                    output << "\\u" << std::hex << std::setw(4)
                           << std::setfill('0') << static_cast<int>(character)
                           << std::dec;
                } else {
                    output << character;
                }
        }
    }
    output << '"';
    return output.str();
}

std::string pointer_json(const void *pointer) {
    std::ostringstream output;
    output << '"' << std::hex << reinterpret_cast<uintptr_t>(pointer) << '"';
    return output.str();
}

uint64_t monotonic_ns() {
    static mach_timebase_info_data_t info = [] {
        mach_timebase_info_data_t value{};
        mach_timebase_info(&value);
        return value;
    }();
    const __uint128_t scaled = static_cast<__uint128_t>(mach_continuous_time()) * info.numer;
    return static_cast<uint64_t>(scaled / info.denom);
}

bool starts_with(const std::string &value, const std::string &prefix) {
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

bool loopback_address(const char *address) {
    if (address == nullptr) return false;
    const std::string value(address);
    return value == "loopback" || starts_with(value, "127.0.0.1:") ||
           starts_with(value, "[::1]:") || value == "127.0.0.1" || value == "::1";
}

bool synthetic_bot_address(const char *address) {
    if (address == nullptr) return true;
    const std::string value(address);
    return value.empty() || value == "none" || value == "bot" || value == "BOT";
}

class CaptureWriter {
public:
    bool open_new(const std::string &path) {
        std::lock_guard<std::mutex> guard(lock_);
        descriptor_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL |
                             O_NOFOLLOW | O_CLOEXEC, 0600);
        return descriptor_ >= 0;
    }

    bool write_record(const std::string &kind, const std::string &fields) {
        std::lock_guard<std::mutex> guard(lock_);
        if (descriptor_ < 0) { ++writer_failures_; return false; }
        uint64_t thread = 0;
        pthread_threadid_np(nullptr, &thread);
        std::ostringstream line;
        line << "{\"schema\":\"" << kSchema << "\",\"ordinal\":"
             << ordinal_++ << ",\"monotonic_ns\":" << monotonic_ns()
             << ",\"thread_id\":" << thread << ",\"kind\":"
             << json_string(kind.c_str());
        if (!fields.empty()) line << ',' << fields;
        line << "}\n";
        const std::string data = line.str();
        size_t offset = 0;
        while (offset < data.size()) {
            const ssize_t count = ::write(descriptor_, data.data() + offset,
                                          data.size() - offset);
            if (count <= 0) { ++writer_failures_; return false; }
            offset += static_cast<size_t>(count);
        }
        return true;
    }

    void terminal(const char *status, const char *reason) {
        if (terminal_written_.exchange(true)) return;
        std::ostringstream fields;
        fields << "\"status\":" << json_string(status)
               << ",\"reason\":" << json_string(reason)
               << ",\"drop_counters\":{\"writer_failures\":"
               << writer_failures_.load() << ",\"safety_violations\":"
               << safety_violations_.load() << ",\"hook_failures\":"
               << hook_failures_.load() << "},\"record_counts\":{"
               << "\"usercmd\":" << usercmd_records_.load()
               << ",\"render_view\":" << render_view_records_.load()
               << ",\"demo_requests\":" << demo_requests_.load() << '}';
        write_record("terminal", fields.str());
        std::lock_guard<std::mutex> guard(lock_);
        if (descriptor_ >= 0) {
            fsync(descriptor_);
            close(descriptor_);
            descriptor_ = -1;
        }
    }

    void safety_violation() { ++safety_violations_; }
    void hook_failure() { ++hook_failures_; }
    void usercmd_record() { ++usercmd_records_; }
    void render_view_record() { ++render_view_records_; }
    void demo_request() { ++demo_requests_; }
    uint64_t usercmd_records() const { return usercmd_records_.load(); }
    uint64_t render_view_records() const { return render_view_records_.load(); }
    bool active() const { return descriptor_ >= 0 && !terminal_written_.load(); }

private:
    mutable std::mutex lock_;
    int descriptor_ = -1;
    uint64_t ordinal_ = 0;
    std::atomic<uint64_t> writer_failures_{0};
    std::atomic<uint64_t> safety_violations_{0};
    std::atomic<uint64_t> hook_failures_{0};
    std::atomic<uint64_t> usercmd_records_{0};
    std::atomic<uint64_t> render_view_records_{0};
    std::atomic<uint64_t> demo_requests_{0};
    std::atomic<bool> terminal_written_{false};
};

CaptureWriter g_writer;
std::string g_session_id;
std::string g_demo_base;
std::string g_monitor_nonce;
std::string g_process_instance;
pid_t g_monitor_pid = -1;
bool g_control_mode = false;

bool monitor_lease_valid() {
    if (!g_control_mode || g_monitor_pid <= 0 || kill(g_monitor_pid, 0) != 0) return false;
    const char *path = "/Users/mdot/dox/runs/csgo-total-capture-active/lease.v1";
    struct stat metadata{};
    if (lstat(path, &metadata) != 0 || !S_ISREG(metadata.st_mode) ||
        metadata.st_uid != getuid() || (metadata.st_mode & 077) != 0) return false;
    const std::time_t now = std::time(nullptr);
    if (metadata.st_mtime > now + 1 || now - metadata.st_mtime > 5) return false;
    const int descriptor = ::open(path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (descriptor < 0) return false;
    std::array<char, 512> data{};
    const ssize_t count = read(descriptor, data.data(), data.size() - 1);
    close(descriptor);
    return count > 0 && std::string(data.data(), static_cast<size_t>(count)) ==
        g_session_id + "\n" + g_monitor_nonce + "\n";
}

bool load_monitor_control(std::string *output, std::string *game_root,
                          std::string *build) {
    const char *path = "/Users/mdot/dox/runs/csgo-total-capture-active/control.v1";
    struct stat metadata{};
    if (lstat(path, &metadata) != 0 || !S_ISREG(metadata.st_mode) ||
        metadata.st_uid != getuid() || (metadata.st_mode & 077) != 0) return false;
    const int descriptor = ::open(path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (descriptor < 0) return false;
    std::string content;
    std::array<char, 4096> buffer{};
    ssize_t count = 0;
    while ((count = read(descriptor, buffer.data(), buffer.size())) > 0 &&
           content.size() <= 16384) content.append(buffer.data(), static_cast<size_t>(count));
    close(descriptor);
    if (count < 0 || content.size() > 16384) return false;
    std::vector<std::pair<std::string, std::string>> fields;
    std::istringstream lines(content);
    std::string line;
    while (std::getline(lines, line)) {
        const size_t separator = line.find('=');
        if (separator == std::string::npos) return false;
        fields.emplace_back(line.substr(0, separator), line.substr(separator + 1));
    }
    auto value = [&fields](const char *key) -> std::string {
        std::string result;
        for (const auto &field : fields) {
            if (field.first == key) {
                if (!result.empty()) return {};
                result = field.second;
            }
        }
        return result;
    };
    if (value("schema") != "tardigrade-csgo-outer-monitor-v1" ||
        value("consent") != "offline-only-v1" || value("build") != "12426195") return false;
    *game_root = value("game_root");
    *build = value("build");
    g_session_id = value("session");
    g_demo_base = value("demo_base");
    g_monitor_nonce = value("nonce");
    try {
        g_monitor_pid = static_cast<pid_t>(std::stol(value("monitor_pid")));
    } catch (...) { return false; }
    if (g_monitor_pid <= 0 || std::to_string(g_monitor_pid) != value("monitor_pid")) return false;
    const std::string output_dir = value("output_dir");
    if (g_session_id.empty() || g_demo_base.empty() || g_monitor_nonce.empty() ||
        output_dir.empty()) return false;
    for (const unsigned char character : g_demo_base) {
        if (!(std::isalnum(character) || character == '_' || character == '-')) return false;
    }
    g_process_instance = std::to_string(getpid()) + "_" + std::to_string(monotonic_ns());
    *output = output_dir + "/native-i" + g_process_instance + ".jsonl";
    g_control_mode = true;
    return monitor_lease_valid();
}

std::vector<std::string> arguments() {
    std::vector<std::string> result;
    const int count = *_NSGetArgc();
    char **values = *_NSGetArgv();
    for (int index = 0; index < count; ++index) result.emplace_back(values[index]);
    return result;
}

bool validate_gates(std::string *reason, std::string *output_path,
                    std::string *root) {
    const char *consent = getenv("TARDIGRADE_TOTAL_CAPTURE_CONSENT");
    const char *output = getenv("TARDIGRADE_TOTAL_CAPTURE_OUTPUT");
    const char *game_root = getenv("TARDIGRADE_CSGO_ROOT");
    const char *build = getenv("TARDIGRADE_CSGO_BUILD_ID");
    std::string controlled_output;
    std::string controlled_root;
    std::string controlled_build;
    const bool environment_gate = consent != nullptr &&
        std::string(consent) == "offline-only-v1" && build != nullptr &&
        game_root != nullptr && output != nullptr;
    if (!environment_gate &&
        !load_monitor_control(&controlled_output, &controlled_root, &controlled_build)) {
        *reason = "missing exact consent environment or outer-monitor control gate"; return false;
    }
    if (!environment_gate) {
        output = controlled_output.c_str();
        game_root = controlled_root.c_str();
        build = controlled_build.c_str();
    } else {
        const char *session = getenv("TARDIGRADE_TOTAL_CAPTURE_SESSION");
        const char *demo = getenv("TARDIGRADE_DEMO_NAME");
        g_session_id = session == nullptr ? "" : session;
        g_demo_base = demo == nullptr ? "" : demo;
    }
    if (std::string(build) != "12426195") {
        *reason = "build-id gate is absent or wrong"; return false;
    }
    if (std::string(game_root) != kExpectedRoot) {
        *reason = "game root differs from exact allowlist"; return false;
    }
    std::array<char, PATH_MAX> executable_buffer{};
    uint32_t executable_size = static_cast<uint32_t>(executable_buffer.size());
    if (_NSGetExecutablePath(executable_buffer.data(), &executable_size) != 0) {
        *reason = "cannot resolve current executable"; return false;
    }
    std::array<char, PATH_MAX> resolved_executable{};
    if (realpath(executable_buffer.data(), resolved_executable.data()) == nullptr ||
        std::string(resolved_executable.data()) !=
            std::string(kExpectedRoot) + "/csgo_osx64") {
        *reason = "module is not hosted by the exact allowlisted csgo_osx64";
        return false;
    }
    if (output == nullptr || output[0] != '/') {
        *reason = "capture output must be an absolute path"; return false;
    }
    const std::string output_value(output);
    const size_t separator = output_value.find_last_of('/');
    if (separator == std::string::npos || separator == 0 ||
        separator + 1 == output_value.size()) {
        *reason = "capture output must name a file under an existing directory";
        return false;
    }
    const std::string output_parent = output_value.substr(0, separator);
    std::array<char, PATH_MAX> resolved_parent{};
    if (realpath(output_parent.c_str(), resolved_parent.data()) == nullptr) {
        *reason = "capture output parent cannot be resolved"; return false;
    }
    const std::string canonical_parent(resolved_parent.data());
    const std::string artifact_root = "/Users/mdot/dox/runs";
    if ((canonical_parent != artifact_root &&
         !starts_with(canonical_parent, artifact_root + "/")) ||
        canonical_parent == kRepository ||
        starts_with(canonical_parent, std::string(kRepository) + "/")) {
        *reason = "capture output is not in the external artifact root"; return false;
    }
    bool insecure = false;
    bool offline_marker = false;
    for (const std::string &argument : arguments()) {
        if (argument == "-insecure") insecure = true;
        if (argument == "-tardigrade-total-capture-offline") offline_marker = true;
        if (argument == "+connect" || argument == "-connect" ||
            argument == "+playcast" || argument == "+connect_lobby" ||
            starts_with(argument, "+connect=") || starts_with(argument, "-connect=")) {
            *reason = "remote-connect argument present"; return false;
        }
    }
    if (!insecure) { *reason = "-insecure is mandatory"; return false; }
    if (!offline_marker && environment_gate) {
        *reason = "offline-only argv marker is mandatory"; return false;
    }
    *output_path = output_value;
    *root = game_root;
    return true;
}

bool validate_hashes(const std::string &root) {
    bool all = true;
    for (const ExpectedFile &expected : kExpectedFiles) {
        const std::string path = root + "/" + expected.relative_path;
        std::string actual;
        const bool readable = hash_file(path, &actual);
        const bool matches = readable && actual == expected.sha256;
        std::ostringstream fields;
        fields << "\"module\":" << json_string(expected.label)
               << ",\"path\":" << json_string(path.c_str())
               << ",\"expected_sha256\":" << json_string(expected.sha256)
               << ",\"actual_sha256\":" << json_string(readable ? actual.c_str() : nullptr)
               << ",\"matches\":" << (matches ? "true" : "false");
        g_writer.write_record("build_validation", fields.str());
        all = all && matches;
    }
    return all;
}

void interface_probe(CreateInterfaceFn factory, const char *provider,
                     const char *name) {
    int status = -1;
    void *value = factory == nullptr ? nullptr : factory(name, &status);
    std::ostringstream fields;
    fields << "\"provider\":" << json_string(provider)
           << ",\"interface\":" << json_string(name)
           << ",\"available\":" << (value != nullptr ? "true" : "false")
           << ",\"factory_status\":" << status;
    g_writer.write_record("interface_probe", fields.str());
}

void capability(const char *name, const char *status, const char *completeness,
                const char *reason) {
    std::ostringstream fields;
    fields << "\"capability\":" << json_string(name)
           << ",\"status\":" << json_string(status)
           << ",\"completeness\":" << json_string(completeness)
           << ",\"reason\":" << json_string(reason);
    g_writer.write_record("capability", fields.str());
}

uintptr_t image_base(const void *symbol) {
    Dl_info info{};
    if (symbol == nullptr || dladdr(symbol, &info) == 0 || info.dli_fbase == nullptr) {
        return 0;
    }
    return reinterpret_cast<uintptr_t>(info.dli_fbase);
}

bool exact_prefix(const void *function, const uint8_t *expected, size_t count) {
    return function != nullptr && std::memcmp(function, expected, count) == 0;
}

template <typename Value>
Value unaligned_value(const void *base, size_t offset) {
    Value result{};
    std::memcpy(&result, static_cast<const uint8_t *>(base) + offset,
                sizeof(result));
    return result;
}

struct UserCmdPrefix {
    void *vtable;
    int32_t command_number;
    int32_t tick_count;
    float viewangles[3];
    float aimdirection[3];
    float forwardmove;
    float sidemove;
    float upmove;
    int32_t buttons;
    uint8_t impulse;
    uint8_t padding[3];
    int32_t weaponselect;
    int32_t weaponsubtype;
    int32_t random_seed;
    int16_t mousedx;
    int16_t mousedy;
    uint8_t has_been_predicted;
};
static_assert(offsetof(UserCmdPrefix, command_number) == 8,
              "installed CUserCmd command offset changed");
static_assert(offsetof(UserCmdPrefix, viewangles) == 16,
              "installed CUserCmd view-angle offset changed");
static_assert(offsetof(UserCmdPrefix, mousedx) == 72,
              "installed CUserCmd mouse offset changed");

class ClientHooks {
public:
    using ClientCreateMoveFn = void (*)(void *, int, float, bool);
    using ClientRenderViewFn = void (*)(void *, const void *, int, int);
    using InputGetUserCmdFn = void *(*)(void *, int, int);
    using GetMatricesForViewFn = void (*)(void *, const void *, float *, float *,
                                          float *, float *);

    bool install(const std::string &root, void *render_view,
                 uintptr_t engine_base, std::string *reason) {
        constexpr uintptr_t kClientCreateMoveOffset = 0x140300;
        constexpr uintptr_t kClientRenderViewOffset = 0x143e50;
        constexpr uintptr_t kClientInputGlobalOffset = 0x1616270;
        constexpr uintptr_t kInputGetUserCmdOffset = 0x204c60;
        constexpr uintptr_t kGetMatricesOffset = 0x494e0;
        static constexpr uint8_t kCreateMovePrefix[] = {
            0x55,0x48,0x89,0xe5,0x41,0x57,0x41,0x56,
            0x53,0x48,0x83,0xec,0x18,0x41,0x89,0xd7};
        static constexpr uint8_t kRenderViewPrefix[] = {
            0x55,0x48,0x89,0xe5,0x41,0x57,0x41,0x56,
            0x41,0x55,0x41,0x54,0x53,0x50,0x41,0x89};
        static constexpr uint8_t kGetUserCmdPrefix[] = {
            0x55,0x48,0x89,0xe5,0x48,0x63,0xc2,0x48,
            0x69,0xc8,0xb5,0x81,0x4e,0x1b,0x49,0x89};
        static constexpr uint8_t kMatricesPrefix[] = {
            0x55,0x48,0x89,0xe5,0x41,0x57,0x41,0x56,
            0x41,0x55,0x41,0x54,0x53,0x50,0x4d,0x89};

        const std::string client_path = root + "/csgo/bin/osx64/client.dylib";
        client_handle_ = dlopen(client_path.c_str(), RTLD_NOW | RTLD_NOLOAD);
        if (client_handle_ == nullptr) {
            *reason = "allowlisted client module is not loaded";
            return false;
        }
        auto factory = reinterpret_cast<CreateInterfaceFn>(
            dlsym(client_handle_, "CreateInterface"));
        client_base_ = image_base(reinterpret_cast<void *>(factory));
        int status = -1;
        client_ = factory == nullptr ? nullptr : factory("VClient018", &status);
        if (client_base_ == 0 || client_ == nullptr || status != 0) {
            *reason = "VClient018 factory resolution failed";
            return false;
        }
        original_vtable_ = *reinterpret_cast<void ***>(client_);
        if (original_vtable_ == nullptr ||
            original_vtable_[24] != reinterpret_cast<void *>(
                client_base_ + kClientCreateMoveOffset) ||
            original_vtable_[28] != reinterpret_cast<void *>(
                client_base_ + kClientRenderViewOffset) ||
            !exact_prefix(original_vtable_[24], kCreateMovePrefix,
                          sizeof(kCreateMovePrefix)) ||
            !exact_prefix(original_vtable_[28], kRenderViewPrefix,
                          sizeof(kRenderViewPrefix))) {
            *reason = "installed VClient018 vtable/signature allowlist mismatch";
            return false;
        }
        input_ = *reinterpret_cast<void **>(client_base_ + kClientInputGlobalOffset);
        if (input_ == nullptr) {
            *reason = "validated CInput global is null";
            return false;
        }
        void **input_vtable = *reinterpret_cast<void ***>(input_);
        if (input_vtable == nullptr ||
            input_vtable[8] != reinterpret_cast<void *>(
                client_base_ + kInputGetUserCmdOffset) ||
            !exact_prefix(input_vtable[8], kGetUserCmdPrefix,
                          sizeof(kGetUserCmdPrefix))) {
            *reason = "installed CInput::GetUserCmd vtable/signature allowlist mismatch";
            return false;
        }
        if (render_view == nullptr || engine_base == 0) {
            *reason = "VEngineRenderView014 is unavailable";
            return false;
        }
        void **render_vtable = *reinterpret_cast<void ***>(render_view);
        if (render_vtable == nullptr ||
            render_vtable[56] != reinterpret_cast<void *>(
                engine_base + kGetMatricesOffset) ||
            !exact_prefix(render_vtable[56], kMatricesPrefix,
                          sizeof(kMatricesPrefix))) {
            *reason = "installed GetMatricesForView vtable/signature allowlist mismatch";
            return false;
        }
        render_view_ = render_view;
        get_usercmd_ = reinterpret_cast<InputGetUserCmdFn>(input_vtable[8]);
        get_matrices_ = reinterpret_cast<GetMatricesForViewFn>(render_vtable[56]);
        original_create_move_ = reinterpret_cast<ClientCreateMoveFn>(
            original_vtable_[24]);
        original_render_view_ = reinterpret_cast<ClientRenderViewFn>(
            original_vtable_[28]);
        // Preserve the Itanium ABI offset-to-top and RTTI words immediately
        // before the address point.  Pointing an object at a bare function
        // array makes dynamic_cast/typeid read before the allocation.
        cloned_vtable_storage_[0] = original_vtable_[-2];
        cloned_vtable_storage_[1] = original_vtable_[-1];
        for (size_t index = 0; index < kClonedFunctionCount; ++index) {
            cloned_vtable_storage_[index + 2] = original_vtable_[index];
        }
        cloned_vtable_storage_[24 + 2] = reinterpret_cast<void *>(&create_move_hook);
        cloned_vtable_storage_[28 + 2] = reinterpret_cast<void *>(&render_view_hook);
        active_ = this;
        __atomic_store_n(reinterpret_cast<void ***>(client_),
                         cloned_address_point(), __ATOMIC_SEQ_CST);
        installed_ = true;
        return true;
    }

    void uninstall() {
        if (!installed_) {
            if (client_handle_ != nullptr) dlclose(client_handle_);
            client_handle_ = nullptr;
            return;
        }
        void **current = __atomic_load_n(reinterpret_cast<void ***>(client_),
                                         __ATOMIC_SEQ_CST);
        if (current == cloned_address_point()) {
            __atomic_store_n(reinterpret_cast<void ***>(client_), original_vtable_,
                             __ATOMIC_SEQ_CST);
        } else {
            g_writer.hook_failure();
            g_writer.write_record(
                "hook_teardown_refusal",
                "\"reason\":\"VClient018 vtable changed after capture install\"");
        }
        installed_ = false;
        active_ = nullptr;
        if (client_handle_ != nullptr) dlclose(client_handle_);
        client_handle_ = nullptr;
    }

    bool installed() const { return installed_; }

private:
    static void create_move_hook(void *self, int sequence_number,
                                 float sample_seconds, bool active) {
        ClientHooks *hooks = active_;
        if (hooks == nullptr || hooks->original_create_move_ == nullptr) return;
        hooks->original_create_move_(self, sequence_number, sample_seconds, active);
        void *raw = hooks->get_usercmd_(hooks->input_, 0, sequence_number);
        if (raw == nullptr) {
            g_writer.hook_failure();
            return;
        }
        const auto *command = static_cast<const UserCmdPrefix *>(raw);
        if (command->command_number != sequence_number ||
            !std::isfinite(sample_seconds) ||
            !std::isfinite(command->viewangles[0]) ||
            !std::isfinite(command->viewangles[1]) ||
            !std::isfinite(command->viewangles[2]) ||
            !std::isfinite(command->aimdirection[0]) ||
            !std::isfinite(command->aimdirection[1]) ||
            !std::isfinite(command->aimdirection[2]) ||
            !std::isfinite(command->forwardmove) ||
            !std::isfinite(command->sidemove) ||
            !std::isfinite(command->upmove)) {
            g_writer.hook_failure();
            return;
        }
        std::ostringstream fields;
        fields << std::setprecision(9)
               << "\"source_sequence\":" << sequence_number
               << ",\"engine_tick\":" << command->tick_count
               << ",\"input_sample_seconds\":" << sample_seconds
               << ",\"active\":" << (active ? "true" : "false")
               << ",\"viewangles_target\":[" << command->viewangles[0] << ','
               << command->viewangles[1] << ',' << command->viewangles[2] << ']'
               << ",\"aim_direction\":[" << command->aimdirection[0] << ','
               << command->aimdirection[1] << ',' << command->aimdirection[2] << ']'
               << ",\"move\":[" << command->forwardmove << ','
               << command->sidemove << ',' << command->upmove << ']'
               << ",\"buttons\":" << command->buttons
               << ",\"impulse\":" << static_cast<unsigned>(command->impulse)
               << ",\"weapon_select\":" << command->weaponselect
               << ",\"weapon_subtype\":" << command->weaponsubtype
               << ",\"random_seed\":" << command->random_seed
               << ",\"mouse_interval_counts\":[" << command->mousedx << ','
               << command->mousedy << ']'
               << ",\"mouse_semantics\":\"interval-integrated-counts-not-instantaneous-velocity\""
               << ",\"predicted\":"
               << (command->has_been_predicted ? "true" : "false");
        if (g_writer.write_record("client_usercmd", fields.str())) {
            g_writer.usercmd_record();
        }
    }

    static void render_view_hook(void *self, const void *view, int clear_flags,
                                 int what_to_draw) {
        ClientHooks *hooks = active_;
        if (hooks == nullptr || hooks->original_render_view_ == nullptr) return;
        if (view != nullptr) hooks->record_render_view(view, clear_flags, what_to_draw);
        else g_writer.hook_failure();
        hooks->original_render_view_(self, view, clear_flags, what_to_draw);
    }

    void record_render_view(const void *view, int clear_flags, int what_to_draw) {
        const float fov = unaligned_value<float>(view, 0xb8);
        const float aspect = unaligned_value<float>(view, 0xe8);
        const float near_plane = unaligned_value<float>(view, 0xd8);
        const float far_plane = unaligned_value<float>(view, 0xdc);
        const int width = unaligned_value<int32_t>(view, 0x10);
        const int height = unaligned_value<int32_t>(view, 0x18);
        float origin[3]{};
        float angles[3]{};
        std::memcpy(origin, static_cast<const uint8_t *>(view) + 0xc0,
                    sizeof(origin));
        std::memcpy(angles, static_cast<const uint8_t *>(view) + 0xcc,
                    sizeof(angles));
        if (width <= 0 || height <= 0 || !std::isfinite(fov) || fov <= 0.0f ||
            !std::isfinite(aspect) || !std::isfinite(near_plane) ||
            !std::isfinite(far_plane) ||
            !std::isfinite(origin[0]) || !std::isfinite(origin[1]) ||
            !std::isfinite(origin[2]) || !std::isfinite(angles[0]) ||
            !std::isfinite(angles[1]) || !std::isfinite(angles[2])) {
            g_writer.hook_failure();
            return;
        }
        float world_to_view[16]{};
        float view_to_projection[16]{};
        float world_to_projection[16]{};
        float world_to_pixels[16]{};
        get_matrices_(render_view_, view, world_to_view, view_to_projection,
                      world_to_projection, world_to_pixels);
        for (const float *matrix : {world_to_view, view_to_projection,
                                    world_to_projection, world_to_pixels}) {
            for (int index = 0; index < 16; ++index) {
                if (!std::isfinite(matrix[index])) {
                    g_writer.hook_failure();
                    return;
                }
            }
        }
        auto matrix_json = [](const float *matrix) {
            std::ostringstream output;
            output << '[' << std::setprecision(9);
            for (int index = 0; index < 16; ++index) {
                if (index != 0) output << ',';
                output << matrix[index];
            }
            output << ']';
            return output.str();
        };
        std::ostringstream fields;
        fields << std::setprecision(9)
               << "\"render_sequence\":" << render_sequence_++
               << ",\"viewport\":[" << width << ',' << height << ']'
               << ",\"origin\":[" << origin[0] << ',' << origin[1] << ','
               << origin[2] << ']'
               << ",\"angles\":[" << angles[0] << ',' << angles[1] << ','
               << angles[2] << ']'
               << ",\"fov\":" << fov << ",\"aspect_ratio\":" << aspect
               << ",\"near\":" << near_plane << ",\"far\":" << far_plane
               << ",\"clear_flags\":" << clear_flags
               << ",\"what_to_draw\":" << what_to_draw
               << ",\"world_to_view\":" << matrix_json(world_to_view)
               << ",\"view_to_projection\":" << matrix_json(view_to_projection)
               << ",\"world_to_projection\":" << matrix_json(world_to_projection)
               << ",\"world_to_pixels\":" << matrix_json(world_to_pixels);
        if (g_writer.write_record("client_render_view", fields.str())) {
            g_writer.render_view_record();
        }
    }

    static ClientHooks *active_;
    void *client_handle_ = nullptr;
    uintptr_t client_base_ = 0;
    void *client_ = nullptr;
    void *input_ = nullptr;
    void *render_view_ = nullptr;
    void **original_vtable_ = nullptr;
    static constexpr size_t kClonedFunctionCount = 96;
    void **cloned_address_point() { return cloned_vtable_storage_.data() + 2; }
    std::array<void *, kClonedFunctionCount + 2> cloned_vtable_storage_{};
    ClientCreateMoveFn original_create_move_ = nullptr;
    ClientRenderViewFn original_render_view_ = nullptr;
    InputGetUserCmdFn get_usercmd_ = nullptr;
    GetMatricesForViewFn get_matrices_ = nullptr;
    std::atomic<uint64_t> render_sequence_{0};
    bool installed_ = false;
};

ClientHooks *ClientHooks::active_ = nullptr;
ClientHooks g_client_hooks;

class TotalCapturePlugin final : public IServerPluginCallbacks004 {
public:
    bool Load(CreateInterfaceFn interface_factory,
              CreateInterfaceFn game_server_factory) override {
        std::string reason;
        std::string output;
        std::string root;
        if (!validate_gates(&reason, &output, &root)) return false;
        if (g_process_instance.empty()) {
            g_process_instance = std::to_string(getpid()) + "_" + std::to_string(monotonic_ns());
        }
        if (!g_writer.open_new(output)) return false;
        g_writer.write_record(
            "session_start",
            "\"session_id\":" + json_string(g_session_id.c_str()) +
            ",\"process_instance_id\":" + json_string(g_process_instance.c_str()) +
            ",\"process_id\":" + std::to_string(getpid()) +
            ",\"evidence\":\"native-server-plugin-diagnostic\","
            "\"capture_claim\":\"boundary-diagnostic-not-total-capture\","
            "\"offline_only\":true,"
            "\"insecure\":true,\"build_id\":\"12426195\"");
        if (!validate_hashes(root)) {
            g_writer.terminal("refused", "exact binary/module SHA allowlist mismatch");
            return false;
        }
        interface_probe(interface_factory, "engine", "VEngineServer023");
        interface_probe(interface_factory, "engine", "VEngineClient014");
        interface_probe(interface_factory, "engine", "VEngineRenderView014");
        interface_probe(interface_factory, "engine", "GAMEEVENTSMANAGER002");
        interface_probe(game_server_factory, "server", "ServerGameDLL005");
        interface_probe(game_server_factory, "server", "ServerGameClients004");
        interface_probe(game_server_factory, "server", "ServerGameEnts001");
        interface_probe(game_server_factory, "server", "PlayerInfoManager002");
        interface_probe(game_server_factory, "server", "BotManager001");
        interface_probe(game_server_factory, "server", "VSERVERTOOLS001");
        int render_status = -1;
        void *render_view = interface_factory == nullptr ? nullptr :
            interface_factory("VEngineRenderView014", &render_status);
        const uintptr_t engine_base = image_base(
            reinterpret_cast<void *>(interface_factory));
        std::string hook_reason;
        const bool client_hooks = g_client_hooks.install(
            root, render_view, engine_base, &hook_reason);
        configure_demo_coordinator(interface_factory, engine_base);
        capability("server_plugin_callbacks", "available",
                   "complete-for-callback-surface",
                   "public IServerPluginCallbacks004 loaded after exact safety/build validation");
        capability("server_frame_boundary", "available",
                   "boundary-only",
                   "public GameFrame callback; records ordering boundary but not private engine state");
        capability("entity_lifecycle", "available",
                   "lifecycle-only-no-state",
                   "public ServerActivate/OnEdictAllocated/OnEdictFreed/client lifecycle callbacks");
        capability("client_console_command_boundary", "available",
                   "metadata-only",
                   "public ClientCommand callback; CCommand payload remains opaque in the minimal ABI");
        capability("native_demo_coordination",
                   demo_command_ == nullptr ? "unavailable-hard-refused" : "available",
                   demo_command_ == nullptr ? "none" : "boundary-only",
                   demo_command_ == nullptr
                       ? "installed VEngineClient014 ClientCmd_Unrestricted ABI validation failed"
                       : "in-process record/stop requests joined to client-ready and level-shutdown boundaries; monitor must verify HL2DEMO bytes");
        capability("ego_create_move_usercmd_viewangles",
                   client_hooks ? "available" : "unavailable-hard-refused",
                   client_hooks ? "authoritative-ego-command" : "none",
                   client_hooks
                       ? "exact-build cloned VClient018 CreateMove observer reads validated CInput CUserCmd after engine creation"
                       : hook_reason.c_str());
        capability("observed_render_view_matrices",
                   client_hooks ? "available" : "unavailable-hard-refused",
                   client_hooks ? "observed-client-render-calls" : "none",
                   client_hooks
                       ? "exact-build VClient018 RenderView observer records CViewSetup and four VEngineRenderView matrices"
                       : hook_reason.c_str());
        capability("create_move_human_usercmd",
                   client_hooks ? "available" : "unavailable-hard-refused",
                   client_hooks ? "authoritative-ego-command" : "none",
                   client_hooks
                       ? "authoritative ego CUserCmd target/viewangles and interval mouse counts at CreateMove completion"
                       : hook_reason.c_str());
        capability("bot_usercmd_generation", "unavailable-hard-refused",
                   "none",
                   "BotManager001 availability does not expose generated CUserCmd order");
        capability("server_usercmd_apply_order", "unavailable-hard-refused",
                   "none",
                   "ServerGameClients004 would require an unvalidated vtable detour of ProcessUsercmds");
        capability("democmdinfo_all_pov_view_fov", "unavailable-hard-refused",
                   "none",
                   "ego client RenderView is observed, but no validated route supplies every scored POV or stable POV identity");
        capability("setup_bones_final_matrices_attachments", "unavailable-hard-refused",
                   "none",
                   "SetupBones/attachment evaluation is not exported as a passive callback");
        capability("vphysics_active_body_state", "unavailable-hard-refused",
                   "none",
                   "VPhysics031 does not passively enumerate the active game's private body environments");
        capability("effects_history", "unavailable-hard-refused",
                   "none",
                   "IEffects exposes commands, not a complete passive history callback");
        capability("panorama_ui_history", "unavailable-hard-refused",
                   "none",
                   "Panorama interfaces expose mutable UI services, not a safe total event/state callback");
        capability("render_history", "unavailable-hard-refused",
                   "none",
                   "VEngineRenderView/VStudioRender expose rendering services but no passive complete history callback");
        loaded_ = true;
        return true;
    }

    void Unload() override {
        request_demo_stop("plugin_unload");
        g_client_hooks.uninstall();
        if (loaded_) g_writer.terminal("clean", "diagnostic plugin unloaded cleanly");
        loaded_ = false;
    }
    void Pause() override { event("pause", ""); }
    void UnPause() override { event("unpause", ""); }
    const char *GetPluginDescription() override {
        return "Tardigrade offline-only Source1 total-capture boundary v1";
    }
    void LevelInit(const char *map_name) override {
        request_demo_stop("level_init");
        current_map_ = map_name == nullptr ? "unknown" : map_name;
        ++map_epoch_;
        event("level_init", "\"map\":" + json_string(map_name));
    }
    void ServerActivate(edict_t *edict_list, int edict_count,
                        int client_max) override {
        std::ostringstream fields;
        fields << "\"edict_list\":" << pointer_json(edict_list)
               << ",\"edict_count\":" << edict_count
               << ",\"client_max\":" << client_max;
        event("server_activate", fields.str());
    }
    void GameFrame(bool simulating) override {
        if (capture_closed_) return;
        std::ostringstream fields;
        fields << "\"simulating\":" << (simulating ? "true" : "false")
               << ",\"frame_ordinal\":" << frame_ordinal_++;
        event("game_frame", fields.str());
        if (simulating && demo_active_ &&
            monotonic_ns() - segment_started_ns_ >= 600ULL * 1000ULL * 1000ULL * 1000ULL) {
            request_demo_stop("periodic_ten_minute_checkpoint");
            request_demo_start("periodic_ten_minute_checkpoint");
        }
        if (frame_ordinal_ % 256 == 0) {
            std::ostringstream health;
            health << "\"frame_ordinal\":" << frame_ordinal_
                   << ",\"usercmd_records\":" << g_writer.usercmd_records()
                   << ",\"render_view_records\":"
                   << g_writer.render_view_records()
                   << ",\"demo_active\":"
                   << (demo_active_ ? "true" : "false")
                   << ",\"map_epoch\":" << map_epoch_
                   << ",\"segment_epoch\":" << segment_epoch_;
            event("health", health.str());
        }
        if (frame_ordinal_ % 64 == 0 && outer_close_requested()) {
            request_demo_stop("outer_session_close");
            g_client_hooks.uninstall();
            event("outer_session_close_ack",
                  "\"session_id\":" + json_string(g_session_id.c_str()));
            g_writer.terminal("clean", "outer monitor requested capture close");
            capture_closed_ = true;
        } else if (frame_ordinal_ % 64 == 0 && g_control_mode &&
                   !monitor_lease_valid()) {
            request_demo_stop("outer_monitor_lease_expired");
            g_client_hooks.uninstall();
            event("outer_monitor_lease_expired",
                  "\"session_id\":" + json_string(g_session_id.c_str()));
            g_writer.terminal("incomplete", "outer monitor lease expired");
            capture_closed_ = true;
        }
    }
    void LevelShutdown() override {
        request_demo_stop("level_shutdown");
        event("level_shutdown", "");
    }
    void ClientActive(edict_t *entity) override { entity_event("client_active", entity); }
    void ClientFullyConnect(edict_t *entity) override {
        entity_event("client_fully_connect", entity);
        if (entity == local_client_) {
            request_demo_stop("client_ready_rotation");
            request_demo_start("client_fully_connect");
        }
    }
    void ClientDisconnect(edict_t *entity) override {
        if (entity == local_client_) {
            request_demo_stop("local_client_disconnect");
            local_client_ = nullptr;
        }
        entity_event("client_disconnect", entity);
    }
    void ClientPutInServer(edict_t *entity, const char *player_name) override {
        std::ostringstream fields;
        fields << "\"edict\":" << pointer_json(entity)
               << ",\"player_name\":" << json_string(player_name);
        event("client_put_in_server", fields.str());
    }
    void SetCommandClient(int index) override {
        command_client_ = index;
        event("set_command_client", "\"client_index\":" + std::to_string(index));
    }
    void ClientSettingsChanged(edict_t *entity) override {
        entity_event("client_settings_changed", entity);
    }
    PLUGIN_RESULT ClientConnect(bool *allow_connect, edict_t *entity,
                                const char *name, const char *address,
                                char *reject, int reject_size) override {
        const bool local = loopback_address(address);
        const bool synthetic_bot = synthetic_bot_address(address);
        std::ostringstream fields;
        fields << "\"edict\":" << pointer_json(entity)
               << ",\"player_name\":" << json_string(name)
               << ",\"address\":" << json_string(address)
               << ",\"loopback\":" << (local ? "true" : "false")
               << ",\"synthetic_bot\":" << (synthetic_bot ? "true" : "false");
        event("client_connect", fields.str());
        if (!local && !synthetic_bot) {
            g_writer.safety_violation();
            if (allow_connect != nullptr) *allow_connect = false;
            if (reject != nullptr && reject_size > 0) {
                snprintf(reject, static_cast<size_t>(reject_size),
                         "offline-only total capture refuses remote clients");
            }
            event("safety_refusal", "\"reason\":\"non-loopback client\"");
            return PLUGIN_STOP;
        }
        if (local) local_client_ = entity;
        return PLUGIN_CONTINUE;
    }
    PLUGIN_RESULT ClientCommand(edict_t *entity,
                                const CCommand &args) override {
        (void)args;
        std::ostringstream fields;
        fields << "\"edict\":" << pointer_json(entity)
               << ",\"command_client\":" << command_client_
               << ",\"payload_status\":\"opaque-minimal-public-abi\"";
        event("client_command", fields.str());
        return PLUGIN_CONTINUE;
    }
    PLUGIN_RESULT NetworkIDValidated(const char *user_name,
                                     const char *network_id) override {
        std::ostringstream fields;
        fields << "\"user_name\":" << json_string(user_name)
               << ",\"network_id\":" << json_string(network_id);
        event("network_id_validated", fields.str());
        return PLUGIN_CONTINUE;
    }
    void OnQueryCvarValueFinished(QueryCvarCookie_t cookie, edict_t *player,
                                  EQueryCvarValueStatus status,
                                  const char *cvar_name,
                                  const char *cvar_value) override {
        std::ostringstream fields;
        fields << "\"cookie\":" << cookie << ",\"edict\":"
               << pointer_json(player) << ",\"status\":" << status
               << ",\"cvar_name\":" << json_string(cvar_name)
               << ",\"cvar_value\":" << json_string(cvar_value);
        event("query_cvar_finished", fields.str());
    }
    void OnEdictAllocated(edict_t *edict) override { entity_event("edict_allocated", edict); }
    void OnEdictFreed(const edict_t *edict) override { entity_event("edict_freed", edict); }
    bool BNetworkCryptKeyCheckRequired(uint32_t from_ip, uint16_t from_port,
                                       uint32_t account_id,
                                       bool client_wants_crypt_key) override {
        std::ostringstream fields;
        fields << "\"reason\":\"network crypt callback forbidden in offline-only capture\""
               << ",\"from_ip_raw\":" << from_ip
               << ",\"from_port\":" << from_port
               << ",\"account_id\":" << account_id
               << ",\"client_wants_crypt_key\":"
               << (client_wants_crypt_key ? "true" : "false");
        g_writer.safety_violation();
        event("safety_refusal", fields.str());
        // Force the engine down the validation path; validation below always
        // rejects and never supplies key material.
        return true;
    }
    bool BNetworkCryptKeyValidate(uint32_t from_ip, uint16_t from_port,
                                  uint32_t account_id,
                                  int encryption_key_index,
                                  int encrypted_byte_count,
                                  uint8_t *encrypted_buffer,
                                  uint8_t *plain_text_key) override {
        (void)encrypted_buffer;
        (void)plain_text_key;
        std::ostringstream fields;
        fields << "\"reason\":\"network crypt validation forbidden in offline-only capture\""
               << ",\"from_ip_raw\":" << from_ip
               << ",\"from_port\":" << from_port
               << ",\"account_id\":" << account_id
               << ",\"encryption_key_index\":" << encryption_key_index
               << ",\"encrypted_byte_count\":" << encrypted_byte_count;
        g_writer.safety_violation();
        event("safety_refusal", fields.str());
        return false;
    }

private:
    using ClientCommandFn = void (*)(void *, const char *);

    void configure_demo_coordinator(CreateInterfaceFn factory,
                                    uintptr_t engine_base) {
        constexpr uintptr_t kClientCmdUnrestrictedOffset = 0x7f080;
        static constexpr uint8_t kClientCmdPrefix[] = {
            0x55,0x48,0x89,0xe5,0x53,0x50,0x48,0x89,
            0xf3,0xe8,0xd2,0x31,0x17,0x00,0x31,0xd2};
        const std::string &name = g_demo_base;
        if (name.empty()) return;
        for (const unsigned char character : name) {
            if (!(std::isalnum(character) || character == '_' || character == '-')) {
                return;
            }
        }
        int status = -1;
        void *engine_client = factory == nullptr ? nullptr :
            factory("VEngineClient014", &status);
        if (engine_client == nullptr || status != 0 || engine_base == 0) return;
        void **vtable = *reinterpret_cast<void ***>(engine_client);
        if (vtable == nullptr ||
            vtable[108] != reinterpret_cast<void *>(
                engine_base + kClientCmdUnrestrictedOffset) ||
            !exact_prefix(vtable[108], kClientCmdPrefix,
                          sizeof(kClientCmdPrefix))) return;
        engine_client_ = engine_client;
        demo_command_ = reinterpret_cast<ClientCommandFn>(vtable[108]);
        demo_name_ = name;
    }

    void request_demo_start(const char *boundary) {
        if (demo_active_ || demo_command_ == nullptr || demo_name_.empty()) return;
        ++segment_epoch_;
        std::ostringstream segment;
        segment << demo_name_ << "_i" << g_process_instance << "_e" << segment_epoch_;
        active_segment_name_ = segment.str();
        const std::string command = "record " + active_segment_name_;
        demo_command_(engine_client_, command.c_str());
        demo_active_ = true;
        segment_started_ns_ = monotonic_ns();
        g_writer.demo_request();
        std::ostringstream fields;
        fields << "\"operation\":\"record-request\",\"name\":"
               << json_string(active_segment_name_.c_str())
               << ",\"boundary\":" << json_string(boundary)
               << ",\"process_id\":" << getpid()
               << ",\"process_instance_id\":" << json_string(g_process_instance.c_str())
               << ",\"map_epoch\":" << map_epoch_
               << ",\"segment_epoch\":" << segment_epoch_
               << ",\"map\":" << json_string(current_map_.c_str())
               << ",\"verification\":\"outer-monitor-must-confirm-HL2DEMO\"";
        event("demo_lifecycle", fields.str());
    }

    void request_demo_stop(const char *boundary) {
        if (!demo_active_ || demo_command_ == nullptr) return;
        demo_command_(engine_client_, "stop");
        demo_active_ = false;
        g_writer.demo_request();
        std::ostringstream fields;
        fields << "\"operation\":\"stop-request\",\"boundary\":"
               << json_string(boundary)
               << ",\"name\":" << json_string(active_segment_name_.c_str())
               << ",\"process_id\":" << getpid()
               << ",\"process_instance_id\":" << json_string(g_process_instance.c_str())
               << ",\"map_epoch\":" << map_epoch_
               << ",\"segment_epoch\":" << segment_epoch_
               << ",\"map\":" << json_string(current_map_.c_str());
        event("demo_lifecycle", fields.str());
        active_segment_name_.clear();
    }

    void event(const std::string &kind, const std::string &fields) {
        if (!capture_closed_ && (loaded_ || kind == "level_init"))
            g_writer.write_record(kind, fields);
    }
    bool outer_close_requested() const {
        const char *path = "/Users/mdot/dox/runs/csgo-total-capture-active/close.request";
        struct stat metadata{};
        if (lstat(path, &metadata) != 0 || !S_ISREG(metadata.st_mode) ||
            metadata.st_uid != getuid() || (metadata.st_mode & 077) != 0) return false;
        const int descriptor = ::open(path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
        if (descriptor < 0) return false;
        std::array<char, 256> value{};
        const ssize_t count = read(descriptor, value.data(), value.size() - 1);
        close(descriptor);
        return count > 0 && std::string(value.data(), static_cast<size_t>(count)) ==
            g_session_id + "\n";
    }
    void entity_event(const char *kind, const void *entity) {
        event(kind, "\"edict\":" + pointer_json(entity));
    }
    bool loaded_ = false;
    uint64_t frame_ordinal_ = 0;
    int command_client_ = -1;
    void *engine_client_ = nullptr;
    ClientCommandFn demo_command_ = nullptr;
    std::string demo_name_;
    std::string active_segment_name_;
    std::string current_map_ = "unknown";
    edict_t *local_client_ = nullptr;
    uint64_t map_epoch_ = 0;
    uint64_t segment_epoch_ = 0;
    uint64_t segment_started_ns_ = 0;
    bool demo_active_ = false;
    bool capture_closed_ = false;
};

TotalCapturePlugin g_plugin;

}  // namespace

extern "C" __attribute__((visibility("default"))) void *
CreateInterface(const char *name, int *return_code) {
    if (name != nullptr && std::strcmp(name, "ISERVERPLUGINCALLBACKS004") == 0) {
        if (return_code != nullptr) *return_code = 0;
        return &g_plugin;
    }
    if (return_code != nullptr) *return_code = 1;
    return nullptr;
}

__attribute__((destructor)) static void total_capture_destructor() {
    if (g_writer.active()) {
        g_writer.terminal("incomplete", "process exited without plugin Unload callback");
    }
}
