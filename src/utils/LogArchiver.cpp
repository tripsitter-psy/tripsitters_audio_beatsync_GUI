#include "LogArchiver.h"
#include <fstream>
#include <cstdint>
#include <ctime>
#include <vector>
#include <cstring>

namespace BeatSync {

static uint32_t crc32_table[256];
static bool crc_table_initialized = false;

static void init_crc32() {
    if (crc_table_initialized) return;
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (size_t j = 0; j < 8; ++j) {
            if (c & 1) c = 0xEDB88320u ^ (c >> 1);
            else c = c >> 1;
        }
        crc32_table[i] = c;
    }
    crc_table_initialized = true;
}

static uint32_t crc32(const unsigned char* data, size_t len) {
    init_crc32();
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i) {
        c = crc32_table[(c ^ data[i]) & 0xFF] ^ (c >> 8);
    }
    return c ^ 0xFFFFFFFFu;
}

static void write_le(std::ofstream& os, uint16_t v) {
    char b[2];
    b[0] = v & 0xFF;
    b[1] = (v >> 8) & 0xFF;
    os.write(b, 2);
}
static void write_le(std::ofstream& os, uint32_t v) {
    char b[4];
    b[0] = v & 0xFF;
    b[1] = (v >> 8) & 0xFF;
    b[2] = (v >> 16) & 0xFF;
    b[3] = (v >> 24) & 0xFF;
    os.write(b, 4);
}
static void write_le64(std::ofstream& os, uint64_t v) {
    char b[8];
    for (int i = 0; i < 8; ++i) b[i] = (v >> (8*i)) & 0xFF;
    os.write(b, 8);
}

bool createZip(const std::vector<std::string>& files, const std::string& dest, std::string& error) {
    error.clear();

    struct Entry {
        std::string name;
        uint32_t crc;
        uint32_t compSize;
        uint32_t uncompSize;
        uint32_t localHeaderOffset;
        std::vector<char> data;
    };

    std::vector<Entry> entries;

    for (const auto& fpath : files) {
        std::ifstream in(fpath, std::ios::binary);
        if (!in) {
            error = "Could not open file: " + fpath;
            return false;
        }
        std::vector<char> buf((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        Entry e;
        // Use filename only (no path)
        size_t pos = fpath.find_last_of("/\\");
        if (pos != std::string::npos) e.name = fpath.substr(pos+1);
        else e.name = fpath;
        e.uncompSize = static_cast<uint32_t>(buf.size());
        e.compSize = e.uncompSize; // store (no compression)
        e.crc = (e.uncompSize > 0) ? crc32(reinterpret_cast<const unsigned char*>(buf.data()), buf.size()) : 0;
        e.data = std::move(buf);
        entries.push_back(std::move(e));
    }

    std::ofstream out(dest, std::ios::binary);
    if (!out) {
        error = "Could not create destination zip: " + dest;
        return false;
    }

    // Write local file headers and file data
    for (auto& e : entries) {
        e.localHeaderOffset = static_cast<uint32_t>(out.tellp());
        // Local file header signature
        write_le(out, static_cast<uint32_t>(0x04034b50));
        // version needed to extract (2 bytes)
        write_le(out, static_cast<uint16_t>(20));
        // general purpose bit flag
        write_le(out, static_cast<uint16_t>(0));
        // compression method (0 = store)
        write_le(out, static_cast<uint16_t>(0));
        // last mod file time/date
        write_le(out, static_cast<uint16_t>(0));
        write_le(out, static_cast<uint16_t>(0));
        // crc32
        write_le(out, e.crc);
        // compressed size
        write_le(out, e.compSize);
        // uncompressed size
        write_le(out, e.uncompSize);
        // file name length
        write_le(out, static_cast<uint16_t>(e.name.size()));
        // extra field length
        write_le(out, static_cast<uint16_t>(0));
        // file name
        out.write(e.name.c_str(), e.name.size());
        // file data
        if (!e.data.empty()) out.write(e.data.data(), e.data.size());
    }

    uint32_t centralDirStart = static_cast<uint32_t>(out.tellp());

    // Write central directory
    for (auto& e : entries) {
        // central dir file header signature
        write_le(out, static_cast<uint32_t>(0x02014b50));
        // version made by
        write_le(out, static_cast<uint16_t>(20));
        // version needed to extract
        write_le(out, static_cast<uint16_t>(20));
        // general purpose bit flag
        write_le(out, static_cast<uint16_t>(0));
        // compression method
        write_le(out, static_cast<uint16_t>(0));
        // last mod file time/date
        write_le(out, static_cast<uint16_t>(0));
        write_le(out, static_cast<uint16_t>(0));
        // crc32
        write_le(out, e.crc);
        // compressed size
        write_le(out, e.compSize);
        // uncompressed size
        write_le(out, e.uncompSize);
        // file name length
        write_le(out, static_cast<uint16_t>(e.name.size()));
        // extra field length
        write_le(out, static_cast<uint16_t>(0));
        // file comment length
        write_le(out, static_cast<uint16_t>(0));
        // disk number start
        write_le(out, static_cast<uint16_t>(0));
        // internal file attributes
        write_le(out, static_cast<uint16_t>(0));
        // external file attributes
        write_le(out, static_cast<uint32_t>(0));
        // relative offset of local header
        write_le(out, e.localHeaderOffset);
        // file name
        out.write(e.name.c_str(), e.name.size());
    }

    uint32_t centralDirEnd = static_cast<uint32_t>(out.tellp());
    uint32_t centralDirSize = centralDirEnd - centralDirStart;

    // End of central directory record
    write_le(out, static_cast<uint32_t>(0x06054b50));
    write_le(out, static_cast<uint16_t>(0)); // number of this disk
    write_le(out, static_cast<uint16_t>(0)); // disk where central directory starts
    write_le(out, static_cast<uint16_t>(entries.size())); // number of central dir records on this disk
    write_le(out, static_cast<uint16_t>(entries.size())); // total central dir records
    write_le(out, centralDirSize);
    write_le(out, centralDirStart);
    write_le(out, static_cast<uint16_t>(0)); // comment length

    out.close();

    return true;
}

} // namespace BeatSync
