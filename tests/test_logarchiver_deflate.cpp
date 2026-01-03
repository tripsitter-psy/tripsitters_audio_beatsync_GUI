#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdint>

#include "../src/utils/LogArchiver.h"

// Minimal test: create a tiny file, zip it using createZip (store method), and
// validate that the local file header compression method is 0 (stored).

int main() {
    const char* testFile = "test_deflate_stub_input.txt";
    const char* outZip = "test_deflate_stub_output.zip";

    // Write a small test file
    std::ofstream fout(testFile, std::ios::binary);
    if (!fout) {
        std::cerr << "Could not write test input file" << std::endl;
        return 1;
    }
    fout << "BeatSync test";
    fout.close();

    std::string err;
    std::vector<std::string> files = { std::string(testFile) };
    if (!BeatSync::createZip(files, std::string(outZip), err)) {
        std::cerr << "createZip failed: " << err << std::endl;
        std::remove(testFile);
        return 2;
    }

    // Open zip and find local file header signature 0x04034b50
    std::ifstream zip(outZip, std::ios::binary);
    if (!zip) {
        std::cerr << "Could not open generated zip" << std::endl;
        std::remove(testFile);
        return 3;
    }

    // Read entire file into memory
    std::vector<char> data((std::istreambuf_iterator<char>(zip)), std::istreambuf_iterator<char>());

    auto find_sig = [&](uint32_t sig)->size_t {
        for (size_t i = 0; i + 4 <= data.size(); ++i) {
            uint32_t v = static_cast<uint8_t>(data[i]) |
                         (static_cast<uint8_t>(data[i+1]) << 8) |
                         (static_cast<uint8_t>(data[i+2]) << 16) |
                         (static_cast<uint8_t>(data[i+3]) << 24);
            if (v == sig) return i;
        }
        return std::string::npos;
    };

    size_t off = find_sig(0x04034b50);
    if (off == std::string::npos) {
        std::cerr << "Local file header signature not found in zip" << std::endl;
        std::remove(testFile);
        std::remove(outZip);
        return 4;
    }

    // Compression method is 2 bytes at offset off + 8
    if (off + 10 > data.size()) {
        std::cerr << "Zip too small to contain compression method" << std::endl;
        std::remove(testFile);
        std::remove(outZip);
        return 5;
    }
    uint16_t method = static_cast<uint8_t>(data[off+8]) | (static_cast<uint8_t>(data[off+9]) << 8);

    if (method != 0) {
        std::cerr << "Unexpected compression method: " << method << " (expected 0 = store).\n";
        std::cerr << "If DEFLATE is implemented later, update or extend this test accordingly." << std::endl;
        std::remove(testFile);
        std::remove(outZip);
        return 6;
    }

    std::cout << "PASS: Zip uses store method (compression=0)" << std::endl;

    // Cleanup
    std::remove(testFile);
    std::remove(outZip);
    return 0;
}
