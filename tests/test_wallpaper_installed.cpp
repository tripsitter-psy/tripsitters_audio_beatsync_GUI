#include <catch2/catch.hpp>
#include <filesystem>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: test_wallpaper_installed <nonmac_path> <mac_path>" << std::endl;
        return 2; // indicate misconfiguration
    }
    std::filesystem::path nonmac = argv[1];
    std::filesystem::path mac = argv[2];

    bool nonmac_exists = std::filesystem::exists(nonmac);
    bool mac_exists = std::filesystem::exists(mac);

    std::cout << "Checking wallpaper locations:\n";
    std::cout << "  non-mac: " << nonmac << " => " << (nonmac_exists ? "FOUND" : "MISSING") << "\n";
    std::cout << "  mac:     " << mac << " => " << (mac_exists ? "FOUND" : "MISSING") << "\n";

    if (!nonmac_exists && !mac_exists) {
        std::cerr << "Wallpaper not found in either build location." << std::endl;
        return 1; // fail the test
    }
    return 0; // success
}
