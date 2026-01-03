#include <catch2/catch_all.hpp>

// Minimal Catch2 stub for future DEFLATE tests
TEST_CASE("DEFLATE stub is present", "[deflate][stub]") {
    // This is a placeholder test. When DEFLATE is implemented, replace or extend
    // this test with concrete assertions (e.g., compressed size < uncompressed size,
    // valid ZIP headers, and that the GUI honors `ZipUseDeflate`).
    REQUIRE(true);
}

// Hidden, actionable test template for future DEFLATE validation. This test is
// intentionally hidden (tag starts with a dot) so it won't run in normal test
// invocations until someone enables it. When DEFLATE is implemented, replace
// the body with actual assertions and remove the leading dot from the tag.
TEST_CASE("DEFLATE integration (enable when implemented)", "[.deflate][todo]") {
    /*
    TODO (when implementing DEFLATE):
    - Create small test files (text/binary).
    - Call the API that creates DEFLATE-compressed ZIPs (e.g., BeatSync::createZip with a new parameter or a dedicated method).
    - Open the generated ZIP and assert:
        * Local file header compression method == 8 (DEFLATE).
        * Compressed size < uncompressed size (for compressible data).
        * CRC matches original data.
        * Extract and verify extracted contents match original files.
    - Optionally, verify GUI `Save Logs...` honors the `ZipUseDeflate` setting by invoking a small helper or mocking the settings store.
    */

    SUCCEED("DEFLATE test placeholder — replace with real assertions when DEFLATE is implemented");
}
