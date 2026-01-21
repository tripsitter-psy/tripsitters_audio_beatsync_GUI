#include <catch2/catch_test_macros.hpp>
#include "backend/beatsync_capi.h"

TEST_CASE("AI create/destroy stress test", "[backend][ai][cleanup]") {
    bs_ai_config_t config = {};
    config.beat_model_path = nullptr; // may deferred-load, we want to test ctor/dtor
    config.use_gpu = 1;
    config.gpu_device_id = 0;

    const int iterations = 50;
    for (int i = 0; i < iterations; ++i) {
        void* a = bs_create_ai_analyzer(&config);
        // If creation fails, that's fine; ensure destroy handles nullptr
        bs_destroy_ai_analyzer(a);
    }

    SUCCEED("Completed create/destroy loop without crashing");
}
