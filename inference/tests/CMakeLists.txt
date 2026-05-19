# Phase 3 unit tests. Currently a single smoke test that confirms GoogleTest is
# wired correctly. Real test files (test_debounce_engine, test_bowl_tracker,
# test_postprocessor) land in subsequent commits.

include(GoogleTest)

add_executable(catbowlwatch_tests test_smoke.cpp)
target_link_libraries(catbowlwatch_tests PRIVATE
    catbowlwatch_warnings
    GTest::gtest_main
)

gtest_discover_tests(catbowlwatch_tests)
