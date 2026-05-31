include(GoogleTest)

# Toolchain smoke test — kept for CI toolchain validation
add_executable(catbowlwatch_tests test_smoke.cpp)
target_link_libraries(catbowlwatch_tests PRIVATE
    catbowlwatch_warnings
    GTest::gtest_main
)
gtest_discover_tests(catbowlwatch_tests)

# Phase 3 component unit tests (no ONNX model required)
add_executable(catbowlwatch_unit_tests
    test_preprocessor.cpp
    test_postprocessor.cpp
    test_bowl_tracker.cpp
    test_debounce_engine.cpp
)
target_link_libraries(catbowlwatch_unit_tests PRIVATE
    catbowlwatch_lib
    GTest::gtest_main
)
gtest_discover_tests(catbowlwatch_unit_tests)

# Phase 4 TelegramNotifier unit tests (no network required — HTTP is injected)
add_executable(catbowlwatch_notifier_tests
    test_telegram_notifier.cpp
)
target_link_libraries(catbowlwatch_notifier_tests PRIVATE
    catbowlwatch_lib
    GTest::gtest_main
)
gtest_discover_tests(catbowlwatch_notifier_tests)
