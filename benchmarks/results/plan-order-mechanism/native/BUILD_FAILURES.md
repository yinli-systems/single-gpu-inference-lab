# Retained build failure

2026-09-29: first Mac host-only compile failed at link, before any tests ran. The selected clang/linker could not parse the MacOSX27.0 SDK .tbd architecture arm64e.x1-macos (libSystem and libc++). The user's SDK and global toolchain were not modified. This is not a C++ algorithm correctness failure and is not reported as a passing test. The portable test now uses xcrun clang++ on Darwin and g++ -shared -fPIC on Linux. Linux CPU qualification is a separate explicitly reported environment, not a Mac GPU test.
