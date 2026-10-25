# iOS compiler and MacMini handoff

`strict build <file> ios` runs host VM tests and builds an arm64 static archive for iOS 15.0. Generated compute loops keep their CPU fallback. The exported C entry point is `int strict_main(int argc, char **argv)`; UIKit retains ownership of the application's `main` and lifecycle.

Release output is `Release/iOS/<Name>.ios.a` and `Release/iOS/Content`. Add the archive and content to an Xcode app target. This is an input to an app build, not a signed app or IPA. Plain Strict programs can produce their archive on Windows using LLVM without an Apple SDK. An archive does not resolve system symbols, so SDK linking and execution remain unverified there.

Native packages require macOS, full Xcode and a working `xcrun --sdk iphoneos --show-sdk-path`. Their `native/IOS.flags` supply include paths, definitions and extra C/Objective-C sources; Xcode supplies final app frameworks and libraries. Host LLVM compiles Strict IR, Xcode Clang compiles native sources against the iPhoneOS SDK, and `llvm-ar` creates the archive. Missing tools or SDKs fail the build. Device and Simulator binaries are different platforms: this target currently builds device arm64 only.

The runtime shares Darwin facilities and arm64 register spilling with macOS. iOS identifies itself separately, uses the app's `TMPDIR` for temporary files, reports temporary-file creation failures, disables desktop GPU loading and reserves a provisional 1 GiB virtual heap. Device memory limits, pointer tagging and collector behavior still require real-device checks.

## Verified on Windows

- Platform-name regression and configurable entry-symbol regression failed before implementation and pass afterward.
- `strict build Examples/Sum.strict ios` produces a static archive containing a 64-bit arm64 Mach-O object with `LC_BUILD_VERSION` platform `ios`, minimum version 15.0 and exported `_strict_main`, without `_main`.
- `strict build Compiler/IOSBuild.strict` runs its native tests on the host.
- SoulCraft's host tests run from the DeltaEngine root; native package compilation needs the Mac SDK.

## MacMini continuation

1. Bring the entire Strict working tree, including the regenerated `Compiler/Bootstrap.mlir`, `NativeRuntimeDarwin.mlir`, `NativeRuntimeIOS.mlir` and existing Android changes. Keep the Strict and DeltaEngine clones side by side. Rebuild the native compiler and Strict command with the documented bootstrap if needed; use full Xcode, not only Command Line Tools.
2. Verify `xcrun --sdk iphoneos --show-sdk-path`, `strict check Compiler`, changed-file VM/native tests and a native macOS Sum build. Build `Examples/Sum.strict ios`, link the archive into a minimal Xcode iOS app, invoke `strict_main` once with the Sum arguments and verify output 35. Resolve SDK availability diagnostics here; Windows archive creation cannot detect them.
3. Run that harness on a physical arm64 device. Verify startup, temporary files in the sandbox, stack bounds, allocation/collection under memory pressure and source-linked Error diagnostics. Calibrate the 1 GiB reservation rather than assuming desktop virtual-memory limits.
4. From DeltaEngine root, run `strict build Games/SoulCraft/SoulCraft.strict ios`. Add focused `IOS.flags` and a native UIKit window, rendering, touch and audio bridge in shared engine packages. Keep SoulCraft rules in Strict. Supply app-bundle resource lookup and writable save paths without writing into the signed bundle. Bind the existing game lifecycle to UIKit's event loop.
5. Restore app identity, save compatibility and platform services from the actual released SoulCraft app. Confirm identifiers and signing with App Store Connect; do not infer SoulCraft 1 identity from the SoulCraft 3 prototype. Create app icons, launch configuration, entitlements and provisioning in the app project. Validate orientation, Retina scaling, suspension/resume, audio interruption and background behavior on hardware.
6. Add a separate Simulator target if needed, then create an XCFramework from separate device/Simulator archives. Build an Xcode archive, validate it and deploy a TestFlight build using the team's current signing setup. No signing credentials or provisioning profiles are copied into this repository.

## Legacy references

- `C:/code/DeltaEngine/DeltaEngine.Frameworks.IOS`: `RootViewController.cs` owns the GLK view, frame callback and Retina scaling; `IOSWindow.cs`, `Input/IOSTouch.cs`, `Multimedia/*`, `Authentication/*` and `Achievements/*` show the previous bindings. Its current project targets plain `net10.0`; this is reference code, not a verified working iOS build.
- `//AiServer/code/MacOSCodeFolderSafe2018/code/SoulCraft3IOS/SoulCraft/Info.plist`: `de.mobilebits.soulcraft3`, version 1.0, minimum iOS 8.0. This identifies an old SoulCraft 3 project, not necessarily the released SoulCraft app.
- `//AiServer/code/MacOSCodeFolderSafe2018/code/SoulCraft1` and `code/HeroesOfSoulCraftIOS`: older engine/game sources. The backup also contains old IPA files and provisioning material; treat these as historical references, not current signing assets.

Apple documents static libraries and separate platform variants in [Creating a multiplatform binary framework bundle](https://developer.apple.com/documentation/xcode/creating-a-multi-platform-binary-framework-bundle).
