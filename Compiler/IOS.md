# iOS compiler and MacMini handoff

`strict build <file> ios` builds an arm64 static archive for iOS 15.0 devices, `strict build <file> ios simulator` one for the Simulator of the host (x86_64 on Intel Macs, arm64 on Apple Silicon). Generated compute loops keep their CPU fallback. The exported C entry point is `int strict_main(int argc, char **argv)`.

Release output is `Release/iOS/<Name>.ios.a`, `Release/iOS/<Name>.simulator.a` and `Release/iOS/Content`. Add the archive and content to an Xcode app target. This is an input to an app build, not a signed app or IPA. Plain Strict programs can produce their device archive on Windows using LLVM without an Apple SDK.

Native packages require macOS, full Xcode and a working `xcrun --sdk iphoneos --show-sdk-path` (`iphonesimulator` for the Simulator). Their `native/iOS.flags` supply include paths, definitions and extra C/Objective-C sources (`-Wl,` options are for the app's link). Host LLVM compiles Strict IR, Xcode Clang compiles native sources against the SDK, and `llvm-ar` creates the archive. Missing tools or SDKs fail the build. The x86_64 Simulator is compiled for `x86-64-v2` (without SSE4.1 `llvm.roundeven` becomes a libm call the iOS SDK does not have).

The runtime shares Darwin facilities with macOS (`NativeRuntimeDarwin`, the 64 bit inode `NativeRuntimeDirectories`) and the register spilling of its architecture (`NativeRuntimeArm64`, `NativeRuntimeX64`). iOS identifies itself separately, uses the app's `TMPDIR` for temporary files, reports temporary-file creation failures, disables desktop GPU loading and reserves a provisional 1 GiB virtual heap. Device memory limits, pointer tagging and collector behavior still require real-device checks.

## DeltaEngine games

`Frameworks/native/IOSWindow.m` is the UIKit host linked from the archive: it owns `main`, changes into the app bundle (Content is read from there), creates an OpenGL ES 3 context on a `CAEAGLLayer` view (Retina, rotation reallocates the framebuffer) and calls `strict_main` on the main thread. `Window_Frame` presents and runs the main run loop until the next `CADisplayLink` refresh, so touches (`IOS_OnTouch`, in pixels) and lifecycle events arrive between frames and a game in the background waits there without touching OpenGL. miniaudio's iOS backend is Objective-C, so `IOSWindow.m` includes `Audio.c`. The Xcode app target needs no sources: the archive, `Content` as a folder reference, an Info.plist (landscape, `UILaunchScreen`) and the frameworks of `Frameworks/native/iOS.flags`.

## Verified on the MacMini (Intel, macOS 15.8, Xcode 26.3, 2026-10-26)

- `Examples/Sum.strict ios` links into an arm64 iOS app with the iPhoneOS 26.2 SDK; `ios simulator` linked into a command line program prints 35 in the iOS 17 Simulator (`xcrun simctl spawn`).
- `Games/SoulCraft/SoulCraft.strict ios` links into an arm64 app; the `ios simulator` archive in a `.app` (ad-hoc signed, Content copied) starts in the iPhone 15 Simulator and renders the menu with the 3D hero, fonts and textures.
- Not verified yet: touch input, audio output, rotation, background and resume, saves (the game writes next to its Content, which is the read-only signed bundle on iOS: needs the app's Documents folder), physical devices (no signing identity on the MacMini).

## Next steps

1. Simulator checks of SoulCraft: touch (tap PLAY, drag), audio, rotation, background and resume. Writable saves: change into the app's Documents folder for writing while Content stays readable from the bundle.
2. A physical arm64 device (needs a signing identity of team DeltaEngine GmbH on the MacMini, none is installed): startup, temporary files in the sandbox, stack bounds, allocation and collection under memory pressure, source-linked Error diagnostics, calibrate the 1 GiB reservation.
3. Restore app identity, save compatibility and platform services from the actual released SoulCraft app. Confirm identifiers and signing with App Store Connect; do not infer SoulCraft 1 identity from the SoulCraft 3 prototype. Create app icons, launch configuration, entitlements and provisioning in the app project.
4. Create an XCFramework from the device and Simulator archives (`xcodebuild -create-xcframework`), build an Xcode archive, validate it and deploy a TestFlight build using the team's current signing setup. No signing credentials or provisioning profiles are copied into this repository.

## Legacy references

- `C:/code/DeltaEngine/DeltaEngine.Frameworks.IOS`: `RootViewController.cs` owns the GLK view, frame callback and Retina scaling; `IOSWindow.cs`, `Input/IOSTouch.cs`, `Multimedia/*`, `Authentication/*` and `Achievements/*` show the previous bindings. Its current project targets plain `net10.0`; this is reference code, not a verified working iOS build.
- `//AiServer/code/MacOSCodeFolderSafe2018/code/SoulCraft3IOS/SoulCraft/Info.plist`: `de.mobilebits.soulcraft3`, version 1.0, minimum iOS 8.0. This identifies an old SoulCraft 3 project, not necessarily the released SoulCraft app.
- `//AiServer/code/MacOSCodeFolderSafe2018/code/SoulCraft1` and `code/HeroesOfSoulCraftIOS`: older engine/game sources. The backup also contains old IPA files and provisioning material; treat these as historical references, not current signing assets.

Apple documents static libraries and separate platform variants in [Creating a multiplatform binary framework bundle](https://developer.apple.com/documentation/xcode/creating-a-multi-platform-binary-framework-bundle).
