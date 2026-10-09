namespace Strict.Compiler;

public enum Platform
{
	Windows,
	Linux,
	MacOS
}

public sealed class UnsupportedPlatform(Platform platform)
	: Exception("Unsupported platform: " + platform);