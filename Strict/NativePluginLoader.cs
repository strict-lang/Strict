using System.Reflection;
using System.Runtime.InteropServices;
using Strict.Expressions;
using Strict.Language;

namespace Strict;

/// <summary>
/// Searches for native implementations of trait methods.
/// Supports two modes:
/// 1. Lifecycle-based native plugins (C/C++/Rust shared libraries) via NativeLibrary.Load.
///    Convention: {TypeName}_Create(path) → handle, {TypeName}_Colors(handle, &count) → bytes,
///    {TypeName}_Delete(handle). The shared library file must be named ImageLoader.so /
///    ImageLoader.dll / ImageLoader.dylib and live in the working directory.
/// 2. Managed .NET assemblies via Assembly.LoadFrom (legacy / fallback path).
/// </summary>
public static class NativePluginLoader
{
	// Delegate types for the three-step native lifecycle: Create → Colors → Delete
	[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
	private delegate IntPtr CreateDelegate([MarshalAs(UnmanagedType.LPUTF8Str)] string path);

	[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
	private delegate IntPtr ColorsDelegate(IntPtr handle, out int outWidth, out int outHeight);

	[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
	private delegate void DeleteDelegate(IntPtr handle);

	private static readonly Dictionary<string, IntPtr> LoadedNativeLibraries = new();

	/// <summary>
	/// Tries to call the native Create → Colors → Delete lifecycle for a trait type that has a
	/// matching native shared library. Returns the RGBA byte data as managed bytes, or null if
	/// no native library was found for the type.
	/// </summary>
	public static byte[]? TryLoadNativeLifecycle(string typeName, string path,
		string searchDirectory) =>
		TryLoadNativeLifecycle(typeName, path, searchDirectory, out _, out _);

	public static byte[]? TryLoadNativeLifecycle(string typeName, string path,
		string searchDirectory, out int width, out int height)
	{
		width = 0;
		height = 0;
		var libPath = FindNativeLibraryPath(typeName, searchDirectory);
		if (libPath == null)
			return null;
		var libHandle = GetOrLoadNativeLibrary(libPath);
		var createFn = GetNativeFunction<CreateDelegate>(libHandle, typeName + "_Create");
		var colorsFn = GetNativeFunction<ColorsDelegate>(libHandle, typeName + "_Colors");
		var deleteFn = GetNativeFunction<DeleteDelegate>(libHandle, typeName + "_Delete");
		var imageHandle = createFn(path);
		if (imageHandle == IntPtr.Zero)
			throw new NativeCreateFailed(typeName, path);
		try
		{
			var dataPtr = colorsFn(imageHandle, out width, out height);
			var count = width * height * 4;
			if (dataPtr == IntPtr.Zero || count <= 0)
				return [];
			var result = new byte[count];
			Marshal.Copy(dataPtr, result, 0, count);
			return result;
		}
		finally
		{
			deleteFn(imageHandle);
		}
	}

	[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
	private delegate int SaveDelegate(
		[MarshalAs(UnmanagedType.LPUTF8Str)] string path, IntPtr data, int width, int height);

	/// <summary>
	/// Calls {TypeName}_Save(path, data, len, width, height) on a native shared library.
	/// Returns true on success, false if no native library was found.
	/// </summary>
	public static bool TrySaveNativeImage(string typeName, string path, byte[] data,
		int width, int height, string searchDirectory)
	{
		var libPath = FindNativeLibraryPath(typeName, searchDirectory);
		if (libPath == null)
			return false;
		var libHandle = GetOrLoadNativeLibrary(libPath);
		var saveFn = GetNativeFunction<SaveDelegate>(libHandle, typeName + "_Save");
		var pinnedData = GCHandle.Alloc(data, GCHandleType.Pinned);
		try
		{
			var result = saveFn(path, pinnedData.AddrOfPinnedObject(), width, height);
			if (result == 0)
				throw new NativeSaveFailed(typeName, path);
			return true;
		}
		finally
		{
			pinnedData.Free();
		}
	}

	private static string? FindNativeLibraryPath(string typeName, string searchDirectory)
	{
		if (!Directory.Exists(searchDirectory))
			return null;
		foreach (var candidate in GetNativeLibraryCandidates(typeName))
		{
			var fullPath = Path.Combine(searchDirectory, candidate);
			if (File.Exists(fullPath))
				return fullPath;
		}
		return null;
	}

	private static IEnumerable<string> GetNativeLibraryCandidates(string typeName)
	{
		if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
			yield return typeName + ".dll";
		else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
		{
			yield return typeName + ".dylib";
			yield return "lib" + typeName + ".dylib";
		}
		else
		{
			yield return typeName + ".so";
			yield return "lib" + typeName + ".so";
		}
	}

	private static IntPtr GetOrLoadNativeLibrary(string libPath)
	{
		if (LoadedNativeLibraries.TryGetValue(libPath, out var existing))
			return existing;
		var handle = NativeLibrary.Load(libPath);
		LoadedNativeLibraries[libPath] = handle;
		return handle;
	}

	private static TDelegate GetNativeFunction<TDelegate>(IntPtr libHandle, string functionName)
		where TDelegate : Delegate
	{
		var exportPtr = NativeLibrary.GetExport(libHandle, functionName);
		return Marshal.GetDelegateForFunctionPointer<TDelegate>(exportPtr);
	}

	/// <summary>
	/// Legacy: searches .NET assemblies in the given directory for a class and method matching
	/// typeName/methodName by reflection.  Kept for managed plugin scenarios.
	/// </summary>
	public static object? TryCallNativeMethod(string typeName, string methodName,
		object?[] arguments, string searchDirectory)
	{
		var matchingMethod = FindManagedMethod(typeName, methodName, arguments.Length,
			searchDirectory);
		return matchingMethod == null
			? throw new NativeMethodNotFound(typeName, methodName, searchDirectory)
			: matchingMethod.IsStatic
				? matchingMethod.Invoke(null, arguments)
				: InvokeInstanceMethod(matchingMethod, arguments);
	}

	private static object? InvokeInstanceMethod(MethodInfo method, object?[] arguments)
	{
		var instance = Activator.CreateInstance(method.DeclaringType!);
		return method.Invoke(instance, arguments);
	}

	private static MethodInfo? FindManagedMethod(string typeName, string methodName,
		int argumentCount, string searchDirectory)
	{
		foreach (var dllPath in GetDllFiles(searchDirectory))
		{
			try
			{
				var assembly = Assembly.LoadFrom(dllPath);
				var matchingType = FindTypeByName(assembly, typeName);
				if (matchingType == null)
					continue;
				var matchingMethod = FindMethodByName(matchingType, methodName, argumentCount);
				if (matchingMethod != null)
					return matchingMethod;
			}
			catch (Exception)
			{
				// Skip DLLs that can't be loaded (bad format, missing deps, etc.)
			}
		}
		return null;
	}

	private static string[] GetDllFiles(string directory) =>
		Directory.Exists(directory)
			? Directory.GetFiles(directory, "*.dll")
			: [];

	private static System.Type? FindTypeByName(Assembly assembly, string typeName)
	{
		foreach (var type in assembly.GetExportedTypes())
			if (type.Name.Equals(typeName, StringComparison.OrdinalIgnoreCase))
				return type;
		return null;
	}

	private static MethodInfo? FindMethodByName(System.Type type, string methodName,
		int argumentCount)
	{
		foreach (var method in type.GetMethods(BindingFlags.Public | BindingFlags.Instance |
			BindingFlags.Static))
			if (method.Name.Equals(methodName, StringComparison.OrdinalIgnoreCase) &&
				method.GetParameters().Length == argumentCount)
				return method;
		return null;
	}

	public static ValueInstance ConvertToValueInstance(object? result, Language.Type returnType)
	{
		if (result == null)
			return new ValueInstance(returnType, 0);
		if (result is byte[] bytes)
			return ConvertBytesToValueInstance(bytes, returnType);
		if (result is string text)
			return new ValueInstance(text);
		if (result is double or float or int or long)
			return new ValueInstance(returnType, Convert.ToDouble(result));
		if (result is bool boolean)
			return new ValueInstance(returnType, boolean);
		return new ValueInstance(returnType, Convert.ToDouble(result));
	}

	public static ValueInstance ConvertBytesToValueInstance(byte[] bytes, Language.Type returnType)
	{
		var elementType = returnType is GenericTypeImplementation generic
			? generic.ImplementationTypes[0]
			: returnType;
		var elements = new ValueInstance[bytes.Length];
		for (var byteIndex = 0; byteIndex < bytes.Length; byteIndex++)
			elements[byteIndex] = new ValueInstance(elementType, (double)bytes[byteIndex]);
		return new ValueInstance(returnType, elements);
	}

	public sealed class NativeMethodNotFound(string typeName, string methodName,
		string searchDirectory) : Exception(
		$"No native implementation found for {typeName}.{methodName} in DLLs in {searchDirectory}");

	public sealed class NativeCreateFailed(string typeName, string path) : Exception(
		$"Native {typeName}_Create returned null for path: {path}");

	public sealed class NativeSaveFailed(string typeName, string path) : Exception(
		$"Native {typeName}_Save failed for path: {path}");
}

