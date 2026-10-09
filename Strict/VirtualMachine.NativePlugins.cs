using System.Globalization;
using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict;

public sealed partial class VirtualMachine
{
	private bool TryHandleNativeTraitInstanceMethod(Invoke invoke, ValueInstance? implicitInstance)
	{
		var info = invoke.MethodInfo;
		var instanceValue = ResolveInvokeInstance(info, implicitInstance);
		if (instanceValue.IsText || instanceValue.IsList || instanceValue.IsDictionary ||
			instanceValue.IsFlatNumeric)
			return false;
		if (!IsTrait(instanceValue.GetType()))
			return false;
		var typeInstance = instanceValue.TryGetValueTypeInstance();
		if (typeInstance == null)
			return false;
		var methodIndex = GetTraitDataMethodIndex(instanceValue.GetType(), info.MethodName);
		if (methodIndex < 0 || methodIndex >= typeInstance.Values.Length)
			return false;
		Memory.Registers[invoke.Register] = typeInstance.Values[methodIndex];
		return true;
	}

	private static int GetTraitDataMethodIndex(Type traitType, string methodName)
	{
		var dataIndex = 0;
		foreach (var method in traitType.Methods)
		{
			if (string.Equals(method.Name, Method.From, StringComparison.OrdinalIgnoreCase))
				continue;
			if (string.Equals(method.Name, methodName, StringComparison.OrdinalIgnoreCase))
				return dataIndex;
			dataIndex++;
		}
		return -1;
	}

	private bool TryHandleNativeTraitStaticMethod(Invoke invoke)
	{
		var info = invoke.MethodInfo;
		if (info.MethodName != "Save")
			return false;
		var typeName = info.TypeFullName.Split('/').Last();
		var searchDirectory = AppContext.BaseDirectory;
		if (!NativePluginLoader.HasNativeLibrary(typeName, searchDirectory))
			return false;
		if (info.ArgumentRegisters.Length < 4)
			return false;
		if (!FileValue.TryGetPathText(Memory.Registers[info.ArgumentRegisters[0]], out var pathText))
			return false;
		var colorsArg = Memory.Registers[info.ArgumentRegisters[1]];
		if (!colorsArg.IsList)
			return false;
		var width = (int)Memory.Registers[info.ArgumentRegisters[2]].Number;
		var height = (int)Memory.Registers[info.ArgumentRegisters[3]].Number;
		var pixelData = ExtractRgbaBytes(colorsArg);
		return NativePluginLoader.TrySaveNativeImage(typeName, pathText, pixelData, width, height,
			searchDirectory);
	}

	private static byte[] ExtractRgbaBytes(ValueInstance listValue)
	{
		var items = listValue.List.Items;
		if (items.Count == 0)
			return [];
		return items[0].TryGetValueTypeInstance() != null
			? ExtractBytesFromColorList(items)
			: ExtractBytesFromNumberList(items);
	}

	private static byte[] ExtractBytesFromNumberList(IReadOnlyList<ValueInstance> items)
	{
		var bytes = new byte[items.Count];
		for (var byteIndex = 0; byteIndex < items.Count; byteIndex++)
			bytes[byteIndex] = (byte)Math.Clamp(items[byteIndex].Number, 0, 255);
		return bytes;
	}

	private static byte[] ExtractBytesFromColorList(IReadOnlyList<ValueInstance> items)
	{
		var bytes = new byte[items.Count * 4];
		var isColorType = IsColorByteType(items[0]);
		for (var colorIndex = 0; colorIndex < items.Count; colorIndex++)
		{
			var typeInst = items[colorIndex].TryGetValueTypeInstance();
			if (typeInst is null || typeInst.Values.Length < 3)
				continue;
			if (isColorType)
			{
				bytes[colorIndex * 4] = ClampToByte(typeInst.Values[0].Number);
				bytes[colorIndex * 4 + 1] = ClampToByte(typeInst.Values[1].Number);
				bytes[colorIndex * 4 + 2] = ClampToByte(typeInst.Values[2].Number);
				bytes[colorIndex * 4 + 3] = typeInst.Values.Length >= 4
					? ClampToByte(typeInst.Values[3].Number)
					: (byte)255;
			}
			else
			{
				bytes[colorIndex * 4] = ClampToByte(typeInst.Values[0].Number * 255);
				bytes[colorIndex * 4 + 1] = ClampToByte(typeInst.Values[1].Number * 255);
				bytes[colorIndex * 4 + 2] = ClampToByte(typeInst.Values[2].Number * 255);
				bytes[colorIndex * 4 + 3] = typeInst.Values.Length >= 4
					? ClampToByte(typeInst.Values[3].Number * 255)
					: (byte)255;
			}
		}
		return bytes;
	}

	private static bool IsColorByteType(ValueInstance colorInstance) =>
		colorInstance.GetType().Name.Equals("Color", StringComparison.OrdinalIgnoreCase);

	private static byte ClampToByte(double value) => (byte)Math.Clamp(Math.Round(value), 0, 255);

	private bool TryCallNativeFromPlugin(Invoke invoke, Type returnType)
	{
		var info = invoke.MethodInfo;
		if (info.ArgumentRegisters.Length == 0)
			return false;
		if (!FileValue.TryGetPathText(Memory.Registers[info.ArgumentRegisters[0]], out var pathText))
			return false;
		var searchDirectory = AppContext.BaseDirectory;
		var bytes = NativePluginLoader.TryLoadNativeLifecycle(returnType.Name, pathText,
			searchDirectory, out var width, out var height);
		if (bytes == null)
			return false;
		var traitValues = BuildNativePluginValues(returnType, bytes, width, height);
		if (traitValues == null)
			return false;
		Memory.Registers[invoke.Register] = new ValueInstance(returnType, traitValues);
		return true;
	}

	private ValueInstance[]? BuildNativePluginValues(Type traitType, byte[] bytes, int width,
		int height)
	{
		var dataMethods = traitType.Methods.
			Where(m => !string.Equals(m.Name, Method.From, StringComparison.OrdinalIgnoreCase)).ToList();
		if (dataMethods.Count == 0)
			return null;
		var values = new ValueInstance[dataMethods.Count];
		for (var methodIndex = 0; methodIndex < dataMethods.Count; methodIndex++)
		{
			var method = dataMethods[methodIndex];
			var returnType = method.ReturnType;
			if (returnType.IsList)
				values[methodIndex] = BuildListValueFromBytes(bytes, returnType);
			else if (returnType.IsNumber)
				values[methodIndex] = new ValueInstance(returnType,
					string.Equals(method.Name, "Width", StringComparison.OrdinalIgnoreCase)
						? width
						: height);
			else
				return null;
		}
		return values;
	}

	private ValueInstance BuildListValueFromBytes(byte[] bytes, Type listType)
	{
		var elementType = listType is GenericTypeImplementation generic
			? generic.ImplementationTypes[0]
			: listType;
		if (elementType.IsNumber || string.Equals(elementType.Name, "Byte",
			StringComparison.OrdinalIgnoreCase))
			return NativePluginLoader.ConvertBytesToValueInstance(bytes, listType);
		var numberType = executable.TypeResolver.FindType("Number")!;
		var colorCount = bytes.Length / 4;
		var colorValues = new ValueInstance[colorCount];
		for (var colorIndex = 0; colorIndex < colorCount; colorIndex++)
		{
			var r = bytes[colorIndex * 4] / 255.0;
			var g = bytes[colorIndex * 4 + 1] / 255.0;
			var b = bytes[colorIndex * 4 + 2] / 255.0;
			var a = bytes[colorIndex * 4 + 3] / 255.0;
			colorValues[colorIndex] = new ValueInstance(elementType, [
				new ValueInstance(numberType, r), new ValueInstance(numberType, g),
				new ValueInstance(numberType, b), new ValueInstance(numberType, a)
			]);
		}
		return new ValueInstance(listType, colorValues);
	}
}
