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
	private bool TryHandleNativeFileMethod(Invoke invoke, ValueInstance? implicitInstance)
	{
		var info = invoke.MethodInfo;
		if (info.MethodName == Method.From)
		{
			if (info.ArgumentRegisters.Length != 1)
				return false;
			if (!FileValue.TryGetPathText(Memory.Registers[info.ArgumentRegisters[0]], out var pathText))
				return false;
			var fileInstance = NativeFileRegistry.Open(executable.TypeResolver.GetType(Type.File),
				pathText);
			Memory.Frame.TrackDisposable(fileInstance);
			Memory.Registers[invoke.Register] = fileInstance;
			return true;
		}
		var instance = ResolveInvokeInstance(info, implicitInstance);
		if (!IsFileInstance(instance))
			return false;
		var handle = GetFileHandle(instance);
		switch (info.MethodName)
		{
		case "ReadLines":
			Memory.Registers[invoke.Register] = CreateTextListValue(NativeFileRegistry.ReadLines(handle));
			return true;
		case "ReadBytes":
			Memory.Registers[invoke.Register] = CreateBytesValue(NativeFileRegistry.ReadBytes(handle));
			return true;
		case "Write":
			WriteFile(handle, Memory.Registers[info.ArgumentRegisters[0]],
				info.TypeFullName.EndsWith(Context.ParentSeparator + Type.TextWriter,
					StringComparison.Ordinal));
			Memory.Registers[invoke.Register] = new ValueInstance(executable.noneType);
			return true;
		case "Delete":
			NativeFileRegistry.Delete(handle);
			Memory.Registers[invoke.Register] = new ValueInstance(executable.noneType);
			return true;
		case "Close":
			NativeFileRegistry.Close(handle);
			Memory.Registers[invoke.Register] = new ValueInstance(executable.noneType);
			return true;
		case "Exists":
			Memory.Registers[invoke.Register] =
				new ValueInstance(executable.booleanType, NativeFileRegistry.Exists(handle));
			return true;
		case "Length":
			Memory.Registers[invoke.Register] =
				new ValueInstance(executable.numberType, NativeFileRegistry.Length(handle));
			return true;
		default:
			return false;
		}
	}

	private long GetFileHandle(ValueInstance instance)
	{
		return FileValue.TryGetHandle(instance, executable.TypeResolver.GetType(Type.File),
			out var handle)
			? handle
			: throw Fail("File instance has no native handle");
	}

	private void WriteFile(long handle, ValueInstance value, bool writesTextLines)
	{
		if (value.IsText)
			NativeFileRegistry.WriteText(handle, value.Text);
		else if (writesTextLines && value.IsList)
			NativeFileRegistry.WriteLines(handle, value.List.Items.Select(item => item.Text));
		else if (value.IsList)
			NativeFileRegistry.WriteBytes(handle, FileValue.GetBytes(value));
		else
			throw Fail("File.Write needs Text or Bytes");
	}

	private ValueInstance CreateBytesValue(byte[] bytes)
	{
		var byteType = executable.TypeResolver.GetType(Type.Byte);
		var bytesType = executable.listType.GetGenericImplementation(byteType);
		return FileValue.CreateBytes(bytesType, byteType, bytes);
	}

	private ValueInstance CreateTextListValue(string[] lines)
	{
		var textType = executable.TypeResolver.GetType(Type.Text);
		var textsType = executable.listType.GetGenericImplementation(textType);
		return new ValueInstance(textsType, lines.Select(line => new ValueInstance(line)).ToArray());
	}

	private bool TryHandleDirectoryStatic(Invoke invoke)
	{
		var info = invoke.MethodInfo;
		if (info.MethodName == "Exists" && info.ArgumentRegisters.Length == 1)
		{
			var path = GetArgumentText(Memory.Registers[info.ArgumentRegisters[0]]);
			Memory.Registers[invoke.Register] =
				new ValueInstance(executable.booleanType, NativeDirectory.Exists(path));
			return true;
		}
		if (info.MethodName == "Create" && info.ArgumentRegisters.Length == 1)
		{
			var path = GetArgumentText(Memory.Registers[info.ArgumentRegisters[0]]);
			NativeDirectory.Create(path);
			Memory.Registers[invoke.Register] = new ValueInstance(executable.noneType);
			return true;
		}
		if (info.MethodName is "Files" or "GetFiles" && info.ArgumentRegisters.Length >= 1)
		{
			var path = GetArgumentText(Memory.Registers[info.ArgumentRegisters[0]]);
			var pattern = info.ArgumentRegisters.Length >= 2
				? GetArgumentText(Memory.Registers[info.ArgumentRegisters[1]])
				: "";
			Memory.Registers[invoke.Register] =
				CreateTextListValue(NativeDirectory.GetFiles(path, pattern));
			return true;
		}
		if (info.MethodName == "Directories" && info.ArgumentRegisters.Length == 1)
		{
			Memory.Registers[invoke.Register] = CreateTextListValue(
				NativeDirectory.GetDirectories(GetArgumentText(Memory.Registers[info.ArgumentRegisters[0]])));
			return true;
		}
		return false;
	}

	private static string GetArgumentText(ValueInstance value) =>
		FileValue.TryGetPathText(value, out var pathText)
			? pathText
			: value.ToExpressionCodeString();

	private bool IsFileInstance(ValueInstance instance)
	{
		if (!instance.HasValue)
			return false;
		var fileType = executable.TypeResolver.FindType(Type.File);
		return fileType != null && instance.GetType().IsSameOrCanBeUsedAs(fileType);
	}
}
