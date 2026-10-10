using System.Runtime.CompilerServices;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

[assembly: InternalsVisibleTo("Strict.HighLevelRuntime.Tests")]
[assembly: InternalsVisibleTo("Strict.TestRunner")]

namespace Strict.HighLevelRuntime;

public partial class Interpreter
{
	private bool TryExecuteNativeFileConstructor(Method method, ValueInstance instance,
		IReadOnlyList<ValueInstance> args, ExecutionContext? parentContext, out ValueInstance result)
	{
		result = noneInstance;
		if (method.Type != fileType || method.Name != Method.From || !instance.Equals(noneInstance) ||
			args.Count != 1 || !FileValue.TryGetPathText(args[0], out var path))
			return false;
		result = NativeFileRegistry.Open(method.Type, path);
		parentContext?.TrackDisposable(result);
		return true;
	}

	private bool TryExecuteNativeFileMethod(Method method, ValueInstance instance,
		IReadOnlyList<ValueInstance> args, out ValueInstance result)
	{
		result = noneInstance;
		if (!FileValue.TryGetHandle(instance, fileType, out var handle) ||
			!IsNativeFileMethod(method.Name))
			return false;
		switch (method.Name)
		{
		case "ReadLines":
			result = CreateTexts(method, NativeFileRegistry.ReadLines(handle));
			return true;
		case "ReadBytes":
			result = CreateBytesValue(method, NativeFileRegistry.ReadBytes(handle));
			return true;
		case "Write":
			WriteFile(handle, args, method);
			return true;
		case "Delete":
			NativeFileRegistry.Delete(handle);
			return true;
		case "Close":
			NativeFileRegistry.Close(handle);
			return true;
		case "Exists":
			result = ToBoolean(NativeFileRegistry.Exists(handle));
			return true;
		case "Length":
			result = new ValueInstance(numberType, NativeFileRegistry.Length(handle));
			return true;
		default:
			return false;
		}
	}

	private static bool IsNativeFileMethod(string methodName) =>
		methodName is "ReadLines" or "ReadBytes" or "Write" or "Delete" or "Close" or "Exists"
			or "Length";

	private static bool TryExecuteTextWriterWrite(Method method, ValueInstance[] args)
	{
		if (method.Name != "Write" || method.Type.Name != Type.TextWriter || args.Length == 0)
			return false;
		Console.WriteLine(FormatWriteArgument(args[0]));
		return true;
	}

	private static string FormatWriteArgument(ValueInstance value)
	{
		if (value.IsText)
			return value.Text;
		return value.IsList
			? string.Join(Environment.NewLine, value.List.Items.Select(FormatWriteArgument))
			: value.ToExpressionCodeString();
	}

	private static void WriteFile(long handle, IReadOnlyList<ValueInstance> args, Method method)
	{
		if (args.Count == 0)
			throw new MissingArgument(method, "text", args);
		if (args[0].IsText)
			NativeFileRegistry.WriteText(handle, args[0].Text);
		else if (method.Type.Name == Type.TextWriter && args[0].IsList)
			NativeFileRegistry.WriteLines(handle, args[0].List.Items.Select(item => item.Text));
		else if (args[0].IsList)
			NativeFileRegistry.WriteBytes(handle, FileValue.GetBytes(args[0]));
		else
			throw new InvalidTypeForArgument(method.Type, args, 0);
	}

	internal ValueInstance CreateTexts(Method method, string[] lines)
	{
		var textsType = method.GetListImplementationType(method.GetType(Type.Text));
		var values = new ValueInstance[lines.Length];
		for (var index = 0; index < lines.Length; index++)
			values[index] = new ValueInstance(lines[index]);
		return new ValueInstance(textsType, values);
	}

	private static ValueInstance CreateBytesValue(Method method, byte[] bytes)
	{
		var byteType = method.GetType(Type.Byte);
		var bytesType = method.GetListImplementationType(byteType);
		return FileValue.CreateBytes(bytesType, byteType, bytes);
	}
}
