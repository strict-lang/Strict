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
	/// <summary>
	/// A type with its own "to Text" uses it, like the interpreter, instead of the generic formatting.
	/// </summary>
	private bool HasOwnCompiledToText(Invoke invoke, ValueInstance rawValue)
	{
		if (rawValue.IsText || rawValue.IsList || rawValue.IsDictionary)
			return false;
		var type = rawValue.GetType();
		return type is { IsNumber: false, IsBoolean: false, IsCharacter: false, IsNone: false } &&
			invoke.MethodInfo.TypeFullName.EndsWith(type.FullName, StringComparison.Ordinal) &&
			(invoke.CachedInstructions ??= GetPrecompiledMethodInstructions(invoke)) != null;
	}

	private bool TryHandleToConversion(Invoke invoke, ValueInstance? implicitInstance)
	{
		var info = invoke.MethodInfo;
		var conversionType = info.ResolveReturnType(executable.TypeResolver);
		var rawValue = ResolveInvokeInstance(info, implicitInstance);
		if (conversionType.IsText && HasOwnCompiledToText(invoke, rawValue))
			return false;
		if (conversionType.IsText)
		{
			Memory.Registers[invoke.Register] = ConvertToText(rawValue);
			return true;
		}
		if (conversionType.IsNumber)
		{
			Memory.Registers[invoke.Register] = rawValue.IsText
				? new ValueInstance(conversionType,
					Convert.ToDouble(rawValue.Text, CultureInfo.InvariantCulture))
				: rawValue;
			return true;
		}
		// "name to Type" / "name to Name" used heavily by Language parsers
		if (conversionType.Name is "Type" or "Name")
		{
			var text = rawValue.IsText
				? rawValue.Text
				: ConvertToText(rawValue).Text;
			Memory.Registers[invoke.Register] = CreateNamedValueType(conversionType, text);
			return true;
		}
		return false;
	}

	private ValueInstance CreateNamedValueType(Type conversionType, string text)
	{
		var members = conversionType.Members;
		if (members.Count == 0)
			return new ValueInstance(text);
		// Prefer a single Text/Name-like member (Type.Name Text, Name itself, etc.)
		var values = new ValueInstance[members.Count];
		for (var index = 0; index < members.Count; index++)
		{
			var memberType = members[index].Type;
			if (memberType.IsText || memberType.Name is "Name" or "Text")
				values[index] = memberType.IsText
					? new ValueInstance(text)
					: CreateNamedValueType(memberType, text);
			else if (memberType.IsList)
				values[index] = new ValueInstance(memberType, Array.Empty<ValueInstance>());
			else
				values[index] = CreateDefaultValue(memberType);
		}
		return new ValueInstance(conversionType, values);
	}

	private bool TryHandleNativeLength(Invoke invoke, ValueInstance? implicitInstance)
	{
		var instanceValue = ResolveInvokeInstance(invoke.MethodInfo, implicitInstance);
		if (!TryGetNativeLength(instanceValue, invoke.MethodInfo.MethodName, out var lengthValue))
			return false;
		Memory.Registers[invoke.Register] = lengthValue;
		return true;
	}

	private bool TryHandleNativeListContains(Invoke invoke, ValueInstance? implicitInstance)
	{
		var list = ResolveInvokeInstance(invoke.MethodInfo, implicitInstance);
		if (!list.IsList || invoke.MethodInfo.ArgumentRegisters.Length != 1)
			return false;
		var searched = Memory.Registers[invoke.MethodInfo.ArgumentRegisters[0]];
		Memory.Registers[invoke.Register] =
			new ValueInstance(executable.booleanType, list.List.Items.Contains(searched));
		return true;
	}

	private bool TryHandleIncrementDecrement(Invoke invoke, bool isIncrement,
		ValueInstance? implicitInstance)
	{
		var current = ResolveInvokeInstance(invoke.MethodInfo, implicitInstance);
		var delta = isIncrement
			? 1.0
			: -1.0;
		Memory.Registers[invoke.Register] =
			new ValueInstance(current.GetType(), current.Number + delta);
		return true;
	}

	private bool TryHandleNativeBooleanMethod(Invoke invoke, ValueInstance? implicitInstance)
	{
		var info = invoke.MethodInfo;
		var instance = ResolveInvokeInstance(info, implicitInstance);
		// Only true Boolean primitives (value pointer is boolean Type; number is 0/1).
		// Text/list/struct markers must not enter this path (avoids GetType cast failures).
		if (instance.IsText || instance.IsList || instance.IsDictionary ||
			instance.TryGetValueTypeInstance() != null)
			return false;
		if (!instance.IsPrimitiveType(executable.booleanType))
			return false;
		var left = instance.Boolean;
		switch (info.MethodName)
		{
		case "not":
			Memory.Registers[invoke.Register] = new ValueInstance(executable.booleanType, !left);
			return true;
		case BinaryOperator.And:
		case BinaryOperator.Or:
		case BinaryOperator.Xor:
			if (info.ArgumentRegisters.Length != 1)
				return false;
			var right = Memory.Registers[info.ArgumentRegisters[0]].Boolean;
			var result = info.MethodName switch
			{
				BinaryOperator.And => left && right,
				BinaryOperator.Or => left || right,
				_ => left ^ right
			};
			Memory.Registers[invoke.Register] = new ValueInstance(executable.booleanType, result);
			return true;
		default:
			return false;
		}
	}

	private bool TryHandleNativeTextMethod(Invoke invoke, ValueInstance? implicitInstance)
	{
		var info = invoke.MethodInfo;
		var instance = ResolveInvokeInstance(info, implicitInstance);
		if (!instance.IsText)
			return false;
		var text = instance.Text;
		var args = new ValueInstance[info.ArgumentRegisters.Length];
		for (var argIndex = 0; argIndex < info.ArgumentRegisters.Length; argIndex++)
			args[argIndex] = Memory.Registers[info.ArgumentRegisters[argIndex]];
		Memory.Registers[invoke.Register] = info.MethodName switch
		{
			"StartsWith" => EvaluateStartsWith(text, args),
			"IndexOf" => new ValueInstance(executable.numberType,
				text.IndexOf(args[0].Text, args.Length > 1
					? (int)args[1].Number
					: 0, StringComparison.Ordinal)),
			"LastIndexOf" => new ValueInstance(executable.numberType,
				text.LastIndexOf(args[0].Text, StringComparison.Ordinal)),
			"Substring" => EvaluateSubstring(text, args),
			"Upper" => new ValueInstance(text.ToUpperInvariant()),
			"Lower" => new ValueInstance(text.ToLowerInvariant()),
			"Capitalize" => new ValueInstance(text.Length == 0
				? ""
				: char.ToUpperInvariant(text[0]) + text[1..]),
			"Trim" => new ValueInstance(text.Trim()),
			"TrimStart" => new ValueInstance(text.TrimStart()),
			"TrimEnd" => new ValueInstance(text.TrimEnd()),
			_ => throw Fail("Unhandled native text method: " + info.MethodName)
		};
		return true;
	}

	private ValueInstance EvaluateStartsWith(string text, ValueInstance[] args)
	{
		var prefix = args[0].Text;
		var start = args.Length > 1
			? (int)args[1].Number
			: 0;
		var matches = start >= 0 && start + prefix.Length <= text.Length &&
			text.AsSpan(start, prefix.Length).SequenceEqual(prefix);
		return new ValueInstance(executable.booleanType, matches);
	}

	private static ValueInstance EvaluateSubstring(string text, ValueInstance[] args)
	{
		var start = (int)args[0].Number;
		var length = args.Length > 1
			? (int)args[1].Number
			: text.Length - start;
		if (start < 0 || start > text.Length || length <= 0)
			return new ValueInstance("");
		if (start + length > text.Length)
			length = text.Length - start;
		return new ValueInstance(text.Substring(start, length));
	}

	private bool TryGetNativeLength(ValueInstance instance, string memberName,
		out ValueInstance result)
	{
		if (memberName is "Length" or "Count")
		{
			if (instance.IsText)
			{
				result = new ValueInstance(executable.numberType, instance.Text.Length);
				return true;
			}
			if (instance.IsList)
			{
				result = new ValueInstance(executable.numberType, instance.List.Count);
				return true;
			}
			if (IsFileInstance(instance))
			{
				result = new ValueInstance(executable.numberType,
					NativeFileRegistry.Length(GetFileHandle(instance)));
				return true;
			}
		}
		result = default;
		return false;
	}

	internal static ValueInstance ConvertToText(ValueInstance rawValue)
	{
		if (rawValue.IsText)
			return rawValue;
		if (rawValue.TryGetValueTypeInstance() is { } typeInstance)
			return TryGetSingleTextMemberValue(typeInstance, out var value)
				? value
				: new ValueInstance(typeInstance.ToAutomaticText());
		return new ValueInstance(rawValue.ToExpressionCodeString());
	}

	private static bool TryGetSingleTextMemberValue(ValueTypeInstance typeInstance,
		out ValueInstance textValue)
	{
		var found = false;
		textValue = default;
		var members = typeInstance.ReturnType.Members;
		for (var memberIndex = 0; memberIndex < members.Count &&
			memberIndex < typeInstance.Values.Length; memberIndex++)
			if (!members[memberIndex].IsConstant)
			{
				var memberValue = typeInstance.Values[memberIndex];
				if (!memberValue.IsText || found)
					return false;
				textValue = memberValue;
				found = true;
			}
		return found;
	}
}
