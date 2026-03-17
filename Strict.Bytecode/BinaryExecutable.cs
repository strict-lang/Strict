using System.IO.Compression;
using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Bytecode;

/// <summary>
/// Loads <see cref="Instruction" /> bytecode for each type used with each method used. Generated
/// from <see cref="BinaryGenerator"/> or loaded from a compact .strictbinary ZIP file, which is
/// done via <see cref="Serialize(string)"/>. Used by the VirtualMachine or executable generation.
/// </summary>
public sealed class BinaryExecutable(Package basePackage)
{
	//TODO: remove: var package = new Package(basePackage,
	//	Path.GetFileNameWithoutExtension(FilePath) + "-" + ++packageCounter);
	internal readonly Package basePackage = basePackage;
	private readonly Package package = basePackage;
	internal Type noneType = basePackage.GetType(Type.None);
	internal Type booleanType = basePackage.GetType(Type.Boolean);
	internal Type numberType = basePackage.GetType(Type.Number);
	internal Type characterType = basePackage.GetType(Type.Character);
	internal Type rangeType = basePackage.GetType(Type.Range);
	internal Type listType = basePackage.GetType(Type.List);

	/// <summary>
	/// Reads a .strictbinary ZIP containing all type bytecode (used types, members, methods) and
	/// instruction bodies for each type.
	/// </summary>
	public BinaryExecutable(string filePath, Package basePackage) : this(basePackage)
	{
		try
		{
			using var zip = ZipFile.OpenRead(filePath);
			foreach (var entry in zip.Entries)
				if (entry.FullName.EndsWith(BinaryType.BytecodeEntryExtension,
					StringComparison.OrdinalIgnoreCase))
				{
					var typeFullName = GetEntryNameWithoutExtension(entry.FullName);
					using var bytecode = entry.Open();
					MethodsPerType.Add(typeFullName,
						new BinaryType(new BinaryReader(bytecode), this, typeFullName));
				}
		}
		catch (InvalidDataException ex)
		{
			throw new InvalidFile(ex.Message);
		}
	}

	private static string GetEntryNameWithoutExtension(string fullName)
	{
		var normalized = fullName.Replace('\\', '/');
		var extensionStart = normalized.LastIndexOf('.');
		return extensionStart > 0
			? normalized[..extensionStart]
			: normalized;
	}

	/// <summary>
	/// Each key is a type.FullName (e.g. Strict/Number, Strict/ImageProcessing/Color), the Value
	/// contains all members of this type and all not stripped out methods that were actually used.
	/// </summary>
	public Dictionary<string, BinaryType> MethodsPerType = new();
	public sealed class InvalidFile(string message) : Exception(message);

	/// <summary>
	/// Writes optimized <see cref="Instruction" /> lists per type into a compact .strictbinary ZIP.
	/// The ZIP contains one entry per type named {typeName}.bytecode.
	/// Entry layout: magic(6) + version(1) + string-table + instruction-count(7bit) + instructions.
	/// </summary>
	public void Serialize(string filePath)
	{
		using var fileStream = new FileStream(filePath, FileMode.Create, FileAccess.Write);
		using var zip = new ZipArchive(fileStream, ZipArchiveMode.Create, leaveOpen: false);
		foreach (var (fullTypeName, membersAndMethods) in MethodsPerType)
		{
			var entry = zip.CreateEntry(fullTypeName + BinaryType.BytecodeEntryExtension,
				CompressionLevel.Optimal);
			using var entryStream = entry.Open();
			using var writer = new BinaryWriter(entryStream);
			membersAndMethods.Write(writer);
		}
	}

	public const string Extension = ".strictbinary";

	public IReadOnlyList<Instruction>? FindInstructions(Type type, Method method) =>
		FindInstructions(type.FullName, method.Name, method.Parameters.Count, method.ReturnType.Name);

	public IReadOnlyList<Instruction>? FindInstructions(string fullTypeName, string methodName,
		int parametersCount, string returnType = "") =>
		MethodsPerType.TryGetValue(fullTypeName, out var methods)
			? methods.MethodGroups.GetValueOrDefault(methodName)?.Find(m =>
				m.Parameters.Count == parametersCount && m.ReturnTypeName == returnType)?.Instructions
			: null;

	public Instruction ReadInstruction(BinaryReader reader, NameTable table)
	{
		var type = (InstructionType)reader.ReadByte();
		return type switch
		{
			InstructionType.LoadConstantToRegister => new LoadConstantInstruction(reader, table, this),
			InstructionType.LoadVariableToRegister => new LoadVariableToRegister(reader, table),
			InstructionType.StoreConstantToVariable => new StoreVariableInstruction(reader, table, this),
			InstructionType.StoreRegisterToVariable => new StoreFromRegisterInstruction(reader, table),
			InstructionType.Set => new SetInstruction(reader, table, this),
			InstructionType.Invoke => new Invoke(reader, table, this),
			InstructionType.Return => new ReturnInstruction(reader),
			InstructionType.LoopBegin => new LoopBeginInstruction(reader),
			InstructionType.LoopEnd => new LoopEndInstruction(reader),
			InstructionType.JumpIfNotZero => new JumpIfNotZero(reader),
			InstructionType.JumpIfTrue => new Jump(reader, InstructionType.JumpIfTrue),
			InstructionType.JumpIfFalse => new Jump(reader, InstructionType.JumpIfFalse),
			InstructionType.JumpEnd => new JumpToId(reader, InstructionType.JumpEnd),
			InstructionType.JumpToIdIfFalse => new JumpToId(reader, InstructionType.JumpToIdIfFalse),
			InstructionType.JumpToIdIfTrue => new JumpToId(reader, InstructionType.JumpToIdIfTrue),
			InstructionType.Jump => new Jump(reader, InstructionType.Jump),
			InstructionType.InvokeWriteToList => new WriteToListInstruction(reader, table),
			InstructionType.InvokeWriteToTable => new WriteToTableInstruction(reader, table),
			InstructionType.InvokeRemove => new RemoveInstruction(reader, table),
			InstructionType.ListCall => new ListCallInstruction(reader, table),
			InstructionType.Print => new PrintInstruction(reader, table),
			_ when IsBinaryOp(type) => new BinaryInstruction(reader, type),
			_ => throw new InvalidFile("Unknown instruction type: " + type) //ncrunch: no coverage
		};
	}

	private static bool IsBinaryOp(InstructionType type) =>
		type is > InstructionType.StoreSeparator and < InstructionType.BinaryOperatorsSeparator;

	internal static void WriteValueInstance(BinaryWriter writer, ValueInstance val, NameTable table)
	{
		if (val.IsText)
		{
			writer.Write((byte)ValueKind.Text);
			writer.Write7BitEncodedInt(table[val.Text]);
			return;
		}
		if (val.IsList)
		{
			writer.Write((byte)ValueKind.List);
			writer.Write7BitEncodedInt(table[val.List.ReturnType.Name]);
			var items = val.List.Items;
			writer.Write7BitEncodedInt(items.Count);
			foreach (var item in items)
				WriteValueInstance(writer, item, table);
			return;
		}
		if (val.IsDictionary)
		{
			writer.Write((byte)ValueKind.Dictionary);
			writer.Write7BitEncodedInt(table[val.GetType().Name]);
			var items = val.GetDictionaryItems();
			writer.Write7BitEncodedInt(items.Count);
			foreach (var kvp in items)
			{
				WriteValueInstance(writer, kvp.Key, table);
				WriteValueInstance(writer, kvp.Value, table);
			}
			return;
		}
		var type = val.GetType();
		if (type.IsBoolean)
		{
			writer.Write((byte)ValueKind.Boolean);
			writer.Write(val.Boolean);
		}
		else if (type.IsNone)
			writer.Write((byte)ValueKind.None);
		else if (type.IsNumber)
		{
			if (IsSmallNumber(val.Number))
			{
				writer.Write((byte)ValueKind.SmallNumber);
				writer.Write((byte)(int)val.Number);
			}
			else if (IsIntegerNumber(val.Number))
			{
				writer.Write((byte)ValueKind.IntegerNumber);
				writer.Write((int)val.Number);
			}
			else
			{
				writer.Write((byte)ValueKind.Number);
				writer.Write(val.Number);
			}
		}
		else
			throw new ValueInstanceNotSupported(val); //ncrunch: no coverage
	}

	public static bool IsSmallNumber(double value) =>
		value is >= 0 and <= 255 && value == Math.Floor(value);

	public static bool IsIntegerNumber(double value) =>
		value is >= int.MinValue and <= int.MaxValue && value == Math.Floor(value);

	public class ValueInstanceNotSupported(ValueInstance instance) : Exception(instance.ToString());

	internal ValueInstance ReadValueInstance(BinaryReader reader, NameTable table)
	{
		var kind = (ValueKind)reader.ReadByte();
		return kind switch
		{
			ValueKind.Text => new ValueInstance(table.Names[reader.Read7BitEncodedInt()]),
			ValueKind.None => new ValueInstance(noneType),
			ValueKind.Boolean => new ValueInstance(booleanType, reader.ReadBoolean()),
			ValueKind.SmallNumber => new ValueInstance(numberType, reader.ReadByte()),
			ValueKind.IntegerNumber => new ValueInstance(numberType, reader.ReadInt32()),
			ValueKind.Number => new ValueInstance(numberType, reader.ReadDouble()),
			ValueKind.List => ReadListValueInstance(reader, table),
			ValueKind.Dictionary => ReadDictionaryValueInstance(reader, table),
			_ => throw new InvalidFile("Unknown ValueKind: " + kind)
		};
	}

	private ValueInstance ReadListValueInstance(BinaryReader reader, NameTable table)
	{
		var typeName = table.Names[reader.Read7BitEncodedInt()];
		var count = reader.Read7BitEncodedInt();
		var items = new ValueInstance[count];
		for (var index = 0; index < count; index++)
			items[index] = ReadValueInstance(reader, table);
		return new ValueInstance(basePackage.GetType(typeName), items);
	}

	private ValueInstance ReadDictionaryValueInstance(BinaryReader reader, NameTable table)
	{
		var typeName = table.Names[reader.Read7BitEncodedInt()];
		var count = reader.Read7BitEncodedInt();
		var items = new Dictionary<ValueInstance, ValueInstance>(count);
		for (var index = 0; index < count; index++)
		{
			var key = ReadValueInstance(reader, table);
			var value = ReadValueInstance(reader, table);
			items[key] = value;
		}
		return new ValueInstance(basePackage.GetType(typeName), items);
	}

	internal MethodCall ReadMethodCall(BinaryReader reader, NameTable table)
	{
		var declaringTypeName = table.Names[reader.Read7BitEncodedInt()];
		var methodName = table.Names[reader.Read7BitEncodedInt()];
		var paramCount = reader.Read7BitEncodedInt();
		var parameters = new BinaryMember[paramCount];
		for (var index = 0; index < paramCount; index++)
			parameters[index] = new BinaryMember(reader, table, this);
		var returnTypeName = table.Names[reader.Read7BitEncodedInt()];
		var hasInstance = reader.ReadBoolean();
		var instance = hasInstance
			? ReadExpression(reader, table)
			: null;
		var argCount = reader.Read7BitEncodedInt();
		var args = new Expression[argCount];
		for (var index = 0; index < argCount; index++)
			args[index] = ReadExpression(reader, table);
		var declaringType = EnsureResolvedType(package, declaringTypeName);
		var returnType = EnsureResolvedType(package, returnTypeName);
		var method = FindMethod(declaringType, methodName, parameters, returnType);
		var methodReturnType = returnType != method.ReturnType
			? returnType
			: null;
		return new MethodCall(method, instance, args, methodReturnType);
	}

	private Method FindMethod(Type type, string methodName, IReadOnlyList<BinaryMember> parameters,
		Type returnType)
	{
		var method = type.Methods.FirstOrDefault(existingMethod =>
				existingMethod.Name == methodName && existingMethod.Parameters.Count == parameters.Count) ??
			type.Methods.FirstOrDefault(existingMethod => existingMethod.Name == methodName);
		if (method != null)
			return method;
		if (type.AvailableMethods.TryGetValue(methodName, out var availableMethods))
		{
			var found = availableMethods.FirstOrDefault(existingMethod =>
				existingMethod.Parameters.Count == parameters.Count) ?? availableMethods.FirstOrDefault();
			if (found != null)
				return found;
		} //ncrunch: no coverage
		var methodHeader = BuildMethodHeader(methodName, parameters, returnType);
		var createdMethod = new Method(type, 0, new MethodExpressionParser(), [methodHeader]);
		type.Methods.Add(createdMethod);
		return createdMethod;
	}

	public static string BuildMethodHeader(string methodName,
		IReadOnlyList<BinaryMember> parameters, Type returnType) =>
		parameters.Count == 0
			? returnType.IsNone
				? methodName
				: methodName + " " + returnType.Name
			: methodName + "(" + string.Join(", ", parameters) + ") " + returnType.Name;

	internal Expression ReadExpression(BinaryReader reader, NameTable table)
	{
		var kind = (ExpressionKind)reader.ReadByte();
		return kind switch
		{
			ExpressionKind.SmallNumberValue => new Number(package, reader.ReadByte()),
			ExpressionKind.IntegerNumberValue => new Number(package, reader.ReadInt32()),
			ExpressionKind.NumberValue => new Number(package, reader.ReadDouble()),
			ExpressionKind.TextValue => new Text(package, table.Names[reader.Read7BitEncodedInt()]),
			ExpressionKind.BooleanValue => ReadBooleanValue(reader, package, table),
			ExpressionKind.VariableRef => ReadVariableRef(reader, package, table),
			ExpressionKind.MemberRef => ReadMemberRef(reader, package, table),
			ExpressionKind.BinaryExpr => ReadBinaryExpr(reader, package, table),
			ExpressionKind.MethodCallExpr => ReadMethodCall(reader, table),
			_ => throw new InvalidFile("Unknown ExpressionKind: " + kind)
		};
	}

	private static Value ReadBooleanValue(BinaryReader reader, Package package, NameTable table)
	{
		var type = EnsureResolvedType(package, table.Names[reader.Read7BitEncodedInt()]);
		return new Value(type, new ValueInstance(type, reader.ReadBoolean()));
	}

	//TODO: avoid!
	private static Type EnsureResolvedType(Package package, string typeName)
	{
		var resolved = package.FindType(typeName) ?? (typeName.Contains('.')
			? package.FindFullType(typeName)
			: null);
		if (resolved != null)
			return resolved;
		if (typeName.EndsWith(')') && typeName.Contains('('))
			return package.GetType(typeName); //ncrunch: no coverage
		if (char.IsLower(typeName[0]))
			throw new TypeNotFoundForBytecode(typeName);
		EnsureTypeExists(package, typeName);
		return package.GetType(typeName);
	}

	public sealed class TypeNotFoundForBytecode(string typeName)
		: Exception("Type '" + typeName + "' not found while deserializing bytecode");

	private static void EnsureTypeExists(Package package, string typeName)
	{
		if (package.FindDirectType(typeName) == null)
			new Type(package, new TypeLines(typeName, Method.Run)).ParseMembersAndMethods(
				new MethodExpressionParser());
	}

	private static Expression ReadVariableRef(BinaryReader reader, Package package, NameTable table)
	{
		var name = table.Names[reader.Read7BitEncodedInt()];
		var type = EnsureResolvedType(package, table.Names[reader.Read7BitEncodedInt()]);
		var param = new Parameter(type, name, new Value(type, new ValueInstance(type)));
		return new ParameterCall(param);
	}

	private MemberCall ReadMemberRef(BinaryReader reader, Package package, NameTable table)
	{
		var memberName = table.Names[reader.Read7BitEncodedInt()];
		var memberTypeName = table.Names[reader.Read7BitEncodedInt()];
		var hasInstance = reader.ReadBoolean();
		var instance = hasInstance
			? ReadExpression(reader, table)
			: null;
		var anyBaseType = EnsureResolvedType(package, Type.Number);
		var fakeMember = new Member(anyBaseType, memberName + " " + memberTypeName, null);
		return new MemberCall(instance, fakeMember);
	}

	private Binary ReadBinaryExpr(BinaryReader reader, Package package, NameTable table)
	{
		var operatorName = table.Names[reader.Read7BitEncodedInt()];
		var left = ReadExpression(reader, table);
		var right = ReadExpression(reader, table);
		var operatorMethod = FindOperatorMethod(operatorName, left.ReturnType);
		return new Binary(left, operatorMethod, [right]);
	}

	private static Method FindOperatorMethod(string operatorName, Type preferredType) =>
		preferredType.Methods.FirstOrDefault(m => m.Name == operatorName) ?? throw new
			MethodNotFoundException(operatorName);

	public sealed class MethodNotFoundException(string methodName)
		: Exception($"Method '{methodName}' not found");

	internal static void WriteExpression(BinaryWriter writer, Expression expr, NameTable table)
	{
		switch (expr)
		{
		case Value { Data.IsText: true } val:
			writer.Write((byte)ExpressionKind.TextValue);
			writer.Write7BitEncodedInt(table[val.Data.Text]);
			break;
		case Value val when val.Data.GetType().IsBoolean:
			writer.Write((byte)ExpressionKind.BooleanValue);
			writer.Write7BitEncodedInt(table[val.Data.GetType().Name]);
			writer.Write(val.Data.Boolean);
			break;
		case Value val when val.Data.GetType().IsNumber:
			if (IsSmallNumber(val.Data.Number))
			{
				writer.Write((byte)ExpressionKind.SmallNumberValue);
				writer.Write((byte)(int)val.Data.Number);
			}
			else if (IsIntegerNumber(val.Data.Number))
			{
				writer.Write((byte)ExpressionKind.IntegerNumberValue);
				writer.Write((int)val.Data.Number);
			}
			else
			{
				writer.Write((byte)ExpressionKind.NumberValue);
				writer.Write(val.Data.Number);
			}
			break;
		case Value val:
			throw new NotSupportedException("WriteExpression not supported value: " + val);
		case MemberCall memberCall:
			writer.Write((byte)ExpressionKind.MemberRef);
			writer.Write7BitEncodedInt(table[memberCall.Member.Name]);
			writer.Write7BitEncodedInt(table[memberCall.Member.Type.Name]);
			writer.Write(memberCall.Instance != null);
			if (memberCall.Instance != null)
				// ReSharper disable TailRecursiveCall
				WriteExpression(writer, memberCall.Instance, table);
			break;
		case Expressions.Binary binary:
			writer.Write((byte)ExpressionKind.BinaryExpr);
			writer.Write7BitEncodedInt(table[binary.Method.Name]);
			WriteExpression(writer, binary.Instance!, table);
			WriteExpression(writer, binary.Arguments[0], table);
			break;
		case MethodCall methodCall:
			writer.Write((byte)ExpressionKind.MethodCallExpr);
			writer.Write7BitEncodedInt(table[methodCall.Method.Type.Name]);
			writer.Write7BitEncodedInt(table[methodCall.Method.Name]);
			writer.Write7BitEncodedInt(methodCall.Method.Parameters.Count);
			writer.Write7BitEncodedInt(table[methodCall.ReturnType.Name]);
			writer.Write(methodCall.Instance != null);
			if (methodCall.Instance != null)
				WriteExpression(writer, methodCall.Instance, table);
			writer.Write7BitEncodedInt(methodCall.Arguments.Count);
			foreach (var argument in methodCall.Arguments)
				WriteExpression(writer, argument, table);
			break;
		default:
			writer.Write((byte)ExpressionKind.VariableRef);
			writer.Write7BitEncodedInt(table[expr.ToString()]);
			writer.Write7BitEncodedInt(table[expr.ReturnType.Name]);
			break;
		}
	}

	internal static void WriteMethodCallData(BinaryWriter writer, MethodCall? methodCall,
		Registry? registry, NameTable table)
	{
		writer.Write(methodCall != null);
		if (methodCall != null)
		{
			writer.Write7BitEncodedInt(table[methodCall.Method.Type.Name]);
			writer.Write7BitEncodedInt(table[methodCall.Method.Name]);
			writer.Write7BitEncodedInt(methodCall.Method.Parameters.Count);
			writer.Write7BitEncodedInt(table[methodCall.ReturnType.Name]);
			writer.Write(methodCall.Instance != null);
			if (methodCall.Instance != null)
				WriteExpression(writer, methodCall.Instance, table);
			writer.Write7BitEncodedInt(methodCall.Arguments.Count);
			foreach (var argument in methodCall.Arguments)
				WriteExpression(writer, argument, table);
		}
		writer.Write(registry != null);
		if (registry == null)
			return;
		writer.Write((byte)registry.NextRegister);
		writer.Write((byte)registry.PreviousRegister);
	}

	public bool UsesConsolePrint => MethodsPerType.Values.Any(type => type.UsesConsolePrint);
}