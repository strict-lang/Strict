using System.Runtime.CompilerServices;
using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

[assembly: InternalsVisibleTo("Strict")]
[assembly: InternalsVisibleTo("Strict.Optimizers")]

namespace Strict.Bytecode;

public sealed partial class BinaryExecutable
{
	public Instruction ReadInstruction(BinaryReader reader, NameTable table)
	{
		var prevSourceLine = 0;
		return ReadInstruction(reader, table, ref prevSourceLine);
	}

	internal Instruction ReadInstruction(BinaryReader reader, NameTable table, ref int prevSourceLine)
	{
		var rawByte = reader.ReadByte();
		var hasSourceLine = (rawByte & (byte)InstructionType.IncludesSourceLine) != 0;
		var type = (InstructionType)(rawByte & ((byte)InstructionType.IncludesSourceLine - 1));
		if (hasSourceLine)
			prevSourceLine = reader.Read7BitEncodedInt(); // 20-40% of instructions have source lines
		Instruction instruction = type switch
		{
			InstructionType.LoadConstantToRegister => new LoadConstantInstruction(reader, table, this),
			InstructionType.LoadVariableToRegister => new LoadVariableToRegister(reader, table),
			InstructionType.StoreConstantToVariable => new StoreVariableInstruction(reader, table, this),
			InstructionType.StoreRegisterToVariable => new StoreFromRegisterInstruction(reader, table),
			InstructionType.Set => new SetInstruction(reader, table, this),
			InstructionType.Invoke => new Invoke(reader, table),
			InstructionType.Return => new ReturnInstruction(reader),
			InstructionType.LoopBegin => new LoopBeginInstruction(reader, table),
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
			InstructionType.ConstructValueType => new ConstructValueTypeInstruction(reader, table, this),
			InstructionType.FieldLoad => new FieldLoadInstruction(reader, table),
			InstructionType.CopyList => new CopyListInstruction(reader, table),
			_ when IsBinaryOp(type) => new BinaryInstruction(reader, type),
			_ => throw new InvalidFile("Unknown instruction type: " + type) //ncrunch: no coverage
		};
		instruction.SourceLine = prevSourceLine;
		return instruction;
	}

	private static bool IsBinaryOp(InstructionType type) =>
		type is > InstructionType.StoreSeparator and < InstructionType.BinaryOperatorsSeparator;

	internal ValueInstance ReadValueInstance(BinaryReader reader, NameTable table)
	{
		var kind = (ValueKind)reader.ReadByte();
		return kind switch
		{
			ValueKind.Text => new ValueInstance(table.names[reader.Read7BitEncodedInt()]),
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
		var typeName = table.names[reader.Read7BitEncodedInt()];
		var count = reader.Read7BitEncodedInt();
		var items = new ValueInstance[count];
		for (var index = 0; index < count; index++)
			items[index] = ReadValueInstance(reader, table);
		return new ValueInstance(ResolveType(typeName), items);
	}

	private ValueInstance ReadDictionaryValueInstance(BinaryReader reader, NameTable table)
	{
		var typeName = table.names[reader.Read7BitEncodedInt()];
		var count = reader.Read7BitEncodedInt();
		var items = new Dictionary<ValueInstance, ValueInstance>(count);
		for (var index = 0; index < count; index++)
		{
			var key = ReadValueInstance(reader, table);
			var value = ReadValueInstance(reader, table);
			items[key] = value;
		}
		return new ValueInstance(ResolveType(typeName), items);
	}

	internal MethodCall ReadMethodCall(BinaryReader reader, NameTable table)
	{
		var declaringTypeName = table.names[reader.Read7BitEncodedInt()];
		var methodName = table.names[reader.Read7BitEncodedInt()];
		var paramCount = reader.Read7BitEncodedInt();
		var parameters = paramCount == 0
			? Array.Empty<BinaryMember>()
			: new BinaryMember[paramCount];
		for (var index = 0; index < paramCount; index++)
			parameters[index] = new BinaryMember(table.names[reader.Read7BitEncodedInt()],
				table.names[reader.Read7BitEncodedInt()], null);
		var returnTypeName = table.names[reader.Read7BitEncodedInt()];
		var hasInstance = reader.ReadBoolean();
		var instance = hasInstance
			? ReadExpression(reader, table)
			: null;
		var argCount = reader.Read7BitEncodedInt();
		var args = argCount == 0
			? Array.Empty<Expression>()
			: new Expression[argCount];
		for (var index = 0; index < argCount; index++)
			args[index] = ReadExpression(reader, table);
		var declaringType = ResolveType(declaringTypeName);
		var returnType = ResolveType(returnTypeName);
		var method = FindMethod(declaringType, methodName, parameters, returnType);
		var methodReturnType = returnType != method.ReturnType
			? returnType
			: null;
		return new MethodCall(method, instance, args, methodReturnType);
	}

	private static Method FindMethod(Type type, string methodName, BinaryMember[] parameters,
		Type returnType)
	{
		var method = type.Methods.FirstOrDefault(existingMethod =>
			existingMethod.Name == methodName && existingMethod.Parameters.Count == parameters.Length);
		if (method != null)
			return method;
		if (type.AvailableMethods.TryGetValue(methodName, out var availableMethods))
		{
			var found = availableMethods.FirstOrDefault(existingMethod =>
				existingMethod.Parameters.Count == parameters.Length);
			if (found != null)
				return found;
			if (parameters.Length == 0)
			{
				var noParameterFallback = availableMethods.FirstOrDefault();
				if (noParameterFallback != null)
					return noParameterFallback;
			}
		} //ncrunch: no coverage
		if (parameters.Length == 0)
		{
			var noParameterMethod = type.Methods.FirstOrDefault(existingMethod =>
				existingMethod.Name == methodName);
			if (noParameterMethod != null)
				return noParameterMethod;
		}
		var methodHeader = BuildMethodHeader(methodName, parameters, returnType);
		var createdMethod = new Method(type, 0, new MethodExpressionParser(), [methodHeader]);
		type.Methods.Add(createdMethod);
		return createdMethod;
	}

	public static string BuildMethodHeader(string methodName, BinaryMember[] parameters,
		Type returnType) =>
		parameters.Length == 0
			? returnType.IsNone
				? methodName
				: methodName + " " + returnType.Name
			: methodName + "(" + string.Join(", ", parameters) + ") " + returnType.Name;

	internal Expression ReadExpression(BinaryReader reader, NameTable table)
	{
		var kind = (ExpressionKind)reader.ReadByte();
		return kind switch
		{
			ExpressionKind.SmallNumberValue => new Number(basePackage, reader.ReadByte()),
			ExpressionKind.IntegerNumberValue => new Number(basePackage, reader.ReadInt32()),
			ExpressionKind.NumberValue => new Number(basePackage, reader.ReadDouble()),
			ExpressionKind.TextValue => new Text(basePackage, table.names[reader.Read7BitEncodedInt()]),
			ExpressionKind.BooleanValue => ReadBooleanValue(reader, table),
			ExpressionKind.VariableRef => ReadVariableRef(reader, table),
			ExpressionKind.MemberRef => ReadMemberRef(reader, table),
			ExpressionKind.BinaryExpr => ReadBinaryExpr(reader, table),
			ExpressionKind.MethodCallExpr => ReadMethodCall(reader, table),
			ExpressionKind.ListExpr => ReadListExpr(reader, table),
			ExpressionKind.ListCallExpr => ReadListCallExpr(reader, table),
			_ => throw new InvalidFile("Unknown ExpressionKind: " + kind)
		};
	}

	private List ReadListExpr(BinaryReader reader, NameTable table)
	{
		var concreteListType =
			ResolveType(table.names[reader.Read7BitEncodedInt()]);
		var itemCount = reader.Read7BitEncodedInt();
		var values = new List<Expression>(itemCount);
		for (var index = 0; index < itemCount; index++)
			values.Add(ReadExpression(reader, table));
		return new List(concreteListType, values, 0, false);
	}

	private ListCall ReadListCallExpr(BinaryReader reader, NameTable table)
	{
		ResolveType(table.names[reader.Read7BitEncodedInt()]);
		var list = ReadExpression(reader, table);
		var index = ReadExpression(reader, table);
		var hasSecondIndex = reader.ReadBoolean();
		var secondIndex = hasSecondIndex
			? ReadExpression(reader, table)
			: null;
		return new ListCall(list, index, secondIndex);
	}

	//TODO: missing test
	private Value ReadBooleanValue(BinaryReader reader, NameTable table)
	{
		var type = ResolveType(table.names[reader.Read7BitEncodedInt()]);
		return new Value(type, new ValueInstance(type, reader.ReadBoolean()));
	}

	private Expression ReadVariableRef(BinaryReader reader, NameTable table)
	{
		var name = table.names[reader.Read7BitEncodedInt()];
		var type = ResolveType(table.names[reader.Read7BitEncodedInt()]);
		var parenIndex = name.IndexOf('(');
		var cleanName = parenIndex > 0
			? name[..parenIndex]
			: name;
		var dotIndex = cleanName.IndexOf('.');
		if (dotIndex > 0)
			cleanName = cleanName[..dotIndex];
		if (!cachedDefaultValuesForVariableRefs.TryGetValue(type, out var defaultValue))
		{
			defaultValue = new Value(type, new ValueInstance(type));
			cachedDefaultValuesForVariableRefs[type] = defaultValue;
		}
		var param = new Parameter(type, cleanName, defaultValue);
		return new ParameterCall(param);
	}

	private MemberCall ReadMemberRef(BinaryReader reader, NameTable table)
	{
		var memberName = table.names[reader.Read7BitEncodedInt()];
		var memberTypeName = table.names[reader.Read7BitEncodedInt()];
		var hasInstance = reader.ReadBoolean();
		var instance = hasInstance
			? ReadExpression(reader, table)
			: null;
		var anyBaseType = ResolveType(Type.Number);
		var memberType = ResolveType(memberTypeName);
		var fakeMember = new Member(anyBaseType, memberName, memberType);
		return new MemberCall(instance, fakeMember);
	}

	private Binary ReadBinaryExpr(BinaryReader reader, NameTable table)
	{
		var operatorName = table.names[reader.Read7BitEncodedInt()];
		var left = ReadExpression(reader, table);
		var right = ReadExpression(reader, table);
		var operatorMethod = FindOperatorMethod(operatorName, left.ReturnType);
		return new Binary(left, operatorMethod, [right]);
	}

	private static Method FindOperatorMethod(string operatorName, Type preferredType) =>
		preferredType.Methods.FirstOrDefault(m => m.Name == operatorName) ??
		throw new MethodNotFoundException(operatorName);

	public sealed class MethodNotFoundException(string methodName)
		: Exception($"Method '{methodName}' not found");
}
