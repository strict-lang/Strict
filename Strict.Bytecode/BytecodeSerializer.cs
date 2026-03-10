using Strict.Bytecode.Instructions;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Bytecode;

/// <summary>
/// Saves optimized <see cref="Instruction"/> lists to a compact binary .sbc (Strict ByteCode) file
/// and restores them by resolving type and method references from a <see cref="Package"/>.
/// File layout: magic(4) + version(1) + source-path + instruction-count(4) + instructions.
/// </summary>
public static class BytecodeSerializer
{
	private static readonly byte[] MagicBytes = [0x73, 0x62, 0x63, 0x01];
	public const string Extension = ".sbc";

	public static void Serialize(IList<Instruction> instructions, string outputPath,
		string sourceFilePath)
	{
		using var writer = new BinaryWriter(File.Create(outputPath));
		writer.Write(MagicBytes);
		writer.Write(sourceFilePath);
		writer.Write(instructions.Count);
		foreach (var instruction in instructions)
			WriteInstruction(writer, instruction);
	}

	public static (List<Instruction> Instructions, string SourceFilePath) Deserialize(
		string sbcFilePath, Package package)
	{
		using var reader = new BinaryReader(File.OpenRead(sbcFilePath));
		ValidateMagic(reader);
		var sourcePath = reader.ReadString();
		var count = reader.ReadInt32();
		var instructions = new List<Instruction>(count);
		for (var i = 0; i < count; i++)
			instructions.Add(ReadInstruction(reader, package));
		return (instructions, sourcePath);
	}

	/// <summary>
	/// Reads only the embedded source file path from the .sbc header without deserializing
	/// the instruction payload.
	/// </summary>
	public static string ReadSourcePath(string sbcFilePath)
	{
		using var reader = new BinaryReader(File.OpenRead(sbcFilePath));
		ValidateMagic(reader);
		return reader.ReadString();
	}

	private static void ValidateMagic(BinaryReader reader)
	{
		var magic = reader.ReadBytes(4);
		if (!magic.SequenceEqual(MagicBytes))
			throw new InvalidBytecodeFileException();
	}

	public sealed class InvalidBytecodeFileException : Exception
	{
		public InvalidBytecodeFileException() : base("Not a valid Strict bytecode (.sbc) file") { }
	}

	private enum InstructionClass : byte
	{
		LoadConstant,
		LoadVariable,
		StoreVariable,
		StoreFromRegister,
		Set,
		Binary,
		Invoke,
		Return,
		LoopBegin,
		LoopEnd,
		Jump,
		JumpIf,
		JumpIfNotZero,
		JumpToId,
		WriteToList,
		WriteToTable,
		Remove,
		ListCall
	}

	private static void WriteInstruction(BinaryWriter w, Instruction inst)
	{
		switch (inst)
		{
		case LoadConstantInstruction loadConst:
			w.Write((byte)InstructionClass.LoadConstant);
			w.Write((byte)loadConst.Register);
			WriteValueInstance(w, loadConst.ValueInstance);
			break;
		case LoadVariableToRegister loadVar:
			w.Write((byte)InstructionClass.LoadVariable);
			w.Write((byte)loadVar.Register);
			w.Write(loadVar.Identifier);
			break;
		case StoreVariableInstruction storeVar:
			w.Write((byte)InstructionClass.StoreVariable);
			WriteValueInstance(w, storeVar.ValueInstance);
			w.Write(storeVar.Identifier);
			w.Write(storeVar.IsMember);
			break;
		case StoreFromRegisterInstruction storeReg:
			w.Write((byte)InstructionClass.StoreFromRegister);
			w.Write((byte)storeReg.Register);
			w.Write(storeReg.Identifier);
			break;
		case SetInstruction set:
			w.Write((byte)InstructionClass.Set);
			WriteValueInstance(w, set.ValueInstance);
			w.Write((byte)set.Register);
			break;
		case BinaryInstruction binary:
			w.Write((byte)InstructionClass.Binary);
			w.Write((short)binary.InstructionType);
			w.Write((byte)binary.Registers.Length);
			foreach (var reg in binary.Registers)
				w.Write((byte)reg);
			break;
		case Invoke invoke:
			w.Write((byte)InstructionClass.Invoke);
			w.Write((byte)invoke.Register);
			WriteMethodCallData(w, invoke.Method, invoke.PersistedRegistry);
			break;
		case ReturnInstruction ret:
			w.Write((byte)InstructionClass.Return);
			w.Write((byte)ret.Register);
			break;
		case LoopBeginInstruction loopBegin:
			w.Write((byte)InstructionClass.LoopBegin);
			w.Write((byte)loopBegin.Register);
			w.Write(loopBegin.IsRange);
			if (loopBegin.IsRange)
				w.Write((byte)loopBegin.EndIndex!.Value);
			break;
		case LoopEndInstruction loopEnd:
			w.Write((byte)InstructionClass.LoopEnd);
			w.Write(loopEnd.Steps);
			break;
		case JumpIfNotZero jumpIfNotZero:
			w.Write((byte)InstructionClass.JumpIfNotZero);
			w.Write(jumpIfNotZero.Steps);
			w.Write((byte)jumpIfNotZero.Register);
			break;
		case JumpIf jumpIf:
			w.Write((byte)InstructionClass.JumpIf);
			w.Write((short)jumpIf.InstructionType);
			w.Write(jumpIf.Steps);
			break;
		case Jump jump:
			w.Write((byte)InstructionClass.Jump);
			w.Write((short)jump.InstructionType);
			w.Write(jump.InstructionsToSkip);
			break;
		case JumpToId jumpToId:
			w.Write((byte)InstructionClass.JumpToId);
			w.Write((short)jumpToId.InstructionType);
			w.Write(jumpToId.Id);
			break;
		case WriteToListInstruction writeToList:
			w.Write((byte)InstructionClass.WriteToList);
			w.Write((byte)writeToList.Register);
			w.Write(writeToList.Identifier);
			break;
		case WriteToTableInstruction writeToTable:
			w.Write((byte)InstructionClass.WriteToTable);
			w.Write((byte)writeToTable.Key);
			w.Write((byte)writeToTable.Value);
			w.Write(writeToTable.Identifier);
			break;
		case RemoveInstruction remove:
			w.Write((byte)InstructionClass.Remove);
			w.Write(remove.Identifier);
			w.Write((byte)remove.Register);
			break;
		case ListCallInstruction listCall:
			w.Write((byte)InstructionClass.ListCall);
			w.Write((byte)listCall.Register);
			w.Write((byte)listCall.IndexValueRegister);
			w.Write(listCall.Identifier);
			break;
		}
	}

	private enum ValueKind : byte
	{
		None,
		Number,
		Text,
		Boolean,
		List
	}

	private static void WriteValueInstance(BinaryWriter w, ValueInstance val)
	{
		if (val.IsText)
		{
			w.Write((byte)ValueKind.Text);
			w.Write(val.Text);
			return;
		}
		if (val.IsList)
		{
			w.Write((byte)ValueKind.List);
			w.Write(val.List.ReturnType.Name);
			var items = val.List.Items;
			w.Write(items.Length);
			foreach (var item in items)
				WriteValueInstance(w, item);
			return;
		}
		var type = val.GetTypeExceptText();
		if (type.IsBoolean)
		{
			w.Write((byte)ValueKind.Boolean);
			w.Write(type.Name);
			w.Write(val.Boolean);
			return;
		}
		if (type.IsNone)
		{
			w.Write((byte)ValueKind.None);
			w.Write(type.Name);
			return;
		}
		w.Write((byte)ValueKind.Number);
		w.Write(type.Name);
		w.Write(val.Number);
	}

	private enum ExpressionKind : byte
	{
		NumberValue,
		TextValue,
		BooleanValue,
		VariableRef,
		MemberRef,
		BinaryExpr,
		MethodCallExpr
	}

	private static void WriteExpression(BinaryWriter w, Expression expr)
	{
		switch (expr)
		{
		case Value val when val.Data.IsText:
			w.Write((byte)ExpressionKind.TextValue);
			w.Write(val.Data.Text);
			break;
		case Value val when val.Data.GetTypeExceptText().IsBoolean:
			w.Write((byte)ExpressionKind.BooleanValue);
			w.Write(val.Data.GetTypeExceptText().Name);
			w.Write(val.Data.Boolean);
			break;
		case Value val:
			w.Write((byte)ExpressionKind.NumberValue);
			w.Write(val.Data.Number);
			break;
		case MemberCall memberCall:
			w.Write((byte)ExpressionKind.MemberRef);
			w.Write(memberCall.Member.Name);
			w.Write(memberCall.Member.Type.Name);
			w.Write(memberCall.Instance != null);
			if (memberCall.Instance != null)
				WriteExpression(w, memberCall.Instance);
			break;
		case Binary binary:
			w.Write((byte)ExpressionKind.BinaryExpr);
			w.Write(binary.Method.Name);
			WriteExpression(w, binary.Instance!);
			WriteExpression(w, binary.Arguments[0]);
			break;
		case MethodCall methodCall:
			w.Write((byte)ExpressionKind.MethodCallExpr);
			w.Write(methodCall.Method.Type.Name);
			w.Write(methodCall.Method.Name);
			w.Write(methodCall.Method.Parameters.Count);
			w.Write(methodCall.ReturnType.Name);
			w.Write(methodCall.Instance != null);
			if (methodCall.Instance != null)
				WriteExpression(w, methodCall.Instance);
			w.Write(methodCall.Arguments.Count);
			foreach (var arg in methodCall.Arguments)
				WriteExpression(w, arg);
			break;
		default:
			w.Write((byte)ExpressionKind.VariableRef);
			w.Write(expr.ToString());
			w.Write(expr.ReturnType.Name);
			break;
		}
	}

	private static void WriteMethodCallData(BinaryWriter w, MethodCall? methodCall,
		Registry? registry)
	{
		w.Write(methodCall != null);
		if (methodCall != null)
		{
			w.Write(methodCall.Method.Type.Name);
			w.Write(methodCall.Method.Name);
			w.Write(methodCall.Method.Parameters.Count);
			w.Write(methodCall.ReturnType.Name);
			w.Write(methodCall.Instance != null);
			if (methodCall.Instance != null)
				WriteExpression(w, methodCall.Instance);
			w.Write(methodCall.Arguments.Count);
			foreach (var arg in methodCall.Arguments)
				WriteExpression(w, arg);
		}
		w.Write(registry != null);
		if (registry != null)
		{
			w.Write((byte)registry.NextRegister);
			w.Write((byte)registry.PreviousRegister);
		}
	}

	private static Instruction ReadInstruction(BinaryReader r, Package package)
	{
		var cls = (InstructionClass)r.ReadByte();
		return cls switch
		{
			InstructionClass.LoadConstant => ReadLoadConstant(r, package),
			InstructionClass.LoadVariable => ReadLoadVariable(r),
			InstructionClass.StoreVariable => ReadStoreVariable(r, package),
			InstructionClass.StoreFromRegister => ReadStoreFromRegister(r),
			InstructionClass.Set => ReadSet(r, package),
			InstructionClass.Binary => ReadBinary(r),
			InstructionClass.Invoke => ReadInvoke(r, package),
			InstructionClass.Return => new ReturnInstruction((Register)r.ReadByte()),
			InstructionClass.LoopBegin => ReadLoopBegin(r),
			InstructionClass.LoopEnd => new LoopEndInstruction(r.ReadInt32()),
			InstructionClass.Jump => ReadJump(r),
			InstructionClass.JumpIf => ReadJumpIf(r),
			InstructionClass.JumpIfNotZero => ReadJumpIfNotZero(r),
			InstructionClass.JumpToId => ReadJumpToId(r),
			InstructionClass.WriteToList => ReadWriteToList(r),
			InstructionClass.WriteToTable => ReadWriteToTable(r),
			InstructionClass.Remove => ReadRemove(r),
			InstructionClass.ListCall => ReadListCall(r),
			_ => throw new InvalidBytecodeFileException()
		};
	}

	private static LoadConstantInstruction ReadLoadConstant(BinaryReader r, Package package) =>
		new((Register)r.ReadByte(), ReadValueInstance(r, package));

	private static LoadVariableToRegister ReadLoadVariable(BinaryReader r) =>
		new((Register)r.ReadByte(), r.ReadString());

	private static StoreVariableInstruction ReadStoreVariable(BinaryReader r, Package package) =>
		new(ReadValueInstance(r, package), r.ReadString(), r.ReadBoolean());

	private static StoreFromRegisterInstruction ReadStoreFromRegister(BinaryReader r) =>
		new((Register)r.ReadByte(), r.ReadString());

	private static SetInstruction ReadSet(BinaryReader r, Package package) =>
		new(ReadValueInstance(r, package), (Register)r.ReadByte());

	private static BinaryInstruction ReadBinary(BinaryReader r)
	{
		var instructionType = (InstructionType)r.ReadInt16();
		var count = r.ReadByte();
		var registers = new Register[count];
		for (var i = 0; i < count; i++)
			registers[i] = (Register)r.ReadByte();
		return new BinaryInstruction(instructionType, registers);
	}

	private static Invoke ReadInvoke(BinaryReader r, Package package)
	{
		var register = (Register)r.ReadByte();
		var (methodCall, registry) = ReadMethodCallData(r, package);
		return new Invoke(register, methodCall!, registry!);
	}

	private static LoopBeginInstruction ReadLoopBegin(BinaryReader r)
	{
		var reg = (Register)r.ReadByte();
		var isRange = r.ReadBoolean();
		if (isRange)
		{
			var endReg = (Register)r.ReadByte();
			return new LoopBeginInstruction(reg, endReg);
		}
		return new LoopBeginInstruction(reg);
	}

	private static Jump ReadJump(BinaryReader r)
	{
		var instructionType = (InstructionType)r.ReadInt16();
		return new Jump(r.ReadInt32(), instructionType);
	}

	private static JumpIf ReadJumpIf(BinaryReader r)
	{
		var instructionType = (InstructionType)r.ReadInt16();
		var steps = r.ReadInt32();
		return new JumpIf(instructionType, steps);
	}

	private static JumpIfNotZero ReadJumpIfNotZero(BinaryReader r) =>
		new(r.ReadInt32(), (Register)r.ReadByte());

	private static JumpToId ReadJumpToId(BinaryReader r) =>
		new((InstructionType)r.ReadInt16(), r.ReadInt32());

	private static WriteToListInstruction ReadWriteToList(BinaryReader r) =>
		new((Register)r.ReadByte(), r.ReadString());

	private static WriteToTableInstruction ReadWriteToTable(BinaryReader r) =>
		new((Register)r.ReadByte(), (Register)r.ReadByte(), r.ReadString());

	private static RemoveInstruction ReadRemove(BinaryReader r) =>
		new(r.ReadString(), (Register)r.ReadByte());

	private static ListCallInstruction ReadListCall(BinaryReader r) =>
		new((Register)r.ReadByte(), (Register)r.ReadByte(), r.ReadString());

	private static ValueInstance ReadValueInstance(BinaryReader r, Package package)
	{
		var kind = (ValueKind)r.ReadByte();
		return kind switch
		{
			ValueKind.Text => new ValueInstance(r.ReadString()),
			ValueKind.None => new ValueInstance(package.GetType(r.ReadString())),
			ValueKind.Boolean => new ValueInstance(package.GetType(r.ReadString()),
				r.ReadBoolean()),
			ValueKind.Number => new ValueInstance(package.GetType(r.ReadString()), r.ReadDouble()),
			ValueKind.List => ReadListValueInstance(r, package),
			_ => throw new InvalidBytecodeFileException()
		};
	}

	private static ValueInstance ReadListValueInstance(BinaryReader r, Package package)
	{
		var typeName = r.ReadString();
		var count = r.ReadInt32();
		var items = new ValueInstance[count];
		for (var i = 0; i < count; i++)
			items[i] = ReadValueInstance(r, package);
		return new ValueInstance(package.GetType(typeName), items);
	}

	private static Expression ReadExpression(BinaryReader r, Package package)
	{
		var kind = (ExpressionKind)r.ReadByte();
		return kind switch
		{
			ExpressionKind.NumberValue => new Number(package, r.ReadDouble()),
			ExpressionKind.TextValue => new Text(package, r.ReadString()),
			ExpressionKind.BooleanValue => ReadBooleanValue(r, package),
			ExpressionKind.VariableRef => ReadVariableRef(r, package),
			ExpressionKind.MemberRef => ReadMemberRef(r, package),
			ExpressionKind.BinaryExpr => ReadBinaryExpr(r, package),
			ExpressionKind.MethodCallExpr => ReadMethodCallExpr(r, package),
			_ => throw new InvalidBytecodeFileException()
		};
	}

	private static Value ReadBooleanValue(BinaryReader r, Package package)
	{
		var type = package.GetType(r.ReadString());
		return new Value(type, new ValueInstance(type, r.ReadBoolean()));
	}

	private static Expression ReadVariableRef(BinaryReader r, Package package)
	{
		var name = r.ReadString();
		var typeName = r.ReadString();
		var type = package.GetType(typeName);
		var param = new Parameter(type, name, new Value(type, new ValueInstance(type)));
		return new ParameterCall(param);
	}

	private static MemberCall ReadMemberRef(BinaryReader r, Package package)
	{
		var memberName = r.ReadString();
		var memberTypeName = r.ReadString();
		var hasInstance = r.ReadBoolean();
		Expression? instance = hasInstance ? ReadExpression(r, package) : null;
		// Construct a lightweight Member whose Name and Type match what the VM needs.
		// The declaring type (first constructor arg) is only used for GetType() lookup during
		// construction, so any accessible base type works as the context.
		var anyBaseType = package.GetType(Type.Number);
		var fakeMember = new Member(anyBaseType, memberName + " " + memberTypeName, null);
		return new MemberCall(instance, fakeMember);
	}

	private static Binary ReadBinaryExpr(BinaryReader r, Package package)
	{
		var operatorName = r.ReadString();
		var left = ReadExpression(r, package);
		var right = ReadExpression(r, package);
		var operatorMethod = FindOperatorMethod(package, operatorName, left.ReturnType);
		return new Binary(left, operatorMethod, [right]);
	}

	private static Method FindOperatorMethod(Package package, string operatorName, Type preferredType)
	{
		var method = preferredType.Methods.FirstOrDefault(m => m.Name == operatorName);
		if (method != null)
			return method;
		foreach (var typeName in new[] { Type.Number, Type.Text, Type.Boolean })
		{
			method = package.GetType(typeName).Methods.FirstOrDefault(m => m.Name == operatorName);
			if (method != null)
				return method;
		}
		throw new MethodNotFoundException(operatorName);
	}

	private static MethodCall ReadMethodCallExpr(BinaryReader r, Package package)
	{
		var declaringTypeName = r.ReadString();
		var methodName = r.ReadString();
		var paramCount = r.ReadInt32();
		var returnTypeName = r.ReadString();
		var hasInstance = r.ReadBoolean();
		Expression? instance = hasInstance ? ReadExpression(r, package) : null;
		var argCount = r.ReadInt32();
		var args = new Expression[argCount];
		for (var i = 0; i < argCount; i++)
			args[i] = ReadExpression(r, package);
		var declaringType = package.GetType(declaringTypeName);
		var method = FindMethod(declaringType, methodName, paramCount);
		var returnType = returnTypeName != method.ReturnType.Name
			? package.GetType(returnTypeName)
			: null;
		return new MethodCall(method, instance, args, returnType);
	}

	private static (MethodCall? MethodCall, Registry? Registry) ReadMethodCallData(BinaryReader r,
		Package package)
	{
		MethodCall? methodCall = null;
		if (r.ReadBoolean())
		{
			var declaringTypeName = r.ReadString();
			var methodName = r.ReadString();
			var paramCount = r.ReadInt32();
			var returnTypeName = r.ReadString();
			var hasInstance = r.ReadBoolean();
			Expression? instance = hasInstance ? ReadExpression(r, package) : null;
			var argCount = r.ReadInt32();
			var args = new Expression[argCount];
			for (var i = 0; i < argCount; i++)
				args[i] = ReadExpression(r, package);
			var declaringType = package.GetType(declaringTypeName);
			var method = FindMethod(declaringType, methodName, paramCount);
			var returnType = returnTypeName != method.ReturnType.Name
				? package.GetType(returnTypeName)
				: null;
			methodCall = new MethodCall(method, instance, args, returnType);
		}
		Registry? registry = null;
		if (r.ReadBoolean())
		{
			registry = new Registry();
			var nextRegisterCount = r.ReadByte();
			var prev = (Register)r.ReadByte();
			// Replay AllocateRegister calls to restore NextRegister; assumes each call increments
			// the counter by exactly 1 (current Registry implementation).
			for (var i = 0; i < nextRegisterCount; i++)
				registry.AllocateRegister();
			registry.PreviousRegister = prev;
		}
		return (methodCall, registry);
	}

	private static Method FindMethod(Type type, string methodName, int paramCount)
	{
		var method = type.Methods.FirstOrDefault(m =>
			m.Name == methodName && m.Parameters.Count == paramCount) ??
			type.Methods.FirstOrDefault(m => m.Name == methodName);
		if (method != null)
			return method;
		if (type.AvailableMethods.TryGetValue(methodName, out var available))
		{
			var found = available.FirstOrDefault(m => m.Parameters.Count == paramCount) ??
				available.FirstOrDefault();
			if (found != null)
				return found;
		}
		throw new MethodNotFoundException(methodName);
	}

	public sealed class MethodNotFoundException(string methodName)
		: Exception($"Method '{methodName}' not found") { }
}
