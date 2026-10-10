using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Optimizers;

/// <summary>
/// Replaces constant constructions of a larger data type with its compact equivalent when a
/// bidirectional conversion exists (from constructor + to method) and the compact type uses
/// fewer bytes per instance. For example ColorValue(0.5, 0.5, 0.25) becomes Color(128, 128, 64)
/// because Color stores each component as a Byte (4 bytes total) instead of Number (32 bytes).
/// Not in AllInstructionOptimizers: Color.from(colorValue) keeps 127.5 and Alpha 255, compacting
/// rounds to 128 and leaves the missing Alpha at 0, results would differ from unoptimized runs.
/// </summary>
public sealed class CompactTypeOptimizer : InstructionOptimizer
{
	public override List<Instruction> Optimize(List<Instruction> instructions) => instructions;

	public override void Optimize(BinaryExecutable binary)
	{
		typePairs = FindCompactableTypePairs(binary.basePackage);
		if (typePairs.Count == 0)
			return;
		foreach (var typeEntry in binary.MethodsPerType)
		foreach (var methodGroup in typeEntry.Value.MethodGroups.Values)
		foreach (var method in methodGroup)
			for (var index = 0; index < method.instructions.Count; index++)
				TryConvertConstruction(method, index);
	}

	private List<TypeConversionPair> typePairs = [];

	/// <summary>
	/// Scans all types in the package (and sub-packages) to find pairs where a compact type has
	/// a from(largerType) constructor and a to LargerType method, and is smaller in memory.
	/// </summary>
	private static List<TypeConversionPair> FindCompactableTypePairs(Package package)
	{
		var pairs = new List<TypeConversionPair>();
		foreach (var compactType in GetAllTypesInPackage(package))
		{
			if (compactType.IsTrait || compactType.Members.Count == 0)
				continue;
			foreach (var method in compactType.Methods)
			{
				if (method.Name != Method.From || method.Parameters.Count != 1)
					continue;
				var largerType = method.Parameters[0].Type;
				if (largerType.IsTrait || largerType.Members.Count == 0 ||
					largerType.Members.Count != compactType.Members.Count)
					continue;
				if (!HasToConversion(compactType, largerType))
					continue;
				var compactSize = EstimateTypeByteSize(compactType);
				var largerSize = EstimateTypeByteSize(largerType);
				if (compactSize >= largerSize)
					continue;
				var conversionFactor = DetectConversionFactor(compactType, largerType);
				if (conversionFactor <= 0)
					continue;
				pairs.Add(new TypeConversionPair(compactType, largerType, conversionFactor));
			}
		}
		return pairs;
	}

	private static IEnumerable<Type> GetAllTypesInPackage(Package package) => package.Types.Values;

	private static bool HasToConversion(Type compactType, Type largerType) =>
		compactType.Methods.Any(method => method.Name == "to" && method.ReturnType == largerType);

	/// <summary>
	/// Estimates the in-memory byte size of a type based on its members.
	/// Byte members = 1 byte, Number members = 8 bytes, Boolean = 1, others = 8.
	/// </summary>
	private static int EstimateTypeByteSize(Type type) =>
		type.Members.Sum(member => EstimateMemberByteSize(member.Type));

	private static int EstimateMemberByteSize(Type memberType) =>
		memberType.Name switch
		{
			"Byte" => 1,
			Type.Boolean => 1,
			Type.Character => 2,
			_ => 8
		};

	/// <summary>
	/// Detects the numeric conversion factor between matching member types.
	/// When compact type has Byte members and larger type has Number members, factor = 255.
	/// </summary>
	private static double DetectConversionFactor(Type compactType, Type largerType)
	{
		for (var index = 0; index < compactType.Members.Count; index++)
		{
			var compactMember = compactType.Members[index];
			var largerMember = largerType.Members[index];
			if (compactMember.Type.Name == "Byte" && largerMember.Type.IsNumber)
				return 255;
		}
		return 0;
	}

	private void TryConvertConstruction(BinaryMethod method, int index)
	{
		switch (method.instructions[index])
		{
		case Invoke invoke when FindMatchingPairForInvoke(invoke.MethodInfo) is { } pair &&
			IsOnlyReadAsCompactType(method, index, pair.CompactType):
			TryConvertInvoke(method.instructions, index, invoke, pair);
			break;
		case ConstructValueTypeInstruction construct
			when FindMatchingPairForConstruct(construct.ReturnType) is { } pair &&
			IsOnlyReadAsCompactType(method, index, pair.CompactType):
			TryConvertConstruct(method.instructions, index, construct, pair);
			break;
		}
	}

	/// <summary>
	/// Compacting changes the value type, so every reader must expect the compact type. ponytail:
	/// only a return from a method declared with the compact type counts, stores, parameters and
	/// list elements read the larger value through their generated Color.from conversion.
	/// </summary>
	private static bool IsOnlyReadAsCompactType(BinaryMethod method, int constructIndex,
		Type compactType)
	{
		var register = ((RegisterInstruction)method.instructions[constructIndex]).Register;
		for (var index = constructIndex + 1; index < method.instructions.Count; index++)
		{
			var instruction = method.instructions[index];
			if (instruction is ReturnInstruction returnInstruction)
				return returnInstruction.Register != register ||
					method.ReturnTypeName == compactType.Name;
			if (GetReadRegisters(instruction).Contains(register))
				return false;
			if (GetWrittenRegister(instruction) == register)
				return true;
			if (IsControlFlow(instruction))
				return false;
		}
		return true;
	}

	private TypeConversionPair? FindMatchingPairForInvoke(InvokeMethodInfo methodInfo)
	{
		if (methodInfo.MethodName != Method.From)
			return null;
		return typePairs.FirstOrDefault(pair =>
			methodInfo.TypeFullName == pair.LargerType.Name ||
			methodInfo.TypeFullName.EndsWith(Context.ParentSeparator + pair.LargerType.Name,
				StringComparison.Ordinal));
	}

	private TypeConversionPair? FindMatchingPairForConstruct(Type returnType) =>
		typePairs.FirstOrDefault(pair => pair.LargerType == returnType);

	private static void TryConvertInvoke(List<Instruction> instructions, int invokeIndex,
		Invoke invoke, TypeConversionPair pair)
	{
		var argRegisters = invoke.MethodInfo.ArgumentRegisters;
		if (argRegisters.Length < 1)
			return;
		var constantArgs = FindConstantArguments(instructions, invokeIndex, argRegisters);
		if (constantArgs == null)
			return;
		ReplaceWithCompactConstruction(instructions, invokeIndex, invoke.Register, constantArgs,
			argRegisters, pair);
	}

	private static void TryConvertConstruct(List<Instruction> instructions, int constructIndex,
		ConstructValueTypeInstruction construct, TypeConversionPair pair)
	{
		var fieldRegisters = construct.FieldRegisters;
		if (fieldRegisters.Length < 1)
			return;
		var constantArgs = FindConstantArguments(instructions, constructIndex, fieldRegisters);
		if (constantArgs == null)
			return;
		ReplaceWithCompactConstruction(instructions, constructIndex, construct.Register, constantArgs,
			fieldRegisters, pair);
	}

	private static double[]? FindConstantArguments(List<Instruction> instructions, int beforeIndex,
		Register[] registers)
	{
		var values = new double[registers.Length];
		for (var argIndex = 0; argIndex < registers.Length; argIndex++)
		{
			var loadIndex = FindLoadConstantIndex(instructions, beforeIndex, registers[argIndex]);
			if (loadIndex < 0)
				return null;
			values[argIndex] = ((LoadConstantInstruction)instructions[loadIndex]).Constant.Number;
		}
		return values;
	}

	private static int FindLoadConstantIndex(List<Instruction> instructions, int beforeIndex,
		Register register)
	{
		for (var scanIndex = beforeIndex - 1; scanIndex >= 0; scanIndex--)
		{
			if (instructions[scanIndex] is LoadConstantInstruction loadConstant &&
				loadConstant.Register == register)
				return scanIndex;
			if (GetWrittenRegister(instructions[scanIndex]) == register ||
				IsControlFlow(instructions[scanIndex]))
				return -1;
		}
		return -1;
	}

	private static void ReplaceWithCompactConstruction(List<Instruction> instructions,
		int targetIndex, Register outputRegister, double[] sourceValues, Register[] argRegisters,
		TypeConversionPair pair)
	{
		var convertedValues = ConvertValues(sourceValues, pair.ConversionFactor);
		for (var argIndex = 0; argIndex < argRegisters.Length; argIndex++)
		{
			var loadIndex = FindLoadConstantIndex(instructions, targetIndex, argRegisters[argIndex]);
			if (loadIndex >= 0)
				instructions[loadIndex] = new LoadConstantInstruction(argRegisters[argIndex],
					new ValueInstance(pair.CompactType.GetType(Type.Number), convertedValues[argIndex]));
		}
		instructions[targetIndex] =
			new ConstructValueTypeInstruction(outputRegister, pair.CompactType, argRegisters);
	}

	private static double[] ConvertValues(double[] sourceValues, double factor)
	{
		var converted = new double[sourceValues.Length];
		for (var index = 0; index < sourceValues.Length; index++)
			converted[index] = Math.Clamp(Math.Round(sourceValues[index] * factor), 0, factor);
		return converted;
	}

	private sealed record TypeConversionPair(Type CompactType,
		Type LargerType,
		double ConversionFactor);
}