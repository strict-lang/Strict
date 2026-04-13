using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Optimizers;

/// <summary>
/// Converts constant ColorValue constructions to the more compact Color (Byte-based) type.
/// For example, ColorValue(0.5, 0.5, 0.25) becomes Color(127, 127, 63) where each component
/// is multiplied by 255 and clamped to byte range. This eliminates the need for ColorValue
/// at runtime, replacing it with the 4-byte Color representation for better memory and speed.
/// </summary>
public sealed class ColorValueToColorOptimizer : InstructionOptimizer
{
	public override List<Instruction> Optimize(List<Instruction> instructions) => instructions;

	public override void Optimize(BinaryExecutable binary)
	{
		colorType = binary.basePackage.FindType(ColorTypeName);
		colorValueType = binary.basePackage.FindType(ColorValueTypeName);
		if (colorType == null || colorValueType == null)
			return;
		foreach (var typeEntry in binary.MethodsPerType)
		foreach (var methodGroup in typeEntry.Value.MethodGroups.Values)
		foreach (var method in methodGroup)
			method.instructions = OptimizeInstructions(method.instructions);
	}

	private const string ColorTypeName = "Color";
	private const string ColorValueTypeName = "ColorValue";
	private Type? colorType;
	private Type? colorValueType;

	private List<Instruction> OptimizeInstructions(List<Instruction> instructions)
	{
		for (var index = 0; index < instructions.Count; index++)
			TryConvertColorValueConstruction(instructions, index);
		return instructions;
	}

	private void TryConvertColorValueConstruction(List<Instruction> instructions, int index)
	{
		switch (instructions[index])
		{
		case Invoke invoke when IsColorValueFromConstructor(invoke.MethodInfo):
			TryConvertInvokeToColor(instructions, index, invoke);
			break;
		case ConstructValueTypeInstruction construct when construct.ReturnType == colorValueType:
			TryConvertConstructToColor(instructions, index, construct);
			break;
		}
	}

	private bool IsColorValueFromConstructor(InvokeMethodInfo methodInfo) =>
		methodInfo.MethodName == Method.From &&
		(methodInfo.TypeFullName == ColorValueTypeName ||
			methodInfo.TypeFullName.EndsWith(Context.ParentSeparator + ColorValueTypeName,
				StringComparison.Ordinal));

	private void TryConvertInvokeToColor(List<Instruction> instructions, int invokeIndex,
		Invoke invoke)
	{
		var argRegisters = invoke.MethodInfo.ArgumentRegisters;
		if (argRegisters.Length < 3)
			return;
		var constantArgs = FindConstantArguments(instructions, invokeIndex, argRegisters);
		if (constantArgs == null)
			return;
		ReplaceWithColorConstruction(instructions, invokeIndex, invoke.Register, constantArgs,
			argRegisters);
	}

	private void TryConvertConstructToColor(List<Instruction> instructions, int constructIndex,
		ConstructValueTypeInstruction construct)
	{
		var fieldRegisters = construct.FieldRegisters;
		if (fieldRegisters.Length < 3)
			return;
		var constantArgs = FindConstantArguments(instructions, constructIndex, fieldRegisters);
		if (constantArgs == null)
			return;
		ReplaceWithColorConstruction(instructions, constructIndex, construct.Register, constantArgs,
			fieldRegisters);
	}

	private static double[]? FindConstantArguments(List<Instruction> instructions,
		int beforeIndex, Register[] registers)
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
			if (WritesToRegister(instructions[scanIndex], register))
				return -1;
		}
		return -1;
	}

	private static bool WritesToRegister(Instruction instruction, Register register) =>
		instruction switch
		{
			LoadVariableToRegister load => load.Register == register,
			LoadConstantInstruction load => load.Register == register,
			BinaryInstruction binary when binary.Registers.Length >= 3 => binary.Registers[2] == register,
			Invoke invoke => invoke.Register == register,
			ConstructValueTypeInstruction construct => construct.Register == register,
			_ => false
		};

	private void ReplaceWithColorConstruction(List<Instruction> instructions, int targetIndex,
		Register outputRegister, double[] colorValueComponents, Register[] argRegisters)
	{
		var byteValues = ConvertToByteValues(colorValueComponents);
		for (var argIndex = 0; argIndex < argRegisters.Length; argIndex++)
		{
			var loadIndex = FindLoadConstantIndex(instructions, targetIndex, argRegisters[argIndex]);
			if (loadIndex >= 0)
				instructions[loadIndex] = new LoadConstantInstruction(argRegisters[argIndex],
					new ValueInstance(colorType!.GetType(Language.Type.Number), byteValues[argIndex]));
		}
		instructions[targetIndex] =
			new ConstructValueTypeInstruction(outputRegister, colorType!, argRegisters);
	}

	private static double[] ConvertToByteValues(double[] floatComponents)
	{
		var byteValues = new double[floatComponents.Length];
		for (var index = 0; index < floatComponents.Length; index++)
			byteValues[index] = Math.Clamp(Math.Round(floatComponents[index] * 255), 0, 255);
		return byteValues;
	}
}
