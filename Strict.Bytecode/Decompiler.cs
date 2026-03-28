using System.Text;
using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Bytecode;

/// <summary>
/// Partially reconstructs .strict source files from Binary (e.g., from .strictbinary) as an
/// approximation. For debugging, will not compile, no tests. Only includes what bytecode reveals.
/// </summary>
public sealed class Decompiler
{
	/// <summary>
	/// Opens a .strictbinary ZIP file, deserializes each bytecode entry, and writes
	/// a reconstructed .strict source file per entry into <paramref name="outputFolder" />.
	/// </summary>
	public void Decompile(BinaryExecutable allInstructions, string outputFolder)
	{
		Directory.CreateDirectory(outputFolder);
		foreach (var typeMethods in allInstructions.MethodsPerType)
		{
			var sourceLines = ReconstructSource(typeMethods.Value);
			var outputPath = Path.Combine(outputFolder, typeMethods.Key + ".strict");
			var outputDirectory = Path.GetDirectoryName(outputPath);
			if (!string.IsNullOrEmpty(outputDirectory))
				Directory.CreateDirectory(outputDirectory);
			File.WriteAllLines(outputPath, sourceLines, Encoding.UTF8);
		}
	}

	private IReadOnlyList<string> ReconstructSource(BinaryType typeData)
	{
		var lines = new List<string>();
		foreach (var member in typeData.Members)
			if (member.InitialValueExpression is SetInstruction setValue)
				lines.Add("const " + member.Name + " = " +
					setValue.ValueInstance.ToExpressionCodeString());
			else
				lines.Add("has " + member.Name + " " + member.JustTypeName);
		foreach (var (methodName, methods) in typeData.MethodGroups.OrderBy(
			group => group.Key == Method.Run
				? 1
				: 0))
		foreach (var method in methods)
		{
			lines.Add(BinaryType.ReconstructMethodName(methodName, method));
			lines.AddRange(ReconstructMethod(method));
		}
		return lines;
	}

	/// <summary>
	/// Converts a binary method instruction list into decompiled body lines.
	/// </summary>
	public IReadOnlyList<string> ReconstructMethod(BinaryMethod method)
	{
		registerExpressions.Clear();
		assignedVariables.Clear();
		var bodyLines = new List<string>();
		for (var instructionIndex = 0; instructionIndex < method.instructions.Count;
			instructionIndex++)
		{
			var instruction = method.instructions[instructionIndex];
			if (TryDeserializeInstruction(method, bodyLines, instruction, instructionIndex))
				continue;
			bodyLines.Add("\t" + instruction);
		}
		return bodyLines;
	}

	private readonly Dictionary<Register, string> registerExpressions = [];
	private readonly HashSet<string> assignedVariables = [];
	private string? lastCondition;

	private bool TryDeserializeInstruction(BinaryMethod method, List<string> bodyLines,
		Instruction instruction, int instructionIndex)
	{
		if (instruction is SetInstruction set)
		{
			registerExpressions[set.Register] = set.ValueInstance.ToExpressionCodeString();
			return true;
		}
		if (instruction is LoadConstantInstruction loadConstant)
		{
			registerExpressions[loadConstant.Register] = loadConstant.Constant.ToExpressionCodeString();
			return true;
		}
		if (instruction is LoadVariableToRegister loadVariable)
		{
			registerExpressions[loadVariable.Register] = loadVariable.Identifier;
			return true;
		}
		if (instruction is StoreVariableInstruction storeVariable)
		{
			if (!storeVariable.IsMember)
				bodyLines.Add("\tconstant " + storeVariable.Identifier + " = " +
					storeVariable.ValueInstance.ToExpressionCodeString());
			assignedVariables.Add(storeVariable.Identifier);
			return true;
		}
		if (instruction is StoreFromRegisterInstruction storeFromRegister)
		{
			var valueExpression = GetRegisterExpression(storeFromRegister.Register);
			bodyLines.Add("\t" + (assignedVariables.Contains(storeFromRegister.Identifier)
				? storeFromRegister.Identifier + " = " + valueExpression
				: "constant " + storeFromRegister.Identifier + " = " + valueExpression));
			assignedVariables.Add(storeFromRegister.Identifier);
			return true;
		}
		if (instruction is BinaryInstruction binary)
			return TryDeserializeBinaryInstruction(binary);
		if (instruction is Invoke invoke)
			return TryDeserializeInvokeInstruction(method, bodyLines, invoke, instructionIndex);
		if (instruction is PrintInstruction print)
		{
			bodyLines.Add("\t" + BuildPrintLine(print));
			return true;
		}
		if (instruction is ReturnInstruction returnInstruction)
		{
			if (method.ReturnTypeName == Type.None)
				return true;
			var returnExpression = GetRegisterExpression(returnInstruction.Register);
			if (bodyLines.Count == 0 || bodyLines[^1] != "\t" + returnExpression)
				bodyLines.Add("\t" + returnExpression);
			return true;
		}
		if (instruction is LoopBeginInstruction loopBegin)
		{
			bodyLines.Add("\tfor " + GetRegisterExpression(loopBegin.Register));
			return true;
		}
		if (instruction is LoopEndInstruction)
			return true;
		if (instruction is Jump or JumpIfNotZero or JumpToId)
			return TryDeserializeJumpInstruction(bodyLines, instruction);
		if (instruction is ListCallInstruction listCall)
		{
			registerExpressions[listCall.Register] = listCall.Identifier + "(" +
				GetRegisterExpression(listCall.IndexValueRegister) + ")";
			return true;
		}
		if (instruction is WriteToListInstruction writeToList)
		{
			bodyLines.Add("\t" + writeToList.Identifier + ".Add(" +
				GetRegisterExpression(writeToList.Register) + ")");
			return true;
		}
		if (instruction is WriteToTableInstruction writeToTable)
		{
			bodyLines.Add("\t" + writeToTable.Identifier + ".Add(" +
				GetRegisterExpression(writeToTable.Register) + ", " +
				GetRegisterExpression(writeToTable.Value) + ")");
			return true;
		}
		if (instruction is RemoveInstruction remove)
		{
			bodyLines.Add("\t" + remove.Identifier + ".Remove(" +
				GetRegisterExpression(remove.Register) + ")");
			return true;
		}
		return false;
	}

	private bool TryDeserializeBinaryInstruction(BinaryInstruction binary)
	{
		if (binary.Registers.Length < 2)
			return false;
		var left = GetRegisterExpression(binary.Registers[0]);
		var right = GetRegisterExpression(binary.Registers[1]);
		if (binary.IsConditional())
		{
			lastCondition = left + " " + GetConditionalOperator(binary.InstructionType) + " " + right;
			return true;
		}
		if (binary.Registers.Length < 3)
			return false;
		registerExpressions[binary.Registers[2]] = left + " " +
			GetArithmeticOperator(binary.InstructionType) + " " + right;
		return true;
	}

	private bool TryDeserializeInvokeInstruction(BinaryMethod method, List<string> bodyLines,
		Invoke invoke, int instructionIndex)
	{
		registerExpressions[invoke.Register] = invoke.Method.ToString();
		if (instructionIndex + 1 < method.instructions.Count &&
			method.instructions[instructionIndex + 1] is StoreFromRegisterInstruction nextStore &&
			nextStore.Register == invoke.Register)
			return true;
		if (invoke.Method.ReturnType.Name == Type.None)
			bodyLines.Add("\t" + invoke.Method);
		return true;
	}

	private bool TryDeserializeJumpInstruction(List<string> bodyLines, Instruction instruction)
	{
		if (instruction is JumpToId { InstructionType: InstructionType.JumpToIdIfFalse })
		{
			bodyLines.Add("\tif " + (lastCondition ?? "condition"));
			return true;
		}
		if (instruction is Jump { InstructionType: InstructionType.JumpIfFalse })
		{
			bodyLines.Add("\tif " + (lastCondition ?? "condition"));
			return true;
		}
		if (instruction is Jump { InstructionType: InstructionType.JumpIfTrue } or JumpToId
			{
				InstructionType: InstructionType.JumpToIdIfTrue
			})
		{
			bodyLines.Add("\tif not " + (lastCondition ?? "condition"));
			return true;
		}
		if (instruction is JumpToId { InstructionType: InstructionType.JumpEnd })
			return true;
		if (instruction is JumpIfNotZero jumpIfNotZero)
		{
			bodyLines.Add("\tif " + GetRegisterExpression(jumpIfNotZero.Register) + " is not 0");
			return true;
		}
		return false;
	}

	private string BuildPrintLine(PrintInstruction print)
	{
		if (!print.ValueRegister.HasValue)
			return "logger.Log(\"" + print.TextPrefix + "\")";
		if (print.TextPrefix.Length == 0)
			return "logger.Log(" + GetRegisterExpression(print.ValueRegister.Value) + ")";
		return "logger.Log(\"" + print.TextPrefix + "\" + " +
			GetRegisterExpression(print.ValueRegister.Value) + ")";
	}

	private string GetRegisterExpression(Register register) =>
		registerExpressions.TryGetValue(register, out var expression)
			? expression
			: register.ToString();

	private static string GetArithmeticOperator(InstructionType instructionType) =>
		instructionType switch
		{
			InstructionType.Add => "+",
			InstructionType.Subtract => "-",
			InstructionType.Multiply => "*",
			InstructionType.Divide => "/",
			InstructionType.Modulo => "%",
			_ => instructionType.ToString()
		};

	private static string GetConditionalOperator(InstructionType instructionType) =>
		instructionType switch
		{
			InstructionType.Equal => "is",
			InstructionType.NotEqual => "is not",
			InstructionType.GreaterThan => ">",
			InstructionType.LessThan => "<",
			_ => instructionType.ToString()
		};
}