using Strict.Expressions;
using Strict.Language;
using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Type = Strict.Language.Type;

namespace Strict.Compiler.Cuda;

/// <summary>
/// Compiles a Strict method to a CUDA C kernel using the Runtime's <see cref="BytecodeGenerator" />
/// to produce bytecode instructions, then translates them to CUDA C.
/// </summary>
public sealed class InstructionsToCudaCompiler
{
	public string Compile(Method method) => BuildCudaKernel(method, GenerateInstructions(method));

	private static List<Instruction> GenerateInstructions(Method method)
	{
		var body = method.GetBodyAndParseIfNeeded();
		var expressions = body is Body b
			? b.Expressions
			: [body];
		var arguments = method.Parameters.ToDictionary(p => p.Name, p => new ValueInstance(p.Type, 0));
		return new BytecodeGenerator(new InvokedMethod(expressions, arguments, method.ReturnType),
			new Registry()).Generate();
	}

	private static string BuildCudaKernel(Method method, List<Instruction> instructions)
	{
		var registers = new Dictionary<Register, string>();
		var outputExpression = "";
		foreach (var instruction in instructions)
			switch (instruction)
			{
			case LoadVariableToRegister load:
				registers[load.Register] = IsScalarParameter(method, load.Identifier)
					? load.Identifier
					: load.Identifier + "[idx]";
				break;
			case LoadConstantInstruction constant:
				//ncrunch: no coverage start
				registers[constant.Register] =
					constant.ValueInstance.Number.ToString(System.Globalization.CultureInfo.InvariantCulture);
				break; //ncrunch: no coverage end
			case BinaryInstruction binary when !binary.IsConditional():
				registers[binary.Registers[2]] =
					$"{registers[binary.Registers[0]]} {GetOperatorSymbol(binary.InstructionType)} " +
					$"{registers[binary.Registers[1]]}";
				break;
			case ReturnInstruction ret:
				outputExpression = registers[ret.Register];
				break;
			}
		var parameterText = BuildParameterText(method) + "float *output";
		if (!HasDimensionParameters(method))
			parameterText += ", const int count";
		return $@"extern ""C"" __global__ void {
			method.Name
		}({
			parameterText
		})
{{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = y * blockDim.x + x;
	output[idx] = {
		outputExpression
	};
}}";
	}

	private static bool IsScalarParameter(Method method, string name) =>
		method.Parameters.Any(p => p.Name == name) &&
		name is "Width" or "Height" or "width" or "height" or "initialDepth";

	private static string GetOperatorSymbol(InstructionType instruction) =>
		instruction switch
		{
			InstructionType.Add => "+",
			InstructionType.Subtract => "-",
			InstructionType.Multiply => "*",
			InstructionType.Divide => "/", //ncrunch: no coverage
			_ => throw new NotSupportedException(instruction.ToString()) //ncrunch: no coverage
		};

	private static string BuildParameterText(Method method) =>
		method.Parameters.Aggregate("", (current, parameter) => current +
			parameter.Type.Name switch //ncrunch: no coverage
			{
				Type.Number when parameter.Name is "Width" or "Height" or "width" or "height" =>
					"const int " + parameter.Name + ", ",
				Type.Number when parameter.Name == "initialDepth" => "const float " + parameter.Name +
					", ",
				Type.Number => "const float *" + parameter.Name + ", ",
				_ => throw new NotSupportedException(parameter.ToString()) //ncrunch: no coverage
			});

	private static bool HasDimensionParameters(Method method) =>
		method.Parameters.Any(p => p.Name is "Width" or "Height" or "width" or "height");
}
