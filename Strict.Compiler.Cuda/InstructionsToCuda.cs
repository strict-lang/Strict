using Strict.Expressions;
using Strict.Language;
using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Type = Strict.Language.Type;

namespace Strict.Compiler.Cuda;

/// <summary>
/// Compiles a Strict method to a CUDA C kernel using the Runtime's <see cref="BinaryGenerator" />
/// to produce bytecode instructions, then translates them to CUDA C.
/// </summary>
public sealed class InstructionsToCuda : InstructionsCompiler
{
	public override Task<string> Compile(BinaryExecutable binary, Platform platform)
	{
		var output = BuildCudaKernel(Method.Run, binary.EntryPoint.Instructions);
		return Task.FromResult(output);
	}

	public override string Extension => ".cu";
	public string Compile(Method method) => BuildCudaKernel(method.Name, []);

	private static string BuildCudaKernel(string methodName, IReadOnlyList<Instruction> instructions)
	{
		var registers = new Dictionary<Register, string>();
		var outputExpression = "0.0f";
		foreach (var instruction in instructions)
			switch (instruction)
			{
			case LoadVariableToRegister load:
				registers[load.Register] = load.Identifier + "[idx]";
				break;
			case LoadConstantInstruction constant:
				registers[constant.Register] =
					constant.Constant.Number.ToString(System.Globalization.CultureInfo.InvariantCulture);
				break;
			case BinaryInstruction binary when !binary.IsConditional() && binary.Registers.Length > 2:
				registers[binary.Registers[2]] =
					$"{registers.GetValueOrDefault(binary.Registers[0], "0")}" +
					$" {GetOperatorSymbol(binary.InstructionType)} " +
					$"{registers.GetValueOrDefault(binary.Registers[1], "0")}";
				break;
			case ReturnInstruction ret when registers.TryGetValue(ret.Register, out var value):
				outputExpression = value;
				break;
			}
		return $@"extern ""C"" __global__ void {
			methodName
		}(float *output, const int count)
{{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = y * blockDim.x + x;
	if (idx >= count) return;
	output[idx] = {
		outputExpression
	};
}}";
	}

	private static string GetOperatorSymbol(InstructionType instruction) =>
		instruction switch
		{
			InstructionType.Add => "+",
			InstructionType.Subtract => "-",
			InstructionType.Multiply => "*",
			InstructionType.Divide => "/", //ncrunch: no coverage
			_ => throw new NotSupportedException(instruction.ToString()) //ncrunch: no coverage
		};
}
