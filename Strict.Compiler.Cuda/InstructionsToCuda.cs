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
   var output = BuildCudaKernel(Method.Run, binary.EntryPoint.instructions, [], true);
		return Task.FromResult(output);
	}

	public override string Extension => ".cu";
  public string Compile(Method method) =>
		BuildCudaKernel(method, new BinaryGenerator(new MethodCall(method)).Generate().EntryPoint.instructions);

  private static string BuildCudaKernel(Method method, IReadOnlyList<Instruction> instructions) =>
		BuildCudaKernel(method.Name, instructions, method.Parameters, NeedsCountParameter(method));

	private static string BuildCudaKernel(string methodName, IReadOnlyList<Instruction> instructions,
		IReadOnlyList<Parameter> parameters, bool addCountParameter)
	{
		var registers = new Dictionary<Register, string>();
		var outputExpression = "0.0f";
		foreach (var instruction in instructions)
			switch (instruction)
			{
			case LoadVariableToRegister load:
       registers[load.Register] = load.Identifier + (IsScalarParameter(parameters, load.Identifier)
					? ""
					: "[idx]");
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
   var kernelParameters = string.Join(", ", parameters.Select(GetParameterDeclaration).Append(
			"float *output").Append(addCountParameter
				? "const int count"
				: string.Empty).Where(parameter => parameter.Length > 0));
		return $@"extern ""C"" __global__ void {
     methodName
		}({kernelParameters})
{{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = y * blockDim.x + x;
  if ({GetBoundsCheck(parameters, addCountParameter)}) return;
	output[idx] = {
		outputExpression
	};
}}";
	}

	private static bool NeedsCountParameter(Method method) =>
    !HasParameter(method.Parameters, "width") || !HasParameter(method.Parameters, "height");

  private static string GetBoundsCheck(IReadOnlyList<Parameter> parameters, bool addCountParameter) =>
		addCountParameter || !HasParameter(parameters, "width") || !HasParameter(parameters, "height")
			? "idx >= count"
			: "x >= width || y >= height";

	private static string GetParameterDeclaration(Parameter parameter) =>
		IsScalarParameter(parameter.Name)
			? parameter.Name is "width" or "height"
				? $"const int {parameter.Name}"
				: $"const float {parameter.Name}"
			: $"const float *{parameter.Name}";

  private static bool IsScalarParameter(IReadOnlyList<Parameter> parameters, string name) =>
		parameters.Any(parameter => parameter.Name == name && IsScalarParameter(name));

	private static bool IsScalarParameter(string name) =>
		name is "width" or "height" or "initialDepth";

  private static bool HasParameter(IReadOnlyList<Parameter> parameters, string name) =>
		parameters.Any(parameter => parameter.Name == name);

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
