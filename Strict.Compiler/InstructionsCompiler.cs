using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Language;

namespace Strict.Compiler;

public class InstructionsCompiler
{
	protected static bool HasPrintInstructionsInternal(IReadOnlyList<Instruction> instructions) =>
		instructions.OfType<PrintInstruction>().Any();

	protected static bool IsPlatformUsingStdLibAndHasPrintInstructionsInternal(Platform platform,
		IReadOnlyList<Instruction> optimizedInstructions,
		IReadOnlyDictionary<string, List<Instruction>>? precompiledMethods,
		bool includeWindowsPlatform)
	{
		var platformUsesStdLib = includeWindowsPlatform
			? platform is Platform.Linux or Platform.Windows
			: platform == Platform.Linux;
		return platformUsesStdLib && (HasPrintInstructionsInternal(optimizedInstructions) ||
			(precompiledMethods?.Values.Any(methodInstructions =>
				HasPrintInstructionsInternal(methodInstructions)) ?? false));
	}

	protected static string BuildMethodHeaderKeyInternal(Method method) =>
		BinaryExecutable.BuildMethodHeader(method.Name,
			method.Parameters.Select(parameter =>
				new BinaryMember(parameter.Name, parameter.Type.Name, null)).ToList(),
			method.ReturnType);
}