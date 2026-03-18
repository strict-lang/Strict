using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Compiler;

public abstract class InstructionsCompiler
{
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

	protected static bool HasPrintInstructionsInternal(IReadOnlyList<Instruction> instructions) =>
		instructions.OfType<PrintInstruction>().Any();

	protected static string BuildMethodHeaderKeyInternal(Method method) =>
		BinaryExecutable.BuildMethodHeader(method.Name,
			method.Parameters.Select(parameter =>
				new BinaryMember(parameter.Name, parameter.Type.Name, null)).ToList(),
			method.ReturnType);

	protected static Dictionary<string, List<Instruction>> BuildPrecompiledMethodsInternal(
		BinaryExecutable binary)
	{
		var methods = new Dictionary<string, List<Instruction>>(StringComparer.Ordinal);
		foreach (var typeData in binary.MethodsPerType.Values)
			foreach (var (methodName, overloads) in typeData.MethodGroups)
				foreach (var overload in overloads)
				{
					var methodKey = BuildMethodHeaderKeyInternal(methodName, overload);
					methods[methodKey] = overload.instructions;
				}
		return methods;
	}

	private static string BuildMethodHeaderKeyInternal(string methodName, BinaryMethod method) =>
		method.parameters.Count == 0
      ? BinaryMemberJustTypeName(method.ReturnTypeName) == Type.None
				? methodName
        : methodName + " " + BinaryMemberJustTypeName(method.ReturnTypeName)
			: methodName + "(" + string.Join(", ", method.parameters) + ") " +
				BinaryMemberJustTypeName(method.ReturnTypeName);

	private static string BinaryMemberJustTypeName(string fullTypeName) =>
		fullTypeName.Split(Context.ParentSeparator)[^1];

	public abstract Task<string> Compile(BinaryExecutable binary, Platform platform);
	public abstract string Extension { get; }
}