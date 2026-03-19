using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Compiler;

public abstract class InstructionsCompiler
{
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

	protected sealed class CompiledMethodInfo(string symbol,
		List<Instruction> instructions, List<string> parameterNames, List<string> memberNames)
	{
		public string Symbol { get; } = symbol;
		public List<Instruction> Instructions { get; } = instructions;
		public List<string> ParameterNames { get; } = parameterNames;
		public List<string> MemberNames { get; } = memberNames;
	}

	protected static Dictionary<string, CompiledMethodInfo> CollectMethods(
		List<Instruction> instructions,
		IReadOnlyDictionary<string, List<Instruction>>? precompiledMethods)
	{
		var methods = new Dictionary<string, CompiledMethodInfo>(StringComparer.Ordinal);
		var queue = new Queue<(Method Method, bool IncludeMembers)>();
		EnqueueInvokedMethods(instructions, queue);
		while (queue.Count > 0)
		{
			var (method, includeMembers) = queue.Dequeue();
			var methodKey = BuildMethodHeaderKeyInternal(method);
			if (methods.TryGetValue(methodKey, out var existing))
			{
				if (includeMembers && existing.MemberNames.Count == 0)
					methods[methodKey] = BuildMethodInfo(method, true, precompiledMethods);
				continue;
			}
			var methodInfo = BuildMethodInfo(method, includeMembers, precompiledMethods);
			methods[methodKey] = methodInfo;
			EnqueueInvokedMethods(methodInfo.Instructions, queue);
		}
		return methods;
	}

	private static CompiledMethodInfo BuildMethodInfo(Method method, bool includeMembers,
		IReadOnlyDictionary<string, List<Instruction>>? precompiledMethods)
	{
		var methodKey = BuildMethodHeaderKeyInternal(method);
		var instructions =
			precompiledMethods != null && precompiledMethods.TryGetValue(methodKey, out var precompiled)
				? new List<Instruction>(precompiled)
				: throw new NotSupportedException("Method " + methodKey + " must be precompiled in BinaryExecutable. Ensure it is included via BuildPrecompiledMethodsInternal.");
		var memberNames = includeMembers
			? method.Type.Members.Where(member => !member.Type.IsTrait).Select(member => member.Name).ToList()
			: new List<string>();
		var parameterNames = new List<string>(memberNames);
		parameterNames.AddRange(method.Parameters.Select(parameter => parameter.Name));
		return new CompiledMethodInfo(BuildMethodSymbol(method), instructions, parameterNames,
			memberNames);
	}

	protected static string BuildMethodSymbol(Method method) =>
		method.Type.Name + "_" + method.Name + "_" + method.Parameters.Count;

	private static void EnqueueInvokedMethods(IEnumerable<Instruction> instructions,
		Queue<(Method Method, bool IncludeMembers)> queue)
	{
		foreach (var invoke in instructions.OfType<Invoke>())
			if (invoke.Method != null && invoke.Method.Method.Name != Method.From)
				queue.Enqueue((invoke.Method.Method, invoke.Method.Instance != null));
	}

	protected static bool HasNumericPrint(IEnumerable<Instruction> instructions) =>
		instructions.OfType<PrintInstruction>().Any(print => print.ValueRegister.HasValue && !print.ValueIsText);

	public abstract Task<string> Compile(BinaryExecutable binary, Platform platform);
	public abstract string Extension { get; }
}