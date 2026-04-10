using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Compiler;

public abstract class InstructionsCompiler
{
	protected static string BuildMethodHeaderKeyInternal(InvokeMethodInfo info) =>
		info.ParameterNames.Length == 0
			? BinaryMemberJustTypeName(info.ReturnTypeName) == Type.None
				? info.MethodName
				: info.MethodName + " " + BinaryMemberJustTypeName(info.ReturnTypeName)
			: info.MethodName + "(" + string.Join(", ", info.ParameterNames) + ") " +
			BinaryMemberJustTypeName(info.ReturnTypeName);

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
			: methodName + "(" + string.Join(", ", method.parameters.Select(parameter => parameter.Name)) +
			") " + BinaryMemberJustTypeName(method.ReturnTypeName);

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
		IReadOnlyDictionary<string, List<Instruction>>? precompiledMethods,
		BinaryExecutable? binary = null)
	{
		var methods = new Dictionary<string, CompiledMethodInfo>(StringComparer.Ordinal);
		if (precompiledMethods == null)
			return methods;
		var queue = new Queue<InvokeMethodInfo>();
		EnqueueInvokedMethodInfos(instructions, queue);
		var processed = new HashSet<string>(StringComparer.Ordinal);
		while (queue.Count > 0)
		{
			var info = queue.Dequeue();
			var methodKey = BuildMethodHeaderKeyInternal(info);
			if (!processed.Add(methodKey))
				continue;
			if (!precompiledMethods.TryGetValue(methodKey, out var precompiled))
				continue;
			var methodInstructions = new List<Instruction>(precompiled);
			var memberNames = info.InstanceRegister.HasValue
				? GetMemberNamesFromBinary(binary, info.TypeFullName)
				: [];
			var parameterNames = new List<string>(memberNames);
			parameterNames.AddRange(info.ParameterNames);
			var typeName = BinaryMemberJustTypeName(info.TypeFullName);
			var symbol = typeName + "_" + info.MethodName + "_" + info.ParameterNames.Length;
			var compiledMethodInfo = new CompiledMethodInfo(symbol, methodInstructions,
				parameterNames, memberNames);
			methods[methodKey] = compiledMethodInfo;
			EnqueueInvokedMethodInfos(methodInstructions, queue);
		}
		return methods;
	}

	private static List<string> GetMemberNamesFromBinary(BinaryExecutable? binary, string typeFullName)
	{
		if (binary == null)
			return [];
		if (binary.MethodsPerType.TryGetValue(typeFullName, out var typeData))
			return typeData.Members.Where(member => !member.FullTypeName.EndsWith("Trait",
				StringComparison.OrdinalIgnoreCase)).Select(member => member.Name).ToList();
		var justTypeName = BinaryMemberJustTypeName(typeFullName);
		foreach (var (key, data) in binary.MethodsPerType)
			if (BinaryMemberJustTypeName(key) == justTypeName)
				return data.Members.Where(member => !member.FullTypeName.EndsWith("Trait",
					StringComparison.OrdinalIgnoreCase)).Select(member => member.Name).ToList();
		return [];
	}

	private static void EnqueueInvokedMethodInfos(IEnumerable<Instruction> instructions,
		Queue<InvokeMethodInfo> queue)
	{
		foreach (var instruction in instructions)
			if (instruction is Invoke invoke && invoke.MethodInfo.MethodName != Method.From)
				queue.Enqueue(invoke.MethodInfo);
	}

	protected static bool HasNumericPrint(IEnumerable<Instruction> instructions) =>
		instructions.OfType<PrintInstruction>().Any(print => print.ValueRegister.HasValue && !print.ValueIsText);

	public abstract Task<string> Compile(BinaryExecutable binary, Platform platform);
	public abstract string Extension { get; }
}