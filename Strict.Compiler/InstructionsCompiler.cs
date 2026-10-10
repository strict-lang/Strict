using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Compiler;

public abstract class InstructionsCompiler
{
	public sealed class NotSupportedByBackend(string message) : Exception(message);

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
			: methodName + "(" +
			string.Join(", ", method.parameters.Select(parameter => parameter.Name)) + ") " +
			BinaryMemberJustTypeName(method.ReturnTypeName);

	private static string BinaryMemberJustTypeName(string fullTypeName) =>
		fullTypeName.Split(Context.ParentSeparator)[^1];

	protected sealed class CompiledMethodInfo(string symbol,
		List<Instruction> instructions,
		List<string> parameterNames,
		List<string> memberNames)
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
				? GetInstanceMembers(binary, info.TypeFullName).Select(member => member.Member.Name).ToList()
				: [];
			var parameterNames = new List<string>(memberNames);
			parameterNames.AddRange(info.ParameterNames);
			var typeName = BinaryMemberJustTypeName(info.TypeFullName);
			var symbol = typeName + "_" + info.MethodName + "_" + info.ParameterNames.Length;
			if (methods.Values.Any(method => method.Symbol == symbol))
				symbol += "_" + methods.Count;
			var compiledMethodInfo = new CompiledMethodInfo(symbol, methodInstructions,
				parameterNames, memberNames);
			methods[methodKey] = compiledMethodInfo;
			EnqueueInvokedMethodInfos(methodInstructions, queue);
		}
		return methods;
	}

	/// <summary>
	/// Number members are the native state of an instance (passed before the method parameters),
	/// Position counts all non-constant members for positional constructor arguments.
	/// </summary>
	protected static List<(BinaryMember Member, int Position)> GetInstanceMembers(
		BinaryExecutable? binary, string typeFullName) =>
	[
		.. (FindBinaryType(binary, typeFullName)?.Members.Where(member => !member.IsConstant) ?? []).
		Select((member, position) => (member, position)).
		Where(pair => pair.member.JustTypeName == Type.Number)
	];

	private static BinaryType? FindBinaryType(BinaryExecutable? binary, string typeFullName)
	{
		if (binary == null)
			return null;
		if (binary.MethodsPerType.TryGetValue(typeFullName, out var typeData))
			return typeData;
		var justTypeName = BinaryMemberJustTypeName(typeFullName);
		foreach (var (key, data) in binary.MethodsPerType)
			if (BinaryMemberJustTypeName(key) == justTypeName)
				return data;
		return null;
	}

	private static void EnqueueInvokedMethodInfos(IEnumerable<Instruction> instructions,
		Queue<InvokeMethodInfo> queue)
	{
		foreach (var instruction in instructions)
			if (instruction is Invoke invoke && invoke.MethodInfo.MethodName != Method.From)
				queue.Enqueue(invoke.MethodInfo);
	}

	protected static bool HasNumericPrint(IEnumerable<Instruction> instructions) =>
		instructions.OfType<PrintInstruction>().
			Any(print => print.ValueRegister.HasValue && !print.ValueIsText);

	public abstract Task<string> Compile(BinaryExecutable binary, Platform platform);
	public abstract string Extension { get; }
}