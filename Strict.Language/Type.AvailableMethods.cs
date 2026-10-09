using System.Collections.Concurrent;
using System.Text.RegularExpressions;
#if DEBUG
using System.Runtime.CompilerServices;
#endif

namespace Strict.Language;

public partial class Type : Context, IDisposable
{
	public Method? FindMethod(string methodName, IReadOnlyList<Expression> arguments,
		string? callText = null) =>
		typeMethodFinder.FindMethod(methodName, arguments, callText);

	public Method GetMethod(string methodName, IReadOnlyList<Expression> arguments) =>
		typeMethodFinder.GetMethod(methodName, arguments);

	/// <summary>
	/// Builds dictionary the first time we use it to access any method of this type or any of the
	/// member types recursively (if not there yet). Filtering is done by <see cref="FindMethod"/>
	/// </summary>
	public IReadOnlyDictionary<string, List<Method>> AvailableMethods
	{
		get
		{
			var cached = cachedAvailableMethods;
			if (cached != null)
				return cached;
			var activeBuilds = activeAvailableMethodBuilds ??= [];
			if (!activeBuilds.Add(this))
				return EmptyAvailableMethods;
			lock (availableMethodsLock)
			{
				try
				{
					cached = cachedAvailableMethods;
					if (cached != null)
						return cached;
					var built = new Dictionary<string, List<Method>>(StringComparer.Ordinal);
					foreach (var method in methods)
						if (method.IsPublic || method.Name == Method.From || method.Name.AsSpan().IsOperator())
							AddAvailableMethod(method, built);
					if (Name == Any)
						return cachedAvailableMethods = built;
					foreach (var member in Members.Where(m =>
						(m is { IsPublic: false, IsConstant: false, InitialValue: null } &&
							!IsTraitImplementation(m.Type)) || IsTraitCompositionMember(m)))
						AddNonGenericMethods(member.Type, built);
					if (!IsTrait && members.Count > 0 &&
						members.Any(m => !m.Type.IsGeneric && !m.IsConstant) &&
						methods.All(m => m.Name != Method.From))
					{
						var fromParser = methods.Count > 0
							? methods[0].Parser
							: savedParser ?? GetType(Any).Methods.FirstOrDefault()?.Parser;
						if (fromParser != null)
							AddFromConstructorWithMembersAsArguments(fromParser, built);
					}
					if (this is GenericTypeImplementation { Generic.IsDictionary: true } dictImpl &&
						dictImpl.Generic.AvailableMethods.TryGetValue(Method.From,
							out var genericFromMethods) &&
						built.TryGetValue(Method.From, out var existingFromMethods))
						foreach (var fromMethod in genericFromMethods)
							existingFromMethods.Add(new Method(fromMethod, dictImpl));
					AddAnyMethods(built);
					return cachedAvailableMethods = built;
				}
				finally
				{
					activeBuilds.Remove(this);
					if (activeBuilds.Count == 0)
						activeAvailableMethodBuilds = null;
				}
			}
		}
	}

	private void AddAvailableMethod(Method method, Dictionary<string, List<Method>> cache)
	{
		// From constructor methods should return the type we are in, not the base type (like Any)
		if (method.Name == Method.From && method.Type != this)
		{
			// If we already have a from constructor, do not add a default one from any base type (Any)
			if (cache.ContainsKey(Method.From))
				return;
			method = new Method(method, this);
		}
		if (cache.TryGetValue(method.Name, out var methodsWithThisName))
		{
			foreach (var existingMethod in methodsWithThisName)
				if (existingMethod.IsSameMethodNameReturnTypeAndParameters(method))
					return;
			methodsWithThisName.Add(method);
		}
		else
		{
			cache.Add(method.Name, [method]);
		}
	}

	protected void AddFromConstructorWithMembersAsArguments(ExpressionParser parser,
		Dictionary<string, List<Method>> cache) =>
		AddAvailableMethod(new Method(this, 0, parser, [
			"from(" + CreateFromMethodParameters() + ")",
			"\tvalue"
		]), cache);

	private string CreateFromMethodParameters()
	{
		var parameters = "";
		foreach (var member in members)
			if (!member.Type.IsGeneric && !member.IsConstant)
			{
				var memberType = member.Type.IsMutable
					? member.Type.GetFirstImplementation()
					: member.Type;
				parameters += (parameters == ""
					? ""
					: ", ") + member.Name.MakeFirstLetterLowercase() + (member.InitialValue != null
					? " = " + member.InitialValue
					: member.InitialValueText != null
						? " = " + member.InitialValueText
						: member.IsMutable && (memberType.IsNumber || memberType.IsBoolean || memberType.IsText)
							? " = " + GetDefaultValueForType(memberType.Name)
							: member.IsMutable
								? " " + memberType.Name
								: CanUseImplicitListParameterType(memberType, member.Name)
									? ""
									: " " + memberType.Name);
			}
		return parameters;
	}

	private static bool CanUseImplicitListParameterType(Type memberType, string memberName) =>
		memberType.IsList && memberType.Name.StartsWith(memberName.MakeFirstLetterUppercase(),
			StringComparison.Ordinal);

	private static string GetDefaultValueForType(string typeName) =>
		typeName switch
		{
			Number => "0",
			Boolean => "false",
			_ => "\"\""
		};

	public bool IsTraitImplementation(Type memberType) =>
		memberType.IsTrait && methods.Count >= memberType.Methods.Count &&
		memberType.Methods.All(typeMethod =>
			methods.Any(method => method.HasEqualSignature(typeMethod)));

	private void AddNonGenericMethods(Type implementType, Dictionary<string, List<Method>> cache)
	{
		foreach (var (_, otherMethods) in implementType.AvailableMethods)
			if (implementType.IsGeneric)
			{
				foreach (var otherMethod in otherMethods)
					if (!otherMethod.IsGeneric && !otherMethod.Parameters.Any(p => p.Type.IsGeneric))
						AddAvailableMethod(otherMethod, cache);
			}
			else
			{
				foreach (var otherMethod in otherMethods)
					if (otherMethod.Name != Method.From)
						AddAvailableMethod(otherMethod, cache);
			}
	}

	private void AddAnyMethods(Dictionary<string, List<Method>> cache)
	{
		cachedAnyMethods ??= GetType(Any).AvailableMethods;
		if (!IsGeneric)
			foreach (var (_, anyMethods) in cachedAnyMethods)
			foreach (var anyMethod in anyMethods)
				AddAvailableMethod(anyMethod, cache);
	}

	public sealed class NoMatchingMethodFound(Type type,
		string methodName,
		IReadOnlyDictionary<string, List<Method>> availableMethods) : Exception("\"" + methodName +
		"\" not found for " + type + ", available methods: " +
		string.Join(", ", availableMethods.Keys));

	public sealed class ArgumentsDoNotMatchMethodParameters(IReadOnlyList<Expression> arguments,
		Type type,
		IEnumerable<Method> allMethods,
		string? callText = null)
		: Exception(CreateArgumentsDoNotMatchMessage(arguments, type, allMethods, callText));

	private static string CreateArgumentsDoNotMatchMessage(IReadOnlyList<Expression> arguments,
		Type type, IEnumerable<Method> allMethods, string? callText) =>
		(callText == null
			? ""
			: "Call " + callText + " with ") + (arguments.Count == 0
			? (callText == null
				? "No"
				: "no") + " arguments does "
			: (arguments.Count == 1
				? "Argument: "
				: "Arguments: ") + string.Join(", ", arguments.Select(a => a.ToStringWithType())) +
			" do ") + "not match these " + type + " method(s):\n" + string.Join("\n", allMethods);
}
