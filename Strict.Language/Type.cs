namespace Strict.Language;

/// <summary>
/// .strict files contain a type or trait and must be in the correct namespace folder.
/// Strict code only contains optional implement, then has*, then methods*. No empty lines.
/// There is no typical lexing/scoping/token splitting needed as Strict syntax is very strict.
/// </summary>
public class Type : Context
{
	public Type(Package package, TypeLines file) : base(package, file.Name)
	{
		if (file.Lines.Length > Limit.LineCount)
			throw new LinesCountMustNotExceedLimit(this, file.Lines.Length);
		if (package.FindDirectType(Name) != null)
			throw new TypeAlreadyExistsInPackage(Name, package);
		package.Add(this);
		lines = file.Lines;
		IsGeneric = Name == Base.Generic || OneOfFirstThreeLinesContainsGeneric();
		CreatedBy = "Package: " + package + ", file=" + file;
		typeMethodFinder = new TypeMethodFinder(this);
		typeParser = new TypeParser(this, lines);
	}

	public sealed class LinesCountMustNotExceedLimit(Type type, int lineCount) : ParsingFailed(type,
		0, $"Type {type.Name} has lines count {lineCount} but limit is {Limit.LineCount}");

	public sealed class TypeAlreadyExistsInPackage(string name, Package package)
		: Exception(name + " in package: " + package);

	private readonly string[] lines;
	/// <summary>
	/// Generic types cannot be used directly as we don't know the implementation to be used (e.g.,
	/// a list, we need to know the type of the elements), you must them from
	/// <see cref="GenericTypeImplementation"/>!
	/// </summary>
	public bool IsGeneric { get; }
	/// <summary>
	/// For debugging purposes to see where this Type was initially created.
	/// </summary>
	public string CreatedBy { get; protected init; }
	private readonly TypeMethodFinder typeMethodFinder;
	private readonly TypeParser typeParser;

	private bool OneOfFirstThreeLinesContainsGeneric()
	{
		for (var line = 0; line < lines.Length && line < 3; line++)
		{
			if (HasGenericMember(lines[line]))
				return true;
			if (HasGenericMethodHeader(lines[line]) && line + 1 < lines.Length &&
				!lines[line + 1].StartsWith('\t'))
				return true;
		}
		return false;
	}

	private static bool HasGenericMember(string line) =>
		(line.StartsWith(HasWithSpaceAtEnd, StringComparison.Ordinal) ||
			line.StartsWith(MutableWithSpaceAtEnd, StringComparison.Ordinal)) &&
		(line.Contains(Base.Generic, StringComparison.Ordinal) ||
			line.Contains(Base.GenericLowercase, StringComparison.Ordinal));

	public const string HasWithSpaceAtEnd = Keyword.Has + " ";
	public const string MutableWithSpaceAtEnd = Keyword.Mutable + " ";
	public const string ConstantWithSpaceAtEnd = Keyword.Constant + " ";
	public const string EmptyBody = nameof(EmptyBody);

	private static bool HasGenericMethodHeader(string line) =>
		line.Contains(Base.Generic, StringComparison.Ordinal) ||
		line.Contains(Base.GenericLowercase, StringComparison.Ordinal);

	/// <summary>
	/// Parsing has to be done OUTSIDE the constructor as we first need all types and inside might not
	/// know all types yet needed for member assignments and method parsing (especially return types).
	/// </summary>
	public Type ParseMembersAndMethods(ExpressionParser parser)
	{
		typeParser.ParseMembersAndMethods(parser);
		lineNumber = typeParser.LineNumber;
		ValidateMethodAndMemberCountLimits();
		// ReSharper disable once ForCanBeConvertedToForeach, for performance reasons:
		// https://codeblog.jonskeet.uk/2009/01/29/for-vs-foreach-on-arrays-and-lists/
		for (var index = 0; index < members.Count; index++)
		{
			var trait = members[index].Type;
			if (trait.IsTrait)
				CheckIfTraitIsImplementedFullyOrNone(trait);
		}
		return this;
	}

	private int lineNumber;

	public sealed class MustImplementAllTraitMethodsOrNone(Type type, string traitName,
		IEnumerable<Method> missingTraitMethods) : ParsingFailed(type, type.lineNumber,
		"Trait Type:" + traitName + " Missing methods: " + string.Join(", ", missingTraitMethods));

	private void ValidateMethodAndMemberCountLimits()
	{
		var memberLimit = IsEnum
			? Limit.MemberCountForEnums
			: Limit.MemberCount;
		if (members.Count > memberLimit)
			throw new MemberCountShouldNotExceedLimit(this, memberLimit);
		if (IsDataType || IsEnum)
			return;
		if (methods.Count == 0 && members.Count < 2 && !IsNoneAnyOrBoolean() && Name != Base.Name)
			throw new NoMethodsFound(this, lineNumber);
		if (methods.Count > Limit.MethodCount && Package.Name != nameof(Base))
			throw new MethodCountMustNotExceedLimit(this);
	}

	/// <summary>
	/// Data types have no methods and just some data. Number, Text, and most Base types are not
	/// data types as they have functionality (which makes sense), only types higher up that only
	/// have data (like Color, which has 4 Numbers) are actually pure Data types!
	/// </summary>
	public bool IsDataType =>
		methods.Count == 0 && (members.Count > 1 || members is [{ Value: not null }]) ||
		Name == Base.Number;
	public bool IsEnum =>
		methods.Count == 0 && members.Count > 1 && members.All(m => m.IsConstant);

	public sealed class MemberCountShouldNotExceedLimit(Type type, int limit) : ParsingFailed(type,
		0, $"{type.Name} type has {type.members.Count} members, max: {limit}");

	private bool IsNoneAnyOrBoolean() => Name is Base.None or Base.Any or Base.Boolean or Base.Mutable;

	public sealed class NoMethodsFound(Type type, int lineNumber) : ParsingFailed(type, lineNumber,
		"Each type must have at least two members (datatypes and enums) or at least one method, " +
		"otherwise it is useless");

	public Package Package => (Package)Parent;

	public sealed class MethodCountMustNotExceedLimit(Type type) : ParsingFailed(type, 0,
		$"Type {type.Name} has method count {type.methods.Count} but limit is {Limit.MethodCount}");

	private void CheckIfTraitIsImplementedFullyOrNone(Type trait)
	{
		var nonImplementedTraitMethods = trait.Methods.Where(traitMethod =>
			traitMethod.Name != Method.From &&
			methods.All(implementedMethod => traitMethod.Name != implementedMethod.Name)).ToList();
		if (nonImplementedTraitMethods.Count > 0 && nonImplementedTraitMethods.Count !=
			trait.Methods.Count(traitMethod => traitMethod.Name != Method.From))
			throw new MustImplementAllTraitMethodsOrNone(this, trait.Name, nonImplementedTraitMethods);
	}

	public List<Member> Members => members;
	protected readonly List<Member> members = [];
	public List<Method> Methods => methods;
	protected readonly List<Method> methods = [];
	public bool IsTrait => Members.Count == 0 && Name != Base.Number && Name != Base.Boolean;
	public Dictionary<string, Type> AvailableMemberTypes
	{
		get
		{
			if (cachedAvailableMemberTypes != null)
				return cachedAvailableMemberTypes;
			cachedAvailableMemberTypes = new Dictionary<string, Type>();
			foreach (var member in members)
				if (cachedAvailableMemberTypes.TryAdd(member.Type.Name, member.Type))
					foreach (var (availableMemberName, availableMemberType) in member.Type.
						AvailableMemberTypes)
						cachedAvailableMemberTypes.TryAdd(availableMemberName, availableMemberType);
			return cachedAvailableMemberTypes;
		}
	}
	private Dictionary<string, Type>? cachedAvailableMemberTypes;

	public override Type? FindType(string name, Context? searchingFrom = null) =>
		name == Name || name.Contains('.') && name == base.ToString() || name is Other or Outer
			? this
			: Package.FindType(name, searchingFrom ?? this);

	/// <summary>
	/// Easy way to get another instance of the class type we are currently in.
	/// </summary>
	public const string Other = nameof(Other);
	/// <summary>
	/// In a for loop a different "value" is used, this way we can still get to the outer instance.
	/// </summary>
	public const string Outer = nameof(Outer);

	public GenericTypeImplementation GetGenericImplementation(params Type[] implementationTypes)
	{
		var key = GetImplementationName(implementationTypes);
		return GetGenericImplementation(key) ?? CreateGenericImplementation(key, implementationTypes);
	}

	internal string GetImplementationName(IEnumerable<Type> implementationTypes) =>
		Name + "(" + implementationTypes.ToWordList() + ")";

	internal string GetImplementationName(IEnumerable<NamedType> implementationTypes) =>
		Name + "(" + implementationTypes.Select(t => t.Name + " " + t.Type).ToWordList() + ")";

	private GenericTypeImplementation? GetGenericImplementation(string key)
	{
		if (!IsGeneric)
			throw new CannotGetGenericImplementationOnNonGeneric(Name, key);
		cachedGenericTypes ??= new Dictionary<string, GenericTypeImplementation>(StringComparer.Ordinal);
		return cachedGenericTypes.GetValueOrDefault(key);
	}

	private Dictionary<string, GenericTypeImplementation>? cachedGenericTypes;

	/// <summary>
	/// Most often called for List (or the Iterator trait), which we want to optimize for
	/// </summary>
	private GenericTypeImplementation CreateGenericImplementation(string key,
		IReadOnlyList<Type> implementationTypes)
	{
		if (Name is Base.List or Base.Iterator or Base.Mutable && implementationTypes.Count == 1 ||
			GetGenericTypeArguments().Count == implementationTypes.Count ||
			HasMatchingConstructor(implementationTypes))
		{
			var genericType = new GenericTypeImplementation(this, implementationTypes);
			cachedGenericTypes!.Add(key, genericType);
			return genericType;
		}
		throw new TypeArgumentsCountDoesNotMatchGenericType(this, implementationTypes);
	}

	private bool HasMatchingConstructor(IReadOnlyList<Type> implementationTypes) =>
		typeMethodFinder.FindMethod(Method.From, implementationTypes) != null;

	public sealed class CannotGetGenericImplementationOnNonGeneric(string name, string key)
		: Exception("Type: " + name + ", Generic Implementation: " + key);

	public string FilePath => Path.Combine(Package.FolderPath, Name) + Extension;
	public const string Extension = ".strict";
	public Member? FindMember(string name) => Members.FirstOrDefault(member => member.Name == name);

	public Method? FindMethod(string methodName, IReadOnlyList<Expression> arguments) =>
		typeMethodFinder.FindMethod(methodName, arguments);

	public Method GetMethod(string methodName, IReadOnlyList<Expression> arguments) =>
		typeMethodFinder.GetMethod(methodName, arguments);

	public class GenericTypesCannotBeUsedDirectlyUseImplementation(Type type,
		string extraInformation) : Exception(type + " " + extraInformation);

	/// <summary>
	/// Any non-public member is automatically iterable if it has Iterator, for example,
	/// Text.strict or Error.strict have public members you have to iterate over yourself.
	/// If there are two private iterators, then pick the first member automatically.
	/// Any number is also iteratable, most iterators are just List(ofSomeType)
	/// </summary>
	public bool IsIterator =>
		Name == Base.Iterator || Name.StartsWith(Base.Iterator + "(", StringComparison.Ordinal) ||
		HasAnyIteratorMember();

	private bool HasAnyIteratorMember()
	{
		if (cachedIteratorResult != null)
			return cachedIteratorResult.Value;
		cachedIteratorResult = ExecuteIsIteratorCheck();
		return (bool)cachedIteratorResult;
	}

	private bool ExecuteIsIteratorCheck()
	{
		foreach (var member in members)
		{
			if (cachedEvaluatedMemberTypes.TryGetValue(member.Type.Name, out var result))
				return result;
			var isIterator = member is { IsPublic: false, Type.IsIterator: true };
			cachedEvaluatedMemberTypes.Add(member.Type.Name, isIterator);
			if (isIterator)
				return true;
		}
		return false;
	}

	private bool? cachedIteratorResult;
	private readonly Dictionary<string, bool> cachedEvaluatedMemberTypes = new();

	/// <summary>
	/// Can OUR type be converted to sameOrUpcastableType and be used as such? Be careful how this is
	/// called. A derived RedApple can be used as the base class Apple, but not the other way around.
	/// </summary>
	public bool IsSameOrCanBeUsedAs(Type sameOrUsableType, int maxDepth = 2)
	{
		if (this == sameOrUsableType)
			return true;
		if (members.Count(m => m.Type == sameOrUsableType) == 1)
			return true;
		if (IsMutableAndHasMatchingInnerType(sameOrUsableType))
			return true;
		if (IsCompatibleOneOfType(sameOrUsableType))
			return true;
		if (IsEnum && members[0].Type.IsSameOrCanBeUsedAs(sameOrUsableType))
			return true;
		return maxDepth >= 0 && Members.Count(m =>
			m.Type.IsSameOrCanBeUsedAs(sameOrUsableType, maxDepth - 1)) == 1;
	}

	internal bool IsMutableAndHasMatchingInnerType(Type argumentType) =>
		this is GenericTypeImplementation { Generic.Name: Base.Mutable } genericTypeImplementation &&
		genericTypeImplementation.ImplementationTypes[0].IsSameOrCanBeUsedAs(argumentType);

	private bool IsCompatibleOneOfType(Type sameOrBaseType) =>
		sameOrBaseType is OneOfType oneOfType && oneOfType.Types.Any(t => IsSameOrCanBeUsedAs(t));

	/// <summary>
	/// When two types are using in a conditional expression, i.e., then and else return types and
	/// both are not based on each other, find the common base type that works for both.
	/// </summary>
	public Type? FindFirstUnionType(Type elseType)
	{
		foreach (var member in members)
			if (elseType.members.Any(otherMember => otherMember.Type == member.Type))
				return member.Type;
		foreach (var member in members)
		{
			if (member.Type == this)
				continue;
			var subUnionType = member.Type.FindFirstUnionType(elseType);
			if (subUnionType != null)
				return subUnionType;
		}
		foreach (var otherMember in elseType.members)
		{
			var otherSubUnionType = otherMember.Type.FindFirstUnionType(this);
			if (otherSubUnionType != null)
				return otherSubUnionType;
		}
		return null;
	}

	/// <summary>
	/// Builds dictionary the first time we use it to access any method of this type or any of the
	/// member types recursively (if not there yet). Filtering has to be done by <see cref="FindMethod"/>
	/// </summary>
	public IReadOnlyDictionary<string, List<Method>> AvailableMethods
	{
		get
		{
			if (cachedAvailableMethods is { Count: > 0 })
				return cachedAvailableMethods;
			cachedAvailableMethods = new Dictionary<string, List<Method>>(StringComparer.Ordinal);
			foreach (var method in methods)
				if (method.IsPublic || method.Name == Method.From || method.Name.AsSpan().IsOperator())
					AddAvailableMethod(method);
			if (Name == Base.Any)
				return cachedAvailableMethods;
			// Types are composed in Strict, we want users to be able to use base methods but exclude
			// public members (e.g., Type.Name), constants (e.g., constant Tab = Character(7)) and if we
			// have implemented a trait here anyway (then all the methods are already implemented).
			foreach (var member in Members.Where(m =>
				m is { IsPublic: false, Value: null } && !IsTraitImplementation(m.Type)))
				AddNonGenericMethods(member.Type);
			if (members.Count > 0 && members.Any(m => !m.Type.IsGeneric))
				AddFromConstructorWithMembersAsArguments();
			AddAnyMethods();
			return cachedAvailableMethods;
		}
	}
	public int AutogeneratedEnumValue { get; internal set; }
	private Dictionary<string, List<Method>>? cachedAvailableMethods;

	private void AddAvailableMethod(Method method)
	{
		// From constructor methods should return the type we are in, not the base type (like Any)
		if (method.Name == Method.From && method.Type != this)
		{
			// If we already have a from constructor and members that need initialization (which is how we
			// got here, a member was without value), do not add the default from constructor from Any!
			if (method.Parameters.Count == 0 && cachedAvailableMethods!.ContainsKey(method.Name))
				return;
			method = method.CloneFrom(this);
		}
		if (cachedAvailableMethods!.ContainsKey(method.Name))
		{
			var methodsWithThisName = cachedAvailableMethods[method.Name];
			foreach (var existingMethod in methodsWithThisName)
				if (existingMethod.IsSameMethodNameReturnTypeAndParameters(method))
					return;
			methodsWithThisName.Add(method);
		}
		else
			cachedAvailableMethods.Add(method.Name, [method]);
	}

	private void AddFromConstructorWithMembersAsArguments() =>
		AddAvailableMethod(new Method(this, 0, methods.Count > 0
			? methods[0].Parser
			: GetType(Base.Any).methods[0].Parser, ["from(" + CreateFromMethodParameters() + ")"]));

	private string CreateFromMethodParameters()
	{
		if (IsEnum)
			return "number";
		var parameters = "";
		foreach (var member in members)
			if (!member.Type.IsGeneric)
				parameters +=
					(parameters == ""
						? ""
						: ", ") +
					member.Name.MakeFirstLetterLowercase() +
					(member.Value != null
						? " = " + member.Value
						: member.Type.Name == Base.List
							? ""
							: " " + member.Type.Name);
		return parameters;
	}

	public bool IsTraitImplementation(Type memberType) =>
		memberType.IsTrait && methods.Count >= memberType.Methods.Count &&
		memberType.Methods.All(typeMethod =>
			methods.Any(method => method.HasEqualSignature(typeMethod)));

	private void AddNonGenericMethods(Type implementType)
	{
		foreach (var (_, otherMethods) in implementType.AvailableMethods)
			if (implementType.IsGeneric)
			{
				foreach (var otherMethod in otherMethods)
					if (!otherMethod.IsGeneric && !otherMethod.Parameters.Any(p => p.Type.IsGeneric))
						AddAvailableMethod(otherMethod);
			}
			else
				foreach (var otherMethod in otherMethods)
					if (otherMethod.Name != Method.From)
						AddAvailableMethod(otherMethod);
	}

	private void AddAnyMethods()
	{
		if (cachedAnyMethods is { Count: 0 })
			cachedAnyMethods = null;
		cachedAnyMethods ??= GetType(Base.Any).AvailableMethods;
		if (!IsGeneric)
			foreach (var (_, anyMethods) in cachedAnyMethods)
			foreach (var anyMethod in anyMethods)
				AddAvailableMethod(anyMethod);
	}

	private static IReadOnlyDictionary<string, List<Method>>? cachedAnyMethods;

	public sealed class NoMatchingMethodFound(Type type, string methodName,
		IReadOnlyDictionary<string, List<Method>> availableMethods) : Exception(methodName +
		" not found for " + type + ", available methods: " + availableMethods.Keys.ToWordList());

	public sealed class ArgumentsDoNotMatchMethodParameters(IReadOnlyList<Expression> arguments,
		Type type, IEnumerable<Method> allMethods) : Exception((arguments.Count == 0
			? "No arguments does "
			: (arguments.Count == 1
				? "Argument: "
				: "Arguments: ") + arguments.Select(a => a.ToStringWithType()).ToWordList() + " do ") +
		"not match these " + type + " method(s):\n" + string.Join("\n", allMethods));

	public bool IsUpcastable(Type otherType) =>
		IsEnum && otherType.IsEnum && otherType.Members.Any(member =>
			member.Name.Equals(Name, StringComparison.OrdinalIgnoreCase));

	public int CountMemberUsage(string memberName) =>
		lines.Count(line => line.Contains(" " + memberName) || line.Contains("(" + memberName));

	public HashSet<NamedType> GetGenericTypeArguments()
	{
		if (!IsGeneric)
			throw new NotSupportedException("This type " + this +
				" must be generic in order to call this method!");
		var genericArguments = new HashSet<NamedType>();
		foreach (var member in Members)
			if (member.Type is GenericType genericType)
				foreach (var namedType in genericType.GenericImplementations)
					genericArguments.Add(namedType);
			else if (member.Type.Name == Base.List || member.Type.IsIterator)
				genericArguments.Add(new Parameter(this, Base.Generic));
			else if (member.Type.IsGeneric)
				genericArguments.Add(member);
		if (genericArguments.Count == 0)
			throw new InvalidGenericTypeWithoutGenericArguments(this);
		//tst: Console.WriteLine(this + " GetGenericTypeArguments: " + genericArguments.ToWordList());
		return genericArguments;
	}

	public sealed class InvalidGenericTypeWithoutGenericArguments(Type type) : Exception(
		"This type is broken and needs to be fixed, check the creation: " + type + ", CreatedBy: " +
		type.CreatedBy);

	public sealed class TypeHasNoMembersAndThusMustBeATraitWithoutMethodBodies(Type type)
		: ParsingFailed(type, 0);

	/// <summary>
	/// Helper for method parameters default values, which don't have a methodBody to parse, but
	/// we still need some basic parsing to assign default values.
	/// </summary>
	public Expression GetMemberExpression(ExpressionParser parser, string memberName,
		string remainingTextSpan) =>
		typeParser.GetMemberExpression(parser, memberName, remainingTextSpan);
}