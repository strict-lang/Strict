#if DEBUG
using System.Runtime.CompilerServices;
#endif

namespace Strict.Language;

/// <summary>
/// .strict files contain a type or trait and must be in the correct namespace folder.
/// Strict code only contains optional implement, then has*, then methods*. No empty lines.
/// There is no typical lexing/scoping/token splitting needed as Strict syntax is very strict.
/// </summary>
public class Type : Context, IDisposable
{
	/// <summary>
	/// Has no implementation and is used for void, empty, or none, which is not valid to assign.
	/// </summary>
	public const string None = nameof(None);
	/// <summary>
	/// Defines all the methods available in any type (everything automatically implements **Any**).
	/// These methods don't have to be implemented by any class, they are automatically implemented.
	/// </summary>
	public const string Any = nameof(Any);
	/// <summary>
	/// Most basic type: can only be true or false, any statement must either be None or return a
	/// Boolean (anything else is a compiler error). Any statement returning false (like a failing
	/// test) will also immediately cause an error at runtime or in the Editor via SCrunch.
	/// </summary>
	public const string Boolean = nameof(Boolean);
	/// <summary>
	/// Can be any floating point or integer number (think byte, short, int, long, float, or double
	/// in other languages). Also, it can be a decimal or BigInteger, the compiler can decide and
	/// optimize this away into anything that makes sense in the current context.
	/// </summary>
	public const string Number = nameof(Number);
	public const string Character = nameof(Character);
	public const string HashCode = nameof(HashCode);
	public const string Range = nameof(Range);
	public const string Text = nameof(Text);
	public const string Error = nameof(Error);
	public const string ErrorWithValue = nameof(ErrorWithValue);
	public const string Iterator = nameof(Iterator);
	public const string List = nameof(List);
	public const string Logger = nameof(Logger);
	public const string App = nameof(App);
	public const string System = nameof(System);
	public const string File = nameof(File);
	public const string Directory = nameof(Directory);
	public const string TextWriter = nameof(TextWriter);
	public const string TextReader = nameof(TextReader);
	public const string Stacktrace = nameof(Stacktrace);
	public const string Mutable = nameof(Mutable);
	public const string Dictionary = nameof(Dictionary);
#if DEBUG
	public Type(Package package, TypeLines file, [CallerFilePath] string callerFilePath = "",
		[CallerLineNumber] int callerLineNumber = 0,
		[CallerMemberName] string callerMemberName = "") : base(package, file.Name, callerFilePath,
		callerLineNumber, callerMemberName)
#else
	public Type(Package package, TypeLines file) : base(package, file.Name)
#endif
	{
		if (file.Lines.Length > Limit.LineCount)
			throw new LinesCountMustNotExceedLimit(this, file.Lines.Length);
		var existingType = package.FindDirectType(Name);
		if (existingType != null)
			throw new TypeAlreadyExistsInPackage(Name, package, existingType);
		package.Add(this);
		Lines = file.Lines;
		IsGeneric = Name == GenericUppercase || OneOfFirstThreeLinesContainsGeneric();
		IsMutable = Name.StartsWith(Mutable, StringComparison.Ordinal);
		typeMethodFinder = new TypeMethodFinder(this);
		typeParser = new TypeParser(this, Lines);
		typeKind = GetTypeKindFromName();
	}

	public sealed class LinesCountMustNotExceedLimit(Type type, int lineCount) : ParsingFailed(type,
		lineCount, $"Type {type.Name} has lines count {lineCount} but limit is {Limit.LineCount}");

	public sealed class TypeAlreadyExistsInPackage(string name, Package package, Type existingType)
		: Exception(name + " in package: " + package + ", existing type : " + existingType
#if DEBUG
			+ ", existing type created by " + existingType.callerFilePath + ":" +
			existingType.callerLineNumber + " from method " + existingType.callerMemberName
#endif
		);

	internal string[] Lines { get; }
	/// <summary>
	/// Generic types cannot be used directly as we don't know the implementation to be used (e.g.,
	/// a list, we need to know the type of the elements), you must them from
	/// <see cref="GenericTypeImplementation"/>!
	/// </summary>
	public bool IsGeneric { get; }
	/// <summary>
	/// Mutable types should be avoided as they make code non parallel and slow. Mutable types have
	/// always an inner type (e.g., Mutable(Text)), use GetFirstImplementation to get to that type.
	/// The TypeKind of the inner type will be mirrored here to make comparisons fast.
	/// </summary>
	public bool IsMutable { get; }
	private readonly TypeMethodFinder typeMethodFinder;
	private readonly TypeParser typeParser;
	internal TypeKind typeKind;

	private bool OneOfFirstThreeLinesContainsGeneric()
	{
		for (var line = 0; line < Lines.Length && line < 3; line++)
			if (HasGenericMember(Lines[line]) || HasGenericMethodHeader(Lines[line]) &&
				line + 1 < Lines.Length && !Lines[line + 1].StartsWith('\t'))
				return true;
		return false;
	}

	private TypeKind GetTypeKindFromName() =>
		Name switch
		{
			None => TypeKind.None,
			Boolean => TypeKind.Boolean,
			Number => TypeKind.Number,
			Text => TypeKind.Text,
			Character => TypeKind.Character,
			List => TypeKind.List,
			Dictionary => TypeKind.Dictionary,
			Error => TypeKind.Error,
			ErrorWithValue => TypeKind.Error,
			Iterator => TypeKind.Iterator,
			nameof(Name) => TypeKind.Text,
			Any => TypeKind.Any,
			_ => TypeKind.Unknown
		};

	private static bool HasGenericMember(string line) =>
		(line.StartsWith(HasWithSpaceAtEnd, StringComparison.Ordinal) ||
			line.StartsWith(MutableWithSpaceAtEnd, StringComparison.Ordinal)) &&
		(line.Contains(GenericUppercase, StringComparison.Ordinal) ||
			line.Contains(GenericLowercase, StringComparison.Ordinal));

	public const string HasWithSpaceAtEnd = Keyword.Has + " ";
	public const string MutableWithSpaceAtEnd = Keyword.Mutable + " ";
	public const string ConstantWithSpaceAtEnd = Keyword.Constant + " ";

	private static bool HasGenericMethodHeader(string line) =>
		line.Contains(GenericUppercase, StringComparison.Ordinal) ||
		line.Contains(GenericLowercase, StringComparison.Ordinal);

	/// <summary>
	/// Parsing has to be done OUTSIDE the constructor as we first need all types and inside might not
	/// know all types yet needed for member assignments and method parsing (especially return types).
	/// </summary>
	public Type ParseMembersAndMethods(ExpressionParser parser)
	{
		if (typeParser.LineNumber >= 0)
			throw new TypeWasAlreadyParsed(this); //ncrunch: no coverage
		typeParser.ParseMembersAndMethods(parser);
		DetermineEnumTypeKind();
		//if ((Name.StartsWith("List") || Name.StartsWith("Mutable(List")) && typeKind != TypeKind.List)
		//	throw new NotSupportedException("Something went wrong with this list creation!");
		ValidateMethodAndMemberCountLimits();
		// ReSharper disable once ForCanBeConvertedToForeach, for performance reasons:
		// https://codeblog.jonskeet.uk/2009/01/29/for-vs-foreach-on-arrays-and-lists/
		for (var index = 0; index < members.Count; index++)
		{
			var trait = members[index].Type;
			if (trait.typeParser.LineNumber > 0 && trait.IsTrait)
				CheckIfTraitIsImplementedFullyOrNone(trait);
		}
		return this;
	}

	private void DetermineEnumTypeKind()
	{
		if (methods.Count == 0 && members.Count > 0)
		{
			for (var i = 0; i < members.Count; i++)
				if (!members[i].IsConstant && !members[i].Type.IsEnum)
					return;
			typeKind = TypeKind.Enum;
		}
	}

	public class TypeWasAlreadyParsed(Type type) : Exception(type.ToString()); //ncrunch: no coverage

	public sealed class MustImplementAllTraitMethodsOrNone(Type type, string traitName,
		IEnumerable<Method> missingTraitMethods) : ParsingFailed(type, type.typeParser.LineNumber,
		"Trait Type:" + traitName + " Missing methods: " + string.Join(", ", missingTraitMethods));

	private void ValidateMethodAndMemberCountLimits()
	{
		var memberLimit = IsEnum
			? Limit.MemberCountForEnums
			: Limit.MemberCount;
		if (members.Count > memberLimit)
			throw new MemberCountShouldNotExceedLimit(this, memberLimit);
		if (IsEnum || IsDataType || IsMutable)
			return;
		if (typeKind == TypeKind.Unknown && methods.Count == 0 && members.Count < 2)
			throw new NoMethodsFound(this, typeParser.LineNumber);
		if (methods.Count > Limit.MethodCount && (Package.Name != nameof(Strict) &&
			Package.Name != "TestPackage" || Name == "MethodCountMustNotExceedFifteen"))
			throw new MethodCountMustNotExceedLimit(this);
	}

	public bool IsEnum => typeKind == TypeKind.Enum;
	/// <summary>
	/// Data types have no methods and just some data. Number, Text, and most Base types are not
	/// data types as they have functionality (which makes sense), only types higher up that only
	/// have data (like Color, which has 4 Numbers) are actually pure Data types!
	/// </summary>
	public bool IsDataType =>
		CheckIfParsed() && methods.Count == 0 &&
		(members.Count > 1 || members is [{ InitialValue: not null }]) || Name == Number ||
		Name == nameof(Name);

	private bool CheckIfParsed()
	{
		if (!IsGeneric && Lines.Length > 1 && typeParser.LineNumber == -1)
			throw new TypeIsNotParsedCallParseMembersAndMethods(this); //ncrunch: no coverage
		return true;
	}

	private sealed class TypeIsNotParsedCallParseMembersAndMethods(Type type)
		: Exception(type.ToString()); //ncrunch: no coverage

	public sealed class MemberCountShouldNotExceedLimit(Type type, int limit) : ParsingFailed(type,
		0, $"{type.Name} type has {type.members.Count} members, max: {limit}");

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
	public bool IsTrait => !IsNumber && !IsBoolean && CheckIfParsed() && Members.Count == 0;
	public Dictionary<string, Type> AvailableMemberTypes
	{
		get
		{
			if (CheckIfParsed() && field != null)
				return field;
			field = new Dictionary<string, Type>();
			foreach (var member in members)
				if (field.TryAdd(member.Type.Name, member.Type))
					foreach (var (availableMemberName, availableMemberType) in member.Type.
						AvailableMemberTypes)
						field.TryAdd(availableMemberName, availableMemberType);
			return field;
		}
	}

	public override Type? FindType(string name, Context? searchingFrom = null) =>
		name == Name || name is Other or Outer || name == FullName
			? this
			: Package.FindType(name, searchingFrom ?? this);

	/// <summary>
	/// Everything internally is Any, cannot be specified as member, parameter, or variable.
	/// </summary>
	public const string AnyLowercase = "any";
	public const string GenericUppercase = "Generic";
	public const string GenericLowercase = "generic";
	public const string IteratorLowercase = "iterator";
	public const string ElementsLowercase = "elements";
	public const string ValueLowercase = "value";
	public const string IndexLowercase = "index";
	/// <summary>
	/// Easy way to get another instance of the class type we are currently in.
	/// </summary>
	public const string Other = nameof(Other);
	/// <summary>
	/// In a for loop a different "value" is used, this way we can still get to the outer instance.
	/// </summary>
	public const string Outer = nameof(Outer);
	public const string OuterLowercase = "outer";

	public GenericTypeImplementation GetGenericImplementation(params Type[] implementationTypes)
	{
		var key = GetImplementationName(implementationTypes);
		return GetGenericImplementation(key) ?? CreateGenericImplementation(key, implementationTypes);
	}

	internal string GetImplementationName(Type[] implementationTypes)
	{
		var key = "";
		for (var i = 0; i < implementationTypes.Length; i++)
			key += (key == ""
				? ""
				: ", ") + implementationTypes[i].Name;
		return Name + "(" + key + ")";
	}

	internal string GetImplementationName(IReadOnlyList<NamedType> implementationTypes)
	{
		var key = "";
		for (var i = 0; i < implementationTypes.Count; i++)
			key += (key == ""
				? ""
				: ", ") + implementationTypes[i];
		return Name + "(" + key + ")";
	}

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
		Type[] implementationTypes)
	{
		if ((IsList || IsIterator || IsMutable) && implementationTypes.Length == 1 ||
			GetGenericTypeArguments().Count == implementationTypes.Length ||
			HasMatchingConstructor(implementationTypes))
		{
			//TODO: this is extemely slow
			var genericType = new GenericTypeImplementation(this, implementationTypes);
			cachedGenericTypes!.Add(key, genericType);
			return genericType;
		}
		throw new TypeArgumentsCountDoesNotMatchGenericType(this, implementationTypes);
	}

	private bool HasMatchingConstructor(IReadOnlyList<Type> implementationTypes) =>
		typeMethodFinder.FindFromMethodImplementation(implementationTypes) != null;

	public sealed class CannotGetGenericImplementationOnNonGeneric(string name, string key)
		: Exception("Type: " + name + ", Generic Implementation: " + key);

	public string FilePath =>
		Path.GetFullPath(Path.Combine(
			Package.FolderPath ?? Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg, "Strict"),
			(this is GenericTypeImplementation genericType
				? genericType.Generic.Name
				: Name) + Extension));
	public const string Extension = ".strict";

	public Member? FindMember(string name)
	{
		CheckIfParsed();
		return Members.FirstOrDefault(member => member.Name == name);
	}

	public Method? FindMethod(string methodName, IReadOnlyList<Expression> arguments) =>
		typeMethodFinder.FindMethod(methodName, arguments);

	public Method GetMethod(string methodName, IReadOnlyList<Expression> arguments) =>
		typeMethodFinder.GetMethod(methodName, arguments);

	public class GenericTypesCannotBeUsedDirectlyUseImplementation(Type type,
		string extraInformation) : Exception(type + " " + extraInformation);

	/// <summary>
	/// Any non-public member is automatically iterable if it has Iterator, for example, Text.strict
	/// or Error.strict have public members you have to iterate over yourself. If there are more
	/// private iterators, pick the first member automatically. List and number are also iterable.
	/// </summary>
	public bool IsIterator =>
		typeKind == TypeKind.Iterator ||
		Name.StartsWith(Iterator + "(", StringComparison.Ordinal) || HasAnyIteratorMember();

	private bool HasAnyIteratorMember()
	{
		if (cachedIteratorResult != null)
			return cachedIteratorResult.Value;
		cachedIteratorResult = ExecuteIsIteratorCheck();
		return (bool)cachedIteratorResult;
	}

	private bool ExecuteIsIteratorCheck()
	{
		CheckIfParsed();
		foreach (var member in members)
		{
			if (cachedEvaluatedMemberTypes.TryGetValue(member.Type.Name, out var result))
				return result; //ncrunch: no coverage
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
	/// Can OUR type be converted to sameOrUsableType and be used as such? Be careful how this is
	/// called. A derived RedApple can be used as the base class Apple, but not the other way around.
	/// </summary>
	public bool IsSameOrCanBeUsedAs(Type sameOrUsableType, bool allowImplicitConversion = true,
		int maxDepth = 2)
	{
		if (this == sameOrUsableType || sameOrUsableType.IsAny || typeKind < TypeKind.List &&
			typeKind == sameOrUsableType.typeKind)
			return true;
		if (allowImplicitConversion && IsImplicitToConversion(sameOrUsableType))
			return true;
		if (IsEnum && members[0].Type.IsSameOrCanBeUsedAs(sameOrUsableType))
			return true;
		if (IsMutable && GetFirstImplementation().IsSameOrCanBeUsedAs(sameOrUsableType) ||
			sameOrUsableType.IsMutable && IsSameOrCanBeUsedAs(sameOrUsableType.GetFirstImplementation()))
			return true;
		if (HasExactlyOneMemberOfType(sameOrUsableType))
			return true;
		if (IsCompatibleOneOfType(sameOrUsableType))
			return true;
		return maxDepth >= 0 &&
			HasExactlyOneUsableMember(sameOrUsableType, allowImplicitConversion, maxDepth);
	}

	private bool HasExactlyOneMemberOfType(Type targetType)
	{
		// Basically members.Count(m => m.Type == targetType) == 1, but more performant
		var found = false;
		foreach (var m in members)
			if (m.Type == targetType)
			{
				if (found)
					return false;
				found = true;
			}
		return found;
	}

	private bool HasExactlyOneUsableMember(Type targetType, bool allowImplicitConversion, int maxDepth)
	{
		var found = false;
		foreach (var m in Members)
			if (!m.IsConstant &&
				m.Type.IsSameOrCanBeUsedAs(targetType, allowImplicitConversion, maxDepth - 1))
			{
				if (found)
					return false;
				found = true;
			}
		return found;
	}

	/// <summary>
	/// Only allow implicit conversions as defined in Any.strict (to Text, to Type, to HashCode)
	/// </summary>
	private static bool IsImplicitToConversion(Context targetType) =>
		targetType.Name is Text or nameof(Type) or HashCode;

	private bool IsCompatibleOneOfType(Type sameOrBaseType)
	{
		if (sameOrBaseType is OneOfType oneOfType)
			for (var index = 0; index < oneOfType.Types.Length; index++)
				if (IsSameOrCanBeUsedAs(oneOfType.Types[index]))
					return true;
		return false;
	}

	/// <summary>
	/// When two types are using in a conditional expression, i.e., then and else return types and
	/// both are not based on each other, find the common base type that works for both.
	/// </summary>
	public Type? FindFirstUnionType(Type elseType)
	{
		if (elseType.IsError)
			return this;
		if (IsError)
			return elseType;
		// Allow number and iterators for return types
		if (Name == Number && elseType.IsIterator)
			return elseType;
		if (elseType.IsNumber && IsIterator)
			return this;
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
	/// member types recursively (if not there yet). Filtering is done by <see cref="FindMethod"/>
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
			if (Name == Any)
				return cachedAvailableMethods;
			// Types are composed in Strict, we want users to be able to use base methods but exclude
			// public members (e.g., Type.Name), constants (e.g., constant Tab = Character(7)) and if we
			// have implemented a trait here anyway (then all the methods are already implemented).
			foreach (var member in Members.Where(m =>
				m is { IsPublic: false, InitialValue: null } && !IsTraitImplementation(m.Type)))
				AddNonGenericMethods(member.Type);
			if (members.Count > 0 && members.Any(m => !m.Type.IsGeneric && !m.IsConstant) &&
				methods.All(m => m.Name != Method.From))
				AddFromConstructorWithMembersAsArguments(methods.Count > 0
					? methods[0].Parser
					: GetType(Any).AvailableMethods.First().Value[0].Parser);
			if (this is GenericTypeImplementation { Generic.IsDictionary: true } dictImpl &&
				dictImpl.Generic.AvailableMethods.TryGetValue(Method.From, out var genericFromMethods) &&
				cachedAvailableMethods!.TryGetValue(Method.From, out var existingFromMethods))
				foreach (var fromMethod in genericFromMethods)
					existingFromMethods.Add(new Method(fromMethod, dictImpl));
			AddAnyMethods();
			return cachedAvailableMethods;
		}
	}
	public int AutogeneratedEnumValue { get; internal set; }
	public int LineNumber => typeParser.LineNumber;
	private Dictionary<string, List<Method>>? cachedAvailableMethods;

	private void AddAvailableMethod(Method method)
	{
		// From constructor methods should return the type we are in, not the base type (like Any)
		if (method.Name == Method.From && method.Type != this)
		{
			// If we already have a from constructor, do not add a default one from any base type (Any)
			if (cachedAvailableMethods!.ContainsKey(Method.From))
				return;
			method = new Method(method, this);
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

	protected void AddFromConstructorWithMembersAsArguments(ExpressionParser parser) =>
		AddAvailableMethod(new Method(this, 0, parser, [
			"from(" + CreateFromMethodParameters() + ")",
			"\tvalue"
		]));

	private string CreateFromMethodParameters()
	{
		var parameters = "";
		foreach (var member in members)
			if (!member.Type.IsGeneric && !member.IsConstant)
				parameters +=
					(parameters == ""
						? ""
						: ", ") +
					member.Name.MakeFirstLetterLowercase() +
					(member.InitialValue != null
						? " = " + member.InitialValue
						: member.Type.Name == List
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
		cachedAnyMethods ??= GetType(Any).AvailableMethods;
		if (!IsGeneric)
			foreach (var (_, anyMethods) in cachedAnyMethods)
			foreach (var anyMethod in anyMethods)
				AddAvailableMethod(anyMethod);
	}

	private static IReadOnlyDictionary<string, List<Method>>? cachedAnyMethods;

	public sealed class NoMatchingMethodFound(Type type, string methodName,
		IReadOnlyDictionary<string, List<Method>> availableMethods) : Exception("\"" + methodName +
		"\" not found for " + type + ", available methods: " + string.Join(", ", availableMethods.Keys));

	public sealed class ArgumentsDoNotMatchMethodParameters(IReadOnlyList<Expression> arguments,
		Type type, IEnumerable<Method> allMethods) : Exception((arguments.Count == 0
			? "No arguments does "
			: (arguments.Count == 1
				? "Argument: "
				: "Arguments: ") + string.Join(", ", arguments.Select(a => a.ToStringWithType())) + " do ") +
		"not match these " + type + " method(s):\n" + string.Join("\n", allMethods));

	public bool IsUpcastable(Type otherType) =>
		IsEnum && otherType.IsEnum && otherType.Members.Any(member =>
			member.Name.Equals(Name, StringComparison.OrdinalIgnoreCase));

	public int CountMemberUsage(string memberName) =>
		Lines.Count(line => line.Contains(' ' + memberName) || line.Contains('\t' + memberName) ||
			line.Contains('(' + memberName));

	[Log]
	public HashSet<NamedType> GetGenericTypeArguments()
	{
		if (!IsGeneric)
			throw new TypeMustBeGenericToCallThis(this); //ncrunch: no coverage
		var genericArguments = new HashSet<NamedType>();
		foreach (var member in Members)
			if (member.Type is GenericType genericType)
				foreach (var namedType in genericType.GenericImplementations)
					genericArguments.Add(namedType);
			else if (member.Type.IsList || member.Type.IsIterator)
				genericArguments.Add(new Parameter(this, GenericUppercase));
			else if (member.Type.IsGeneric)
				genericArguments.Add(member);
		return genericArguments.Count == 0
			? throw new InvalidGenericTypeWithoutGenericArguments(this)
			: genericArguments;
	}

	//ncrunch: no coverage start
	public sealed class TypeMustBeGenericToCallThis(Type type) : Exception(type.FullName);

	public sealed class InvalidGenericTypeWithoutGenericArguments(Type type) : Exception(
		"This type is broken and needs to be fixed, check the creation: " + type + ", Package: " +
		type.Package + ", file=" + type.FilePath);
	//ncrunch: no coverage end

	public sealed class TypeHasNoMembersAndThusMustBeATraitWithoutMethodBodies(Type type)
		: ParsingFailed(type, 0);

	/// <summary>
	/// Helper for method parameters default values, which don't have a methodBody to parse, but
	/// we still need some basic parsing to assign default values.
	/// </summary>
	internal Expression GetMemberExpression(ExpressionParser parser, string memberName,
		string remainingTextSpan, int typeLineNumber) =>
		typeParser.GetMemberExpression(parser, memberName, remainingTextSpan, typeLineNumber);

	public bool IsNone => typeKind == TypeKind.None;
	/// <summary>
	/// Is this a boolean or if OneOfType, is one of the types a boolean? Used to check for tests
	/// </summary>
	public virtual bool IsBoolean => typeKind == TypeKind.Boolean;
	public bool IsText => typeKind == TypeKind.Text;
	public bool IsNumber => typeKind == TypeKind.Number;
	public bool IsCharacter => typeKind == TypeKind.Character;
	public bool IsError => typeKind == TypeKind.Error;
	public bool IsList => typeKind == TypeKind.List;
	public bool IsDictionary => typeKind == TypeKind.Dictionary;
	public bool IsAny => typeKind == TypeKind.Any;

	public void Dispose()
	{
		GC.SuppressFinalize(this);
		((Package)Parent).Remove(this);
	}

	public int FindLineNumber(string firstLineThatContains)
	{
		for (var lineNumber = 0; lineNumber < Lines.Length; lineNumber++)
			if (Lines[lineNumber].Contains(firstLineThatContains))
				return lineNumber;
		return -1;
	}

	public Type GetFirstImplementation() => ((GenericTypeImplementation)this).ImplementationTypes[0];
}