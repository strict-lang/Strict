using System.Collections.Concurrent;
using System.Text.RegularExpressions;
#if DEBUG
using System.Runtime.CompilerServices;
#endif

namespace Strict.Language;

/// <summary>
/// .strict files contain a type or trait and must be in the correct namespace folder.
/// Strict code only contains optional implement, then has*, then methods*. No empty lines.
/// There is no typical lexing/scoping/token splitting needed as Strict syntax is very strict.
/// </summary>
public partial class Type : Context, IDisposable
{
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
		IsMutable = Name == Mutable || Name.StartsWith(Mutable + "(", StringComparison.Ordinal);
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

	/// <summary>Source lines of this type (no trailing final-newline artifact).</summary>
	public string[] Lines { get; }

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
	/// Most basic type: can only be true or false, any expression must either be None or return a
	/// Boolean (anything else is a compiler error). Any expression returning false (like a failing
	/// test) will also immediately cause an error at runtime or in the Editor via SCrunch.
	/// </summary>
	public const string Boolean = nameof(Boolean);

	/// <summary>
	/// Can be any floating point or integer number (think byte, short, int, long, float, or double
	/// in other languages). Also, it can be a decimal or BigInteger, the compiler can decide and
	/// optimize this away into anything that makes sense in the current context.
	/// </summary>
	public const string Number = nameof(Number);

	public const string Byte = nameof(Byte);

	public const string Character = nameof(Character);

	public const string HashCode = nameof(HashCode);

	public const string Range = nameof(Range);

	public const string Text = nameof(Text);

	public const string Error = nameof(Error);

	public const string ErrorWithValue = nameof(ErrorWithValue);

	public const string Iterator = nameof(Iterator);

	public const string List = nameof(List);

	public const string Logger = nameof(Logger);

	public const string System = nameof(System);

	public const string File = nameof(File);

	public const string Directory = nameof(Directory);

	public const string TextWriter = nameof(TextWriter);

	public const string TextReader = nameof(TextReader);

	public const string Stacktrace = nameof(Stacktrace);

	public const string Mutable = nameof(Mutable);

	public const string Dictionary = nameof(Dictionary);

	private TypeKind GetTypeKindFromName() =>
		Name switch
		{
			None => TypeKind.None,
			Boolean => TypeKind.Boolean,
			Number => TypeKind.Number,
			Byte => TypeKind.Number,
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

	public const string HasWithSpaceAtEnd = Keyword.Has + " ";

	public const string MutableWithSpaceAtEnd = Keyword.Mutable + " ";

	public const string ConstantWithSpaceAtEnd = Keyword.Constant + " ";

	/// <summary>
	/// Parsing has to be done OUTSIDE the constructor as we first need all types and inside might not
	/// know all types yet needed for member assignments and method parsing (especially return types).
	/// </summary>
	public Type ParseMembersAndMethods(ExpressionParser parser)
	{
		if (typeParser.LineNumber >= 0)
			throw new TypeWasAlreadyParsed(this); //ncrunch: no coverage
		savedParser = parser;
		typeParser.ParseMembersAndMethods(parser);
		typeParser.ParseDeferredConstraints(parser);
		DetermineEnumTypeKind();
		ValidateMethodAndMemberCountLimits();
		// ReSharper disable once ForCanBeConvertedToForeach, for performance reasons:
		// https://codeblog.jonskeet.uk/2009/01/29/for-vs-foreach-on-arrays-and-lists/
		for (var index = 0; index < members.Count; index++)
		{
			var trait = members[index].Type;
			if (!IsTrait && trait.typeParser.LineNumber > 0 && trait.IsTrait)
				CheckIfTraitIsImplementedFullyOrNone(trait);
		}
		return this;
	}

	internal Type ParseMembersAndMethodsForPackage(ExpressionParser parser)
	{
		if (typeParser.LineNumber >= 0)
			throw new TypeWasAlreadyParsed(this); //ncrunch: no coverage
		savedParser = parser;
		typeParser.ParseMembersAndMethods(parser);
		DetermineEnumTypeKind();
		ValidateMethodAndMemberCountLimits();
		// ReSharper disable once ForCanBeConvertedToForeach, for performance reasons:
		// https://codeblog.jonskeet.uk/2009/01/29/for-vs-foreach-on-arrays-and-lists/
		for (var index = 0; index < members.Count; index++)
		{
			var trait = members[index].Type;
			if (!IsTrait && trait.typeParser.LineNumber > 0 && trait.IsTrait)
				CheckIfTraitIsImplementedFullyOrNone(trait);
		}
		return this;
	}

	public void ParseDeferredConstraints(ExpressionParser parser) =>
		typeParser.ParseDeferredConstraints(parser);

	internal void InvalidateAvailableMethodsCache()
	{
		if (Package.Name is nameof(Strict) or "TestPackage")
			cachedAnyMethods = null;
		cachedAvailableMethods = null;
		lock (genericImplementationLock)
		{
			if (cachedGenericTypes == null)
				return;
			foreach (var genericType in cachedGenericTypes.Values)
				genericType.cachedAvailableMethods = null;
		}
	}

	internal void ReimplementGenericTypeMethods()
	{
		lock (genericImplementationLock)
		{
			if (cachedGenericTypes == null)
				return;
			foreach (var genericType in cachedGenericTypes.Values)
			{
				genericType.ReimplementMembers();
				genericType.ReimplementMethods();
			}
		}
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
		if (methods.Count > Limit.MethodCount &&
			((Package.Name != nameof(Strict) && Package.Name != "TestPackage") ||
				Name == "MethodCountMustNotExceedFifteen"))
			throw new MethodCountMustNotExceedLimit(this);
	}

	public bool IsEnum => typeKind == TypeKind.Enum;

	/// <summary>
	/// Data types have no methods and just some data. Number, Text, and most Base types are not
	/// data types as they have functionality (which makes sense), only types higher up that only
	/// have data (like Color, which has 4 Numbers) are actually pure Data types!
	/// </summary>
	public bool IsDataType =>
		(CheckIfParsed() && methods.Count == 0 &&
			(members.Count > 1 || members is [{ InitialValue: not null }])) || Name == Number ||
		Name == nameof(Name);

	private bool CheckIfParsed()
	{
		if (!IsGeneric && Lines.Length > 1 && typeParser.LineNumber == -1)
			throw new TypeIsNotParsedCallParseMembersAndMethods(this); //ncrunch: no coverage
		return true;
	}

	private sealed class TypeIsNotParsedCallParseMembersAndMethods(Type type)
		: Exception(type.ToString()); //ncrunch: no coverage

	public sealed class MemberCountShouldNotExceedLimit(Type type, int limit) : ParsingFailed(type, 0,
		$"{type.Name} type has {type.members.Count} members, max: {limit}");

	public sealed class NoMethodsFound(Type type, int lineNumber) : ParsingFailed(type, lineNumber,
		"Each type must have at least two members (datatypes and enums) or at least one method, " +
		"otherwise it is useless");

	public Package Package => (Package)Parent;

	public sealed class MethodCountMustNotExceedLimit(Type type) : ParsingFailed(type, 0,
		$"Type {type.Name} has method count {type.methods.Count} but limit is {Limit.MethodCount}");

	public List<Member> Members => members;

	protected readonly List<Member> members = [];

	public List<Method> Methods => methods;

	protected readonly List<Method> methods = [];

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

	private Dictionary<string, GenericTypeImplementation>? cachedGenericTypes;
	private readonly Lock genericImplementationLock = new();
	public string FilePath =>
		this is GenericTypeImplementation genericType
			? genericType.Generic.FilePath
			: Path.GetFullPath(Path.Combine(Package.FolderPath, Name + Extension));

	public const string Extension = ".strict";

	public Member? FindMember(string name)
	{
		CheckIfParsed();
		return Members.FirstOrDefault(member => member.Name == name);
	}

	public class GenericTypesCannotBeUsedDirectlyUseImplementation : Exception
	{
		public GenericTypesCannotBeUsedDirectlyUseImplementation(Type type, string extraInformation,
			string? methodName = null, IReadOnlyList<Expression>? arguments = null) : base(
			BuildMessage(type, extraInformation, methodName, arguments)) { }

		public GenericTypesCannotBeUsedDirectlyUseImplementation(
			GenericTypesCannotBeUsedDirectlyUseImplementation innerException, string calledFrom) : base(
			innerException.Message + ", Called from: " + calledFrom, innerException) { }

		private static string BuildMessage(Type type, string extraInformation, string? methodName,
			IReadOnlyCollection<Expression>? arguments)
		{
			var message = "Lookup context type: " + type + ", Reason: " + extraInformation;
			if (string.IsNullOrEmpty(methodName))
				return message;
			message += ", Attempted method: " + methodName;
			if (arguments == null || arguments.Count == 0)
				message += ", Arguments: none";
			else
				message += ", Arguments: " + string.Join(", ",
					arguments.Select(argument => argument + " => " + argument.ReturnType));
			return message;
		}
	}

	protected int cachedIteratorState;

	private const int IteratorFalse = 1;

	private const int IteratorTrue = 2;

	protected ConcurrentDictionary<string, bool>? cachedEvaluatedMemberTypes;

	internal bool
		CanBeCreatedFromSingleMember(Type sourceType, bool allowImplicitConversion = false) =>
		TryGetSingleValueMemberType(out var memberType) &&
		sourceType.IsSameOrCanBeUsedAs(memberType, allowImplicitConversion, 1);

	public override Type? FindTypeCore(string name, Context? searchingFrom = null) =>
		name == Name || name is Other or Outer || name == FullName
			? this
			: Package.FindTypeCore(name, searchingFrom ?? this);

	private ExpressionParser? savedParser;

	public int AutogeneratedEnumValue { get; internal set; }

	public int LineNumber => typeParser.LineNumber;

	private static readonly IReadOnlyDictionary<string, List<Method>> EmptyAvailableMethods =
		new Dictionary<string, List<Method>>(StringComparer.Ordinal);

	[ThreadStatic]
	private static HashSet<Type>? activeAvailableMethodBuilds;

	private readonly object availableMethodsLock = new();

	private volatile Dictionary<string, List<Method>>? cachedAvailableMethods;

	private static IReadOnlyDictionary<string, List<Method>>? cachedAnyMethods;

	[GeneratedRegex(@"^\t+(?:let|constant|mutable) (\w+) = ")]
	private static partial Regex DeclarationPattern { get; }

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
		RemoveGenericImplementations();
	}

	public int FindLineNumber(string firstLineThatContains)
	{
		for (var lineNumber = 0; lineNumber < Lines.Length; lineNumber++)
			if (Lines[lineNumber].Contains(firstLineThatContains))
				return lineNumber;
		return -1;
	}

	public string ToCodeString() =>
		IsList
			? GetFirstImplementation().Name.Pluralize()
			: Name;
}
