using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Strict.Language;

/// <summary>
/// .strict files contain a type or trait and must be in the correct namespace folder.
/// Strict code only contains optionally implement, then has*, then methods*. No empty lines.
/// There is no typical lexing/scoping/token splitting needed as Strict syntax is very strict.
/// </summary>
// ReSharper disable once ClassTooBig
public class Type : Context
{
	public Type(Package package, TypeLines file) : base(package, file.Name)
	{
		if (file.Lines.Length > Limit.LineCount)
			throw new LinesCountMustNotExceedTwoHundredFiftySix(this, file.Lines.Length);
		if (package.FindDirectType(Name) != null)
			throw new TypeAlreadyExistsInPackage(Name, package);
		package.Add(this);
		lines = file.Lines;
		IsGeneric = Name == Base.Generic;
		for (lineNumber = 0; lineNumber < lines.Length; lineNumber++)
			if (ValidateCurrentLineIsNonEmptyAndTrimmed().
				StartsWith(Implement, StringComparison.Ordinal))
				implements.Add(ParseImplement(lines[lineNumber][Implement.Length..]));
			else
				break;
	}

	public sealed class LinesCountMustNotExceedTwoHundredFiftySix : ParsingFailed
	{
		public LinesCountMustNotExceedTwoHundredFiftySix(Type type, int lineCount) : base(type, 0, $"Type {type.Name} has lines count {lineCount} but limit is {Limit.LineCount}") { }
	}

	public sealed class TypeAlreadyExistsInPackage : Exception
	{
		public TypeAlreadyExistsInPackage(string name, Package package) : base(
			name + " in package: " + package) { }
	}

	private string ValidateCurrentLineIsNonEmptyAndTrimmed()
	{
		var line = lines[lineNumber];
		if (line.Length == 0)
			throw new EmptyLineIsNotAllowed(this, lineNumber);
		if (char.IsWhiteSpace(line[0]))
			throw new ExtraWhitespacesFoundAtBeginningOfLine(this, lineNumber, line);
		if (char.IsWhiteSpace(line[^1]))
			throw new ExtraWhitespacesFoundAtEndOfLine(this, lineNumber, line);
		if (HasGenericMember(line))
			IsGeneric = true;
		return line;
	}

	private static bool HasGenericMember(string line) =>
		line.StartsWith(Has, StringComparison.Ordinal) && (line.Contains(Base.Generic, StringComparison.Ordinal)
			|| line.Contains(Base.Generic.MakeFirstLetterLowercase(), StringComparison.Ordinal));

	private string[] lines;
	private int lineNumber;
	public string FilePath => Path.Combine(Package.FolderPath, Name) + Extension;
	public Package Package => (Package)Parent;
	/// <summary>
	/// Generic types cannot be used directly as we don't know the implementation to be used (e.g. a
	/// list, we need to know the type of the elements), you must them from <see cref="GenericType"/>!
	/// </summary>
	public bool IsGeneric { get; private set; }

	public class GenericTypesCannotBeUsedDirectlyUseImplementation : Exception
	{
		public GenericTypesCannotBeUsedDirectlyUseImplementation(Type type, string extraInformation) :
			base(type + " " + extraInformation) { }
	}

	private void CheckIfTraitIsImplemented(Type trait)
	{
		var nonImplementedTraitMethods = trait.Methods.Where(traitMethod =>
			traitMethod.Name != Method.From &&
			methods.All(implementedMethod => traitMethod.Name != implementedMethod.Name)).ToList();
		if (nonImplementedTraitMethods.Count > 0)
			throw new MustImplementAllTraitMethods(this, nonImplementedTraitMethods);
	}

	private Type ParseImplement(string remainingLine)
	{
		if (remainingLine == Base.Any)
			throw new ImplementAnyIsImplicitAndNotAllowed(this, lineNumber, remainingLine);
		try
		{
			return Package.GetType(remainingLine);
		}
		catch (TypeNotFound ex)
		{
			throw new ParsingFailed(this, lineNumber, ex.Message, ex);
		}
	}

	public sealed class ImplementMustComeBeforeMembersAndMethods : ParsingFailed
	{
		public ImplementMustComeBeforeMembersAndMethods(Type type, int lineNumber, string name) :
			base(type, lineNumber, name) { }
	}

	public sealed class ImplementAnyIsImplicitAndNotAllowed : ParsingFailed
	{
		public ImplementAnyIsImplicitAndNotAllowed(Type type, int lineNumber, string name) : base(
			type, lineNumber, name) { }
	}

	/// <summary>
	/// Extra parsing step that has to be done OUTSIDE the constructor as we might not know all types
	/// needed for member (especially assignments) and method parsing (especially return types).
	/// </summary>
	public Type ParseMembersAndMethods(ExpressionParser parser)
	{
		ParseAllRemainingLinesIntoMembersAndMethods(parser);
		ValidateMethodAndMemberCountLimits();
		// ReSharper disable once ForCanBeConvertedToForeach, for performance reasons:
		// https://codeblog.jonskeet.uk/2009/01/29/for-vs-foreach-on-arrays-and-lists/
		for (var index = 0; index < implements.Count; index++)
		{
			var trait = implements[index];
			if (trait.IsTrait)
				CheckIfTraitIsImplemented(trait);
		}
		return this;
	}

	private void ValidateMethodAndMemberCountLimits()
	{
		if (members.Count > Limit.MemberCount)
			throw new MemberCountShouldNotExceedFifty(this);
		if (IsEnum)
			return;
		if (HasNoMethodsAndLessThanTwoMembersOrImplements() && !IsNoneAnyOrBoolean())
			throw new NoMethodsFound(this, lineNumber);
		if (methods.Count > Limit.MethodCount && Package.Name != nameof(Base))
			throw new MethodCountMustNotExceedFifteen(this);
	}

	public sealed class MemberCountShouldNotExceedFifty : ParsingFailed
	{
		public MemberCountShouldNotExceedFifty(Type type) : base(type, 0,
			$"{type.Name} has member count {type.members.Count} but limit is {Limit.MemberCount}") { }
	}

	private bool HasNoMethodsAndLessThanTwoMembersOrImplements() => methods.Count == 0 && members.Count + implements.Count < 2;
	private bool IsNoneAnyOrBoolean() => Name is Base.None or Base.Any or Base.Boolean;

	public sealed class NoMethodsFound : ParsingFailed
	{
		public NoMethodsFound(Type type, int lineNumber) : base(type, lineNumber,
			"Each type must have at least one method, otherwise it is useless") { }
	}

	public sealed class MethodCountMustNotExceedFifteen : ParsingFailed
	{
		public MethodCountMustNotExceedFifteen(Type type) : base(type, 0,
			$"Type {type.Name} has method count {type.methods.Count} but limit is {Limit.MethodCount}") { }
	}

	private void ParseAllRemainingLinesIntoMembersAndMethods(ExpressionParser parser)
	{
		for (; lineNumber < lines.Length; lineNumber++)
		{
			var rememberStartMethodLineNumber = lineNumber;
			ParseInTryCatchBlock(parser, rememberStartMethodLineNumber);
		}
		if (methods.Count == 0 && members.Count > 0)
			IsEnum = true;
	}

	public bool IsEnum { get; private set; }

	private void ParseInTryCatchBlock(ExpressionParser parser, int rememberStartMethodLineNumber)
	{
		try
		{
			ParseLineForMembersAndMethods(parser);
		}
		catch (TypeNotFound ex)
		{
			throw new ParsingFailed(this, rememberStartMethodLineNumber, ex.Message, ex);
		}
		catch (ParsingFailed)
		{
			throw;
		}
		catch (Exception ex)
		{
			throw new ParsingFailed(this, rememberStartMethodLineNumber,
				string.IsNullOrEmpty(ex.Message)
					? ex.GetType().Name
					: ex.Message, ex);
		}
	}

	private void ParseLineForMembersAndMethods(ExpressionParser parser)
	{
		if (ValidateCurrentLineIsNonEmptyAndTrimmed().StartsWith(Has, StringComparison.Ordinal))
			members.Add(GetNewMember(parser));
		else if (lines[lineNumber].StartsWith(Implement, StringComparison.Ordinal))
			throw new ImplementMustComeBeforeMembersAndMethods(this, lineNumber, lines[lineNumber]);
		else
			methods.Add(new Method(this, lineNumber, parser, GetAllMethodLines()));
	}

	private Member GetNewMember(ExpressionParser parser)
	{
		var member = ParseMember(parser, lines[lineNumber].AsSpan(Has.Length));
		if (members.Any(m => m.Name == member.Name))
			throw new DuplicateMembersAreNotAllowed(this, lineNumber, member.Name);
		return member;
	}

	private Member ParseMember(ExpressionParser parser, ReadOnlySpan<char> remainingLine)
	{
		if (methods.Count > 0)
			throw new MembersMustComeBeforeMethods(this, lineNumber, remainingLine.ToString());
		try
		{
			return TryParseMember(parser, remainingLine);
		}
		catch (ParsingFailed)
		{
			throw;
		}
		catch (Exception ex)
		{
			throw new ParsingFailed(this, lineNumber, ex.Message, ex);
		}
	}

	private Member TryParseMember(ExpressionParser parser, ReadOnlySpan<char> remainingLine)
	{
		var nameAndExpression = remainingLine.Split();
		nameAndExpression.MoveNext();
		var nameAndType = nameAndExpression.Current.ToString();
		if (nameAndExpression.MoveNext() && nameAndExpression.Current[0] != '=')
			nameAndType += " " + GetMemberType(nameAndExpression);
		return IsMemberTypeAny(nameAndType, nameAndExpression)
			? throw new MemberWithTypeAnyIsNotAllowed(this, lineNumber, nameAndType)
			: new Member(this, nameAndType, HasMemberExpression(nameAndExpression)
				? GetMemberExpression(parser, nameAndType.MakeFirstLetterUppercase(),
					remainingLine[(nameAndType.Length + 3)..])
				: null);
	}

	private static string GetMemberType(SpanSplitEnumerator nameAndExpression)
	{
		var memberType = nameAndExpression.Current.ToString();
		while (nameAndExpression.Current[^1] == ',')
		{
			nameAndExpression.MoveNext();
			memberType += " " + nameAndExpression.Current.ToString();
		}
		return memberType;
	}

	private static bool IsMemberTypeAny(string nameAndType, SpanSplitEnumerator nameAndExpression) => nameAndType == Base.Any.MakeFirstLetterLowercase() || nameAndExpression.Current.Equals(Base.Any, StringComparison.Ordinal);

	public sealed class MemberWithTypeAnyIsNotAllowed : ParsingFailed
	{
		public MemberWithTypeAnyIsNotAllowed(Type type, int lineNumber, string name) : base(type, lineNumber, name) { }
	}

	private static bool HasMemberExpression(SpanSplitEnumerator nameAndExpression) =>
		nameAndExpression.Current[0] == '=' ||
		nameAndExpression.MoveNext() && nameAndExpression.Current[0] == '=';

	public sealed class MembersMustComeBeforeMethods : ParsingFailed
	{
		public MembersMustComeBeforeMethods(Type type, int lineNumber, string line) : base(type,
			lineNumber, line) { }
	}

	private Expression GetMemberExpression(ExpressionParser parser, string memberName,
		ReadOnlySpan<char> remainingTextSpan) =>
		parser.ParseExpression(new Body(new Method(this, 0, parser, new[] { "EmptyBody" })),
			GetFromConstructorCallFromUpcastableMemberOrJustEvaluate(memberName, remainingTextSpan));

	private ReadOnlySpan<char> GetFromConstructorCallFromUpcastableMemberOrJustEvaluate(
		string memberName, ReadOnlySpan<char> remainingTextSpan) =>
		FindType(memberName) != null && !remainingTextSpan.StartsWith(memberName)
			? string.Concat(memberName, "(", remainingTextSpan, ")").AsSpan()
			: remainingTextSpan;

	public sealed class DuplicateMembersAreNotAllowed : ParsingFailed
	{
		public DuplicateMembersAreNotAllowed(Type type, int lineNumber, string name) :
			base(type, lineNumber, name) { }
	}

	public const string Implement = "implement ";
	public const string Has = "has ";

	public sealed class ExtraWhitespacesFoundAtBeginningOfLine : ParsingFailed
	{
		public ExtraWhitespacesFoundAtBeginningOfLine(Type type, int lineNumber, string message,
			string method = "") : base(type, lineNumber, message, method) { }
	}

	public sealed class ExtraWhitespacesFoundAtEndOfLine : ParsingFailed
	{
		public ExtraWhitespacesFoundAtEndOfLine(Type type, int lineNumber, string message,
			string method = "") : base(type, lineNumber, message, method) { }
	}

	public sealed class EmptyLineIsNotAllowed : ParsingFailed
	{
		public EmptyLineIsNotAllowed(Type type, int lineNumber) : base(type, lineNumber) { }
	}

	public sealed class MustImplementAllTraitMethods : ParsingFailed
	{
		public MustImplementAllTraitMethods(Type type, IEnumerable<Method> missingTraitMethods) :
			base(type, type.lineNumber,
				"Missing methods: " + string.Join(", ", missingTraitMethods)) { }
	}

	private string[] GetAllMethodLines()
	{
		if (IsTrait && IsNextLineValidMethodBody())
			throw new TypeHasNoMembersAndThusMustBeATraitWithoutMethodBodies(this);
		if (!IsTrait && !IsNextLineValidMethodBody())
			throw new MethodMustBeImplementedInNonTrait(this, lines[lineNumber]);
		var methodLineNumber = lineNumber;
		IncrementLineNumberTillMethodEnd();
		return listStartLineNumber != -1
			? throw new UnterminatedMultiLineListFound(this, listStartLineNumber - 1,
				lines[listStartLineNumber])
			: lines[methodLineNumber..(lineNumber + 1)];
	}

	private bool IsNextLineValidMethodBody()
	{
		if (lineNumber + 1 >= lines.Length)
			return false;
		var line = lines[lineNumber + 1];
		ValidateNestingAndLineCharacterCountLimit(line);
		if (line.StartsWith('\t'))
			return true;
		if (line.Length != line.TrimStart().Length)
			throw new ExtraWhitespacesFoundAtBeginningOfLine(this, lineNumber, line);
		return false;
	}

	private void ValidateNestingAndLineCharacterCountLimit(string line)
	{
		if (line.StartsWith(SixTabs, StringComparison.Ordinal))
			throw new NestingMoreThanFiveLevelsIsNotAllowed(this, lineNumber + 1);
		if (line.Length > Limit.CharacterCount)
			throw new CharacterCountMustBeWithinOneHundredTwenty(this, line.Length, lineNumber + 1);
	}

	private const string SixTabs = "\t\t\t\t\t\t";

	public sealed class NestingMoreThanFiveLevelsIsNotAllowed : ParsingFailed
	{
		public NestingMoreThanFiveLevelsIsNotAllowed(Type type, int lineNumber) : base(type,
			lineNumber,
			$"Type {type.Name} has more than {Limit.NestingLevel} levels of nesting in line: {lineNumber + 1}") { }
	}

	public sealed class CharacterCountMustBeWithinOneHundredTwenty : ParsingFailed
	{
		public CharacterCountMustBeWithinOneHundredTwenty(Type type, int lineLength, int lineNumber) :
			base(type, lineNumber,
				$"Type {type.Name} has character count {lineLength} in line: {lineNumber + 1} but limit is {Limit.CharacterCount}") { }
	}

	public sealed class TypeHasNoMembersAndThusMustBeATraitWithoutMethodBodies : ParsingFailed
	{
		public TypeHasNoMembersAndThusMustBeATraitWithoutMethodBodies(Type type) : base(type, 0) { }
	}

	public sealed class MethodMustBeImplementedInNonTrait : ParsingFailed
	{
		public MethodMustBeImplementedInNonTrait(Type type, string definitionLine) : base(type,
			type.lineNumber, definitionLine) { }
	}

	private void IncrementLineNumberTillMethodEnd()
	{
		while (IsNextLineValidMethodBody())
		{
			lineNumber++;
			if (lines[lineNumber - 1].EndsWith(','))
				MergeMultiLineListIntoSingleLine();
			if (listStartLineNumber != -1 && listEndLineNumber != -1)
				SetNewLinesAndLineNumbersAfterMerge();
		}
	}

	private void MergeMultiLineListIntoSingleLine()
	{
		if (listStartLineNumber == -1)
			listStartLineNumber = lineNumber - 1;
		lines[listStartLineNumber] += ' ' + lines[lineNumber].TrimStart();
		if (!lines[lineNumber].EndsWith(','))
		{
			listEndLineNumber = lineNumber;
			if (lines[listStartLineNumber].Length < Limit.ListCharacterCount)
				throw new MultiLineListsAllowedOnlyWhenLengthIsMoreThanHundred(this,
					listStartLineNumber - 1, lines[listStartLineNumber].Length);
		}
	}

	private int listStartLineNumber = -1;
	private int listEndLineNumber = -1;

	public sealed class MultiLineListsAllowedOnlyWhenLengthIsMoreThanHundred : ParsingFailed
	{
		public MultiLineListsAllowedOnlyWhenLengthIsMoreThanHundred(Type type, int lineNumber, int length) : base(type, lineNumber, $"Current length: {length}, Minimum Length for Multi line list expression: {Limit.ListCharacterCount}") { }
	}

	private void SetNewLinesAndLineNumbersAfterMerge()
	{
		var newLines = new List<string>(lines[..(listStartLineNumber + 1)]);
		newLines.AddRange(lines[(listEndLineNumber + 1)..]);
		lines = newLines.ToArray();
		lineNumber = listStartLineNumber;
		listStartLineNumber = -1;
		listEndLineNumber = -1;
	}

	public sealed class UnterminatedMultiLineListFound : ParsingFailed
	{
		public UnterminatedMultiLineListFound(Type type, int lineNumber, string line) : base(type, lineNumber, line) { }
	}

	public IReadOnlyList<Type> Implements => implements;
	private readonly List<Type> implements = new();
	public IReadOnlyList<Member> Members => members;
	private readonly List<Member> members = new();
	public IReadOnlyList<Method> Methods => methods;
	protected readonly List<Method> methods = new();
	public bool IsTrait =>
		Implements.Count == 0 && Members.Count == 0 && Name != Base.Number && Name != Base.Boolean;

	public override string ToString() =>
		base.ToString() + (implements.Count > 0
			? " " + nameof(Implements) + " " + Implements.ToWordList()
			: "");

	public override Type? FindType(string name, Context? searchingFrom = null) =>
		name == Name || name.Contains('.') && name == base.ToString() || name == Other
			? this
			: Package.FindType(name, searchingFrom ?? this);

	/// <summary>
	/// Easy way to get another instance of the class type we are currently in.
	/// </summary>
	public const string Other = nameof(Other);

	public GenericType GetGenericImplementation(List<Type> implementationTypes)
	{
		if (!IsGeneric)
			throw new CannotGetGenericImplementationOnNonGeneric(Name, implementationTypes);
		cachedGenericTypes ??= new Dictionary<string, GenericType>(StringComparer.Ordinal);
		if (cachedGenericTypes.TryGetValue(Name + implementationTypes.ToBrackets(), out var genericType))
			return genericType;
		genericType = new GenericType(this, implementationTypes);
		cachedGenericTypes.Add(Name + implementationTypes.ToBrackets(), genericType);
		return genericType;
	}

	private Dictionary<string, GenericType>? cachedGenericTypes;

	public sealed class CannotGetGenericImplementationOnNonGeneric : Exception
	{
		public CannotGetGenericImplementationOnNonGeneric(string name, List<Type> implementations) :
			base("Type: " + name + ", Generic Implementation: " + implementations.ToWordList()) { }
	}

	public const string Extension = ".strict";
	public Member? FindMember(string name) => Members.FirstOrDefault(member => member.Name == name);

	public Method GetMethod(string methodName, IReadOnlyList<Expression> arguments) =>
		FindMethod(methodName, arguments) ??
		throw new NoMatchingMethodFound(this, methodName, AvailableMethods);

	public Method? FindMethod(string methodName, IReadOnlyList<Expression> arguments)
	{
		if (IsGeneric)
			throw new GenericTypesCannotBeUsedDirectlyUseImplementation(this,
				"Type is Generic and cannot be used directly");
		if (!AvailableMethods.TryGetValue(methodName, out var matchingMethods))
			return FindAndCreateFromBaseMethod(methodName, arguments);
		foreach (var method in matchingMethods)
			if (method.Parameters.Count == arguments.Count &&
				IsMethodWithMatchingParameters(arguments, method))
				return method;
		return FindAndCreateFromBaseMethod(methodName, arguments) ??
			throw new ArgumentsDoNotMatchMethodParameters(arguments, matchingMethods);
	}

	private static bool IsMethodWithMatchingParameters(IEnumerable<Expression> arguments,
		Method method) =>
		IsMethodWithMatchingParameters(arguments.Select(a => a.ReturnType).ToList(), method);

	// ReSharper disable once MethodTooLong
	private static bool IsMethodWithMatchingParameters(IReadOnlyList<Type> argumentTypes, Method method)
	{
		for (var index = 0; index < method.Parameters.Count; index++)
		{
			var methodParameterType = method.Parameters[index].Type;
			var argumentReturnType = argumentTypes[index];
			if (methodParameterType.IsList != argumentReturnType.IsList && methodParameterType.Name != Base.Any)
				return false;
			if (argumentReturnType == methodParameterType || method.IsGeneric)
				continue;
			if (methodParameterType is GenericType parameterGenericType)
			{
				if (!argumentReturnType.IsCompatible(parameterGenericType.ImplementationTypes[index]))
					return false;
				continue;
			}
			if (argumentReturnType.IsMutable() && argumentReturnType.MutableReturnType?.Name ==
				methodParameterType.Name)
				continue;
			if (methodParameterType.IsGeneric)
				throw new GenericTypesCannotBeUsedDirectlyUseImplementation(methodParameterType,
					"(parameter " + index + ") is not usable with argument " +
					argumentTypes[index] + " in " + method);
			if (!argumentReturnType.IsCompatible(methodParameterType))
				return false;
		}
		return true;
	}

	public bool IsList =>
		IsGeneric
			? Name == Base.List
			: this is GenericType generic && generic.Generic.Name == Base.List;
	public bool IsMutable() => Name == Base.Mutable || Implements.Any(type => type.Name == Base.Mutable);
	public Type? MutableReturnType { get; set; }

	private Method? FindAndCreateFromBaseMethod(string methodName,
		IReadOnlyList<Expression> arguments)
	{
		if (methodName != Method.From)
			return null;
		var fromMethod = "from(";
		fromMethod += GetMatchingMemberParametersIfExist(arguments) ??
			GetMatchingImplementParametersIfExists(arguments);
		return fromMethod.Length > 5 && isMatchedWithList
			? BuildMethod($"{fromMethod})")
			: fromMethod.Length > 5 && fromMethod.Split(',').Length - 1 == arguments.Count
				? BuildMethod($"{fromMethod[..^2]})")
				: IsEnum
					? BuildMethod(fromMethod[..^1])
					: null;
	}

	private Method BuildMethod(string fromMethod) => new(this, 0, dummyExpressionParser, new[] { fromMethod });

	private string? GetMatchingMemberParametersIfExist(IReadOnlyList<Expression> arguments)
	{
		var argumentIndex = 0;
		string? parameters = null;
		foreach (var member in members)
			if (arguments.Count > argumentIndex && member.Type == arguments[argumentIndex].ReturnType)
			{
				parameters += $"{member.Name.MakeFirstLetterLowercase()} {member.Type.Name}, ";
				argumentIndex++;
			}
		return TryGetMatchedParameters(arguments, parameters);
	}

	private string? TryCheckForMatchingMembersAndGetParameters(IReadOnlyList<Expression> arguments)
	{
		var matchedMember = members.FirstOrDefault(member =>
			member.Type.IsList && arguments.All(argument => member.Type.Name == Base.List ||
				argument.ReturnType == ((GenericType)member.Type).ImplementationTypes[0]));
		if (matchedMember == null)
			return null;
		isMatchedWithList = true;
		return $"{matchedMember.Name.MakeFirstLetterLowercase()} {matchedMember.Type.Name}";
	}

	private string? TryCheckForMatchingImplementsAndGetParameters(IReadOnlyList<Expression> arguments)
	{
		var matchedImplement = implements.FirstOrDefault(implement =>
			implement.IsList && arguments.All(argument =>
				argument.ReturnType == ((GenericType)implement).ImplementationTypes[0]));
		if (matchedImplement == null)
			return null;
		isMatchedWithList = true;
		return $"{((GenericType)matchedImplement).ImplementationTypes[0].Name.MakeFirstLetterLowercase()}s";
	}

	private bool isMatchedWithList;

	private string? GetMatchingImplementParametersIfExists(IReadOnlyList<Expression> arguments)
	{
		var argumentIndex = 0;
		string? parameters = null;
		foreach (var implement in implements)
			if (arguments.Count > argumentIndex && implement == arguments[argumentIndex].ReturnType)
			{
				parameters += $"{implement.Name.MakeFirstLetterLowercase()} {implement.Name}, ";
				argumentIndex++;
			}
		return TryGetMatchedParameters(arguments, parameters, false);
	}

	private string? TryGetMatchedParameters(IReadOnlyList<Expression> arguments, string? parameters,
		bool isMember = true) =>
		isMember
			? parameters == null && arguments.Count > 0
				? TryCheckForMatchingMembersAndGetParameters(arguments)
				: parameters
			: parameters == null && arguments.Count > 0
				? TryCheckForMatchingImplementsAndGetParameters(arguments)
				: parameters;

	private readonly ExpressionParser dummyExpressionParser = new DummyExpressionParser();

	//ncrunch: no coverage start
	private sealed class DummyExpressionParser : ExpressionParser
	{
		public override Expression ParseLineExpression(Body body, ReadOnlySpan<char> line) => body;
		public override Expression ParseExpression(Body body, ReadOnlySpan<char> text) => body;

		public override List<Expression> ParseListArguments(Body body, ReadOnlySpan<char> text) =>
			new();
	} //ncrunch: no coverage end

	public bool IsCompatible(Type sameOrBaseType) =>
		this == sameOrBaseType || sameOrBaseType.Name == Base.Any ||
		implements.Contains(sameOrBaseType) || sameOrBaseType.Implements.Contains(this) ||
		CanUpCast(sameOrBaseType);

	private bool CanUpCast(Context sameOrBaseType)
	{
		if (sameOrBaseType.Name is Base.Text or Base.List)
			return Name == Base.Number || implements.Contains(GetType(Base.Number));
		return false;
	}

	/// <summary>
	/// Builds dictionary the first time we use it to access any method of this type or any of the
	/// implements parent types recursively. Filtering has to be done by <see cref="FindMethod"/>
	/// </summary>
	public IReadOnlyDictionary<string, List<Method>> AvailableMethods
	{
		get
		{
			if (cachedAvailableMethods != null)
				return cachedAvailableMethods;
			cachedAvailableMethods = new Dictionary<string, List<Method>>(StringComparer.Ordinal);
			foreach (var method in methods)
				if (cachedAvailableMethods.ContainsKey(method.Name))
					cachedAvailableMethods[method.Name].Add(method);
				else
					cachedAvailableMethods.Add(method.Name, new List<Method> { method });
			foreach (var implementType in implements)
				// If we are in a specific implementation (GenericType), don't add generic methods
				if (implementType.Name != Base.Mutable)
					AddAvailableMethods(implementType);
			if (Name != Base.Any)
				AddAvailableMethods(GetType(Base.Any));
			return cachedAvailableMethods;
		}
	}

	private void AddAvailableMethods(Type implementType)
	{
		foreach (var (methodName, otherMethods) in implementType.AvailableMethods)
		{
			var nonGenericMethods = new List<Method>(implementType.IsGeneric
				? otherMethods.Where(m => !m.IsGeneric && !m.Parameters.Any(p => p.Type.IsGeneric))
				: otherMethods);
			if (cachedAvailableMethods!.ContainsKey(methodName))
				cachedAvailableMethods[methodName].AddRange(nonGenericMethods);
			else
				cachedAvailableMethods.Add(methodName, nonGenericMethods);
		}
	}

	private Dictionary<string, List<Method>>? cachedAvailableMethods;

	public class NoMatchingMethodFound : Exception
	{
		public NoMatchingMethodFound(Type type, string methodName,
			IReadOnlyDictionary<string, List<Method>> availableMethods) : base(methodName +
			" not found for " + type + ", available methods: " + availableMethods.Keys.ToWordList()) { }
	}

	public sealed class ArgumentsDoNotMatchMethodParameters : Exception
	{
		public ArgumentsDoNotMatchMethodParameters(IReadOnlyList<Expression> arguments,
			IEnumerable<Method> allMethods) : base((arguments.Count == 0
				? "No arguments does "
				: (arguments.Count == 1
					? "Argument: "
					: "Arguments: ") + arguments.Select(a => a.ToStringWithType()).ToWordList() +
				" do ") +
			"not match these method(s):\n" + string.Join("\n",
				allMethods)) { }

		public ArgumentsDoNotMatchMethodParameters(IReadOnlyCollection<Type> argumentTypes,
			IEnumerable<Method> matchingMethods) : base((argumentTypes.Count == 0
				? "No arguments does "
				: (argumentTypes.Count == 1
					? "Argument: "
					: "Arguments: ") + argumentTypes.ToWordList() + " do ") +
			"not match these method(s):\n" +
			string.Join("\n", matchingMethods)) { }
	}

	public void AddDataReturnTypeToMutableImplements(Type dataReturnType)
	{
		if (Name.StartsWith(Base.Mutable, StringComparison.Ordinal) && !implements.Contains(dataReturnType))
			implements.Add(dataReturnType);
	}

	public Method? FindMethodByArgumentTypes(string methodName, List<Type> argumentTypes)
	{
		if (!AvailableMethods.TryGetValue(methodName, out var matchingMethods))
			return null;
		foreach (var method in matchingMethods)
			if (method.Parameters.Count == argumentTypes.Count &&
				IsMethodWithMatchingParameters(argumentTypes, method))
				return method;
		throw new ArgumentsDoNotMatchMethodParameters(argumentTypes,
			matchingMethods);
	}
}