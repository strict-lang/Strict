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
			throw new LinesCountMustNotExceedLimit(this, file.Lines.Length);
		if (package.FindDirectType(Name) != null)
			throw new TypeAlreadyExistsInPackage(Name, package);
		package.Add(this);
		lines = file.Lines;
		IsGeneric = Name == Base.Generic || OneOfFirstThreeLinesContainsGeneric();
	}

	public sealed class LinesCountMustNotExceedLimit : ParsingFailed
	{
		public LinesCountMustNotExceedLimit(Type type, int lineCount) : base(type, 0,
			$"Type {type.Name} has lines count {lineCount} but limit is {Limit.LineCount}") { }
	}

	public sealed class TypeAlreadyExistsInPackage : Exception
	{
		public TypeAlreadyExistsInPackage(string name, Package package) : base(
			name + " in package: " + package) { }
	}

	private string[] lines;
	private int lineNumber;
	/// <summary>
	/// Generic types cannot be used directly as we don't know the implementation to be used (e.g. a
	/// list, we need to know the type of the elements), you must them from <see cref="GenericTypeImplementation"/>!
	/// </summary>
	public bool IsGeneric { get; }

	private bool OneOfFirstThreeLinesContainsGeneric()
	{
		for (var line = 0; line < lines.Length && line < 3; line++)
			if (HasGenericMember(lines[line]))
				return true;
		return false;
	}

	private static bool HasGenericMember(string line) =>
		(line.StartsWith(HasWithSpaceAtEnd, StringComparison.Ordinal) ||
			line.StartsWith(MutableWithSpaceAtEnd, StringComparison.Ordinal)) &&
		(line.Contains(Base.Generic, StringComparison.Ordinal) ||
			line.Contains(Base.GenericLowercase, StringComparison.Ordinal));

	/// <summary>
	/// Parsing has to be done OUTSIDE the constructor as we first need all types and inside might not
	/// know all types yet needed for member assignments and method parsing (especially return types).
	/// </summary>
	public Type ParseMembersAndMethods(ExpressionParser parser)
	{
		ParseAllRemainingLinesIntoMembersAndMethods(parser);
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

	private void ParseAllRemainingLinesIntoMembersAndMethods(ExpressionParser parser)
	{
		for (; lineNumber < lines.Length; lineNumber++)
		{
			var rememberStartMethodLineNumber = lineNumber;
			ParseInTryCatchBlock(parser, rememberStartMethodLineNumber);
		}
	}

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
		var line = ValidateCurrentLineIsNonEmptyAndTrimmed();
		if (line.StartsWith(HasWithSpaceAtEnd, StringComparison.Ordinal))
			members.Add(GetNewMember(parser));
		else if (line.StartsWith(MutableWithSpaceAtEnd, StringComparison.Ordinal) &&
			!(lineNumber + 1 < lines.Length && lines[lineNumber + 1].StartsWith('\t')))
			members.Add(GetNewMember(parser, true));
		else
			methods.Add(new Method(this, lineNumber, parser, GetAllMethodLines()));
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
		return line;
	}

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

	private Member GetNewMember(ExpressionParser parser, bool usedMutableKeyword = false)
	{
		var member = ParseMember(parser, lines[lineNumber].AsSpan((usedMutableKeyword
			? MutableWithSpaceAtEnd
			: HasWithSpaceAtEnd).Length), usedMutableKeyword);
		if (members.Any(m => m.Name == member.Name))
			throw new DuplicateMembersAreNotAllowed(this, lineNumber, member.Name);
		return member;
	}

	private Member ParseMember(ExpressionParser parser, ReadOnlySpan<char> remainingLine,
		bool usedMutableKeyword)
	{
		if (methods.Count > 0)
			throw new MembersMustComeBeforeMethods(this, lineNumber, remainingLine.ToString());
		try
		{
			return TryParseMember(parser, remainingLine, usedMutableKeyword);
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

	private Member TryParseMember(ExpressionParser parser, ReadOnlySpan<char> remainingLine,
		bool usedMutableKeyword)
	{
		var nameAndExpression = remainingLine.Split();
		nameAndExpression.MoveNext();
		var nameAndType = nameAndExpression.Current.ToString();
		if (nameAndExpression.MoveNext())
		{
			var wordAfterName = nameAndExpression.Current.ToString();
			if (nameAndExpression.Current[0] == EqualCharacter)
				return new Member(this, nameAndType,
					GetMemberExpression(parser, nameAndType.MakeFirstLetterUppercase(),
						remainingLine[(nameAndType.Length + 3)..]), usedMutableKeyword);
			if (wordAfterName != Keyword.With)
				nameAndType += " " + GetMemberType(nameAndExpression);
			if (HasConstraints(wordAfterName, ref nameAndExpression))
				return !nameAndExpression.MoveNext()
					? throw new MemberMissingConstraintExpression(this, lineNumber, nameAndType)
					: IsMemberTypeAny(nameAndType, nameAndExpression)
						? throw new MemberWithTypeAnyIsNotAllowed(this, lineNumber, nameAndType)
						: GetMemberWithConstraints(parser, remainingLine, usedMutableKeyword, nameAndType);
			if (nameAndExpression.Current[0] == EqualCharacter)
				throw new NamedType.AssignmentWithInitializerTypeShouldNotHaveNameWithType(nameAndType);
		}
		return IsMemberTypeAny(nameAndType, nameAndExpression)
			? throw new MemberWithTypeAnyIsNotAllowed(this, lineNumber, nameAndType)
			: new Member(this, nameAndType, null, usedMutableKeyword);
	}

	private Member GetMemberWithConstraints(ExpressionParser parser, ReadOnlySpan<char> remainingLine,
		bool usedMutableKeyword, string nameAndType)
	{
		var member = new Member(this, nameAndType,
			ExtractConstraintsSpanAndValueExpression(parser, remainingLine, nameAndType,
				out var constraintsSpan), usedMutableKeyword);
		if (!constraintsSpan.IsEmpty)
			member.ParseConstraints(parser,
				constraintsSpan.ToString().Split(BinaryOperator.And, StringSplitOptions.TrimEntries));
		return member;
	}

	protected Expression? ExtractConstraintsSpanAndValueExpression(ExpressionParser parser,
		ReadOnlySpan<char> remainingLine, string nameAndType,
		out ReadOnlySpan<char> constraintsSpan)
	{
		var equalIndex = remainingLine.IndexOf(EqualCharacter);
		if (equalIndex > 0)
		{
			constraintsSpan = remainingLine[(nameAndType.Length + 1 + Keyword.With.Length + 1)..(equalIndex - 1)];
			return GetMemberExpression(parser, nameAndType.MakeFirstLetterUppercase(),
				remainingLine[(equalIndex + 2)..]);
		}
		constraintsSpan = remainingLine[(nameAndType.Length + 1 + Keyword.With.Length + 1)..];
		return null;
	}

	private const char EqualCharacter = '=';

	private Expression GetMemberExpression(ExpressionParser parser, string memberName,
		ReadOnlySpan<char> remainingTextSpan) =>
		parser.ParseExpression(new Body(new Method(this, 0, parser, new[] { "EmptyBody" })),
			GetFromConstructorCallFromUpcastableMemberOrJustEvaluate(memberName, remainingTextSpan));

	private ReadOnlySpan<char> GetFromConstructorCallFromUpcastableMemberOrJustEvaluate(
		string memberName, ReadOnlySpan<char> remainingTextSpan) =>
		FindType(memberName) != null && !remainingTextSpan.StartsWith(memberName)
			? string.Concat(memberName, "(", remainingTextSpan, ")").AsSpan()
			: remainingTextSpan;

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

	private static bool HasConstraints(string wordAfterName, ref SpanSplitEnumerator nameAndExpression) => wordAfterName == Keyword.With || nameAndExpression.MoveNext() && nameAndExpression.Current.ToString() == Keyword.With;

	public sealed class MemberMissingConstraintExpression : ParsingFailed
	{
		public MemberMissingConstraintExpression(Type type, int lineNumber, string memberName) : base(
			type, lineNumber, memberName) { }
	}

	private static bool
		IsMemberTypeAny(string nameAndType, SpanSplitEnumerator nameAndExpression) =>
		nameAndType == Base.AnyLowercase ||
		nameAndExpression.Current.Equals(Base.Any, StringComparison.Ordinal);

	public sealed class MemberWithTypeAnyIsNotAllowed : ParsingFailed
	{
		public MemberWithTypeAnyIsNotAllowed(Type type, int lineNumber, string name) : base(type, lineNumber, name) { }
	}

	public sealed class MembersMustComeBeforeMethods : ParsingFailed
	{
		public MembersMustComeBeforeMethods(Type type, int lineNumber, string line) : base(type,
			lineNumber, line) { }
	}

	public sealed class DuplicateMembersAreNotAllowed : ParsingFailed
	{
		public DuplicateMembersAreNotAllowed(Type type, int lineNumber, string name) :
			base(type, lineNumber, name) { }
	}

	public const string HasWithSpaceAtEnd = Keyword.Has + " ";
	public const string MutableWithSpaceAtEnd = Keyword.Mutable + " ";

	public sealed class MustImplementAllTraitMethodsOrNone : ParsingFailed
	{
		public MustImplementAllTraitMethodsOrNone(Type type, string traitName,
			IEnumerable<Method> missingTraitMethods) :
			base(type, type.lineNumber,
				"Trait Type:" + traitName + " Missing methods: " + string.Join(", ", missingTraitMethods)) { }
	}

	private void ValidateMethodAndMemberCountLimits()
	{
		var memberLimit = IsDatatypeOrEnum
			? Limit.MemberCountForEnums
			: Limit.MemberCount;
		if (members.Count > memberLimit)
			throw new MemberCountShouldNotExceedLimit(this, memberLimit);
		if (IsDatatypeOrEnum)
			return;
		if (methods.Count == 0 && members.Count < 2 && !IsNoneAnyOrBoolean() &&
			Name != Base.Name)
			throw new NoMethodsFound(this, lineNumber);
		if (methods.Count > Limit.MethodCount && Package.Name != nameof(Base))
			throw new MethodCountMustNotExceedLimit(this);
	}

	public bool IsDatatypeOrEnum =>
		methods.Count == 0 &&
		(members.Count > 1 || members.Count == 1 && members[0].Value is not null);

	public sealed class MemberCountShouldNotExceedLimit : ParsingFailed
	{
		public MemberCountShouldNotExceedLimit(Type type, int limit) : base(type, 0,
			$"{type.Name} type has {type.members.Count} members, max: {limit}") { }
	}

	private bool IsNoneAnyOrBoolean() => Name is Base.None or Base.Any or Base.Boolean or Base.Mutable;

	public sealed class NoMethodsFound : ParsingFailed
	{
		public NoMethodsFound(Type type, int lineNumber) : base(type, lineNumber,
			"Each type must have at least two members (datatypes and enums) or at least one method, otherwise it is useless") { }
	}

	public Package Package => (Package)Parent;

	public sealed class MethodCountMustNotExceedLimit : ParsingFailed
	{
		public MethodCountMustNotExceedLimit(Type type) : base(type, 0,
			$"Type {type.Name} has method count {type.methods.Count} but limit is {Limit.MethodCount}") { }
	}

	private void CheckIfTraitIsImplementedFullyOrNone(Type trait)
	{
		var nonImplementedTraitMethods = trait.Methods.Where(traitMethod =>
			traitMethod.Name != Method.From &&
			methods.All(implementedMethod => traitMethod.Name != implementedMethod.Name)).ToList();
		if (nonImplementedTraitMethods.Count > 0 && nonImplementedTraitMethods.Count != trait.Methods.Count(traitMethod => traitMethod.Name != Method.From))
			throw new MustImplementAllTraitMethodsOrNone(this, trait.Name, nonImplementedTraitMethods);
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
			throw new CharacterCountMustBeWithinLimit(this, line.Length, lineNumber + 1);
	}

	private const string SixTabs = "\t\t\t\t\t\t";

	public sealed class NestingMoreThanFiveLevelsIsNotAllowed : ParsingFailed
	{
		public NestingMoreThanFiveLevelsIsNotAllowed(Type type, int lineNumber) : base(type,
			lineNumber,
			$"Type {type.Name} has more than {Limit.NestingLevel} levels of nesting in line: {lineNumber + 1}") { }
	}

	public sealed class CharacterCountMustBeWithinLimit : ParsingFailed
	{
		public CharacterCountMustBeWithinLimit(Type type, int lineLength, int lineNumber) :
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
		if (lines[lineNumber].EndsWith(','))
			return;
		listEndLineNumber = lineNumber;
		if (lines[listStartLineNumber].Length < Limit.ListCharacterCount)
			throw new MultiLineListsAllowedOnlyWhenLengthIsMoreThanHundred(this,
				listStartLineNumber - 1, lines[listStartLineNumber].Length);
	}

	private int listStartLineNumber = -1;
	private int listEndLineNumber = -1;

	public sealed class MultiLineListsAllowedOnlyWhenLengthIsMoreThanHundred : ParsingFailed
	{
		public MultiLineListsAllowedOnlyWhenLengthIsMoreThanHundred(Type type, int lineNumber,
			int length) : base(type, lineNumber,
			"Current length: " + length +
			$", Minimum Length for Multi line list expression: {Limit.ListCharacterCount}") { }
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

	public IReadOnlyList<Member> Members => members;
	protected readonly List<Member> members = new();
	public IReadOnlyList<Method> Methods => methods;
	protected readonly List<Method> methods = new();
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

	public GenericTypeImplementation GetGenericImplementation(Type singleImplementationType)
	{
		var key = Name + "(" + singleImplementationType.Name + ")";
		return GetGenericImplementation(key) ?? CreateGenericImplementation(key, new[] { singleImplementationType });
	}

	private GenericTypeImplementation? GetGenericImplementation(string key)
	{
		if (!IsGeneric)
			throw new CannotGetGenericImplementationOnNonGeneric(Name, key);
		cachedGenericTypes ??= new Dictionary<string, GenericTypeImplementation>(StringComparer.Ordinal);
		return cachedGenericTypes.TryGetValue(key, out var genericType)
			? genericType
			: null;
	}

	private Dictionary<string, GenericTypeImplementation>? cachedGenericTypes;

	private GenericTypeImplementation CreateGenericImplementation(string key, IReadOnlyList<Type> implementationTypes)
	{
		if (Name != Base.List && Members.Count(m => m.Type.IsGeneric) != implementationTypes.Count)
			throw new TypeArgumentsCountDoesNotMatchGenericType(this, implementationTypes);
		var genericType = new GenericTypeImplementation(this, implementationTypes);
		cachedGenericTypes!.Add(key, genericType);
		return genericType;
	}

	public GenericTypeImplementation GetGenericImplementation(List<Type> implementationTypes)
	{
		var key = Name + implementationTypes.Select(t => t.Name).ToList().ToBrackets();
		return GetGenericImplementation(key) ?? CreateGenericImplementation(key, implementationTypes);
	}

	public sealed class CannotGetGenericImplementationOnNonGeneric : Exception
	{
		public CannotGetGenericImplementationOnNonGeneric(string name, string key) :
			base("Type: " + name + ", Generic Implementation: " + key) { }
	}

	public string FilePath => Path.Combine(Package.FolderPath, Name) + Extension;
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

	public class GenericTypesCannotBeUsedDirectlyUseImplementation : Exception
	{
		public GenericTypesCannotBeUsedDirectlyUseImplementation(Type type, string extraInformation) :
			base(type + " " + extraInformation) { }
	}

	private static bool IsMethodWithMatchingParameters(IReadOnlyList<Expression> arguments,
		Method method)
	{
		for (var index = 0; index < method.Parameters.Count; index++)
		{
			var methodParameterType = method.Parameters[index].Type;
			var argumentReturnType = arguments[index].ReturnType;
			if (argumentReturnType == methodParameterType || method.IsGeneric || methodParameterType.Name == Base.Any ||
				IsArgumentImplementationTypeMatchParameterType(argumentReturnType, methodParameterType))
				continue;
			if (methodParameterType.IsDatatypeOrEnum && methodParameterType.Members[0].Type == argumentReturnType)
				continue;
			if (methodParameterType.IsIterator != argumentReturnType.IsIterator && methodParameterType.Name != Base.Any)
				return false;
			if (methodParameterType.IsGeneric)
				throw new GenericTypesCannotBeUsedDirectlyUseImplementation(methodParameterType,
					"(parameter " + index + ") is not usable with argument " +
					arguments[index].ReturnType + " in " + method);
			if (!argumentReturnType.IsCompatible(methodParameterType))
				return false;
		}
		return true;
	}

	private static bool IsArgumentImplementationTypeMatchParameterType(Type argumentReturnType, Type methodParameterType) => argumentReturnType is GenericTypeImplementation argumentGenericType && argumentGenericType.ImplementationTypes.Any(t => t == methodParameterType);
	/// <summary>
	/// Any non public member is automatically iteratable if it has Iterator, for example Text.strict
	/// or Error.strict have public members you have to iterate over yourself.
	/// If there are two private iterators, then pick the first member automatically
	/// </summary>
	public bool IsIterator =>
		Name == Base.Iterator || members.Any(member => !member.IsPublic && member.Type.IsIterator);

	private Method? FindAndCreateFromBaseMethod(string methodName,
		IReadOnlyList<Expression> arguments)
	{
		if (methodName != Method.From)
			return null;
		var fromMethod = "from(";
		fromMethod += GetMatchingMemberParametersIfExist(arguments);
		return fromMethod.Length > 5 && fromMethod.Split(',').Length - 1 == arguments.Count
			? BuildMethod($"{fromMethod[..^2]})")
			: IsDatatypeOrEnum
				? BuildMethod(fromMethod[..^1])
				: null;
	}

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
		return parameters;
	}

	private Method BuildMethod(string fromMethod) => new(this, 0, dummyExpressionParser, new[] { fromMethod });
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
		this == sameOrBaseType ||
		HasAnyCompatibleMember(sameOrBaseType) ||
		CanUpCast(sameOrBaseType);

	private bool HasAnyCompatibleMember(Type sameOrBaseType) =>
		members.Any(member =>
			member.Type == sameOrBaseType && members.Count(m => m.Type == member.Type) == 1);

	// Created a case https://deltaengine.fogbugz.com/f/cases/27017
	private bool CanUpCast(Type sameOrBaseType) =>
		sameOrBaseType.Name == Base.Text && Name == Base.Number || sameOrBaseType.IsIterator &&
		members.Any(member => member.Type == GetType(Base.Number));

	/// <summary>
	/// When two types are using in a conditional expression, i.e. then and else return types and both
	/// are not based on each other, find the common base type that works for both.
	/// </summary>
	public Type? FindFirstUnionType(Type elseType)
	{
		foreach (var member in members)
			if (elseType.members.Any(otherMember => otherMember.Type == member.Type))
				return member.Type;
		foreach (var member in members)
		{
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
			if (cachedAvailableMethods != null)
				return cachedAvailableMethods;
			cachedAvailableMethods = new Dictionary<string, List<Method>>(StringComparer.Ordinal);
			foreach (var method in methods)
			{
				if (!method.IsPublic && method.Name != Method.From && !method.Name.AsSpan().IsOperator())
					continue;
				if (cachedAvailableMethods.ContainsKey(method.Name))
					cachedAvailableMethods[method.Name].Add(method);
				else
					cachedAvailableMethods.Add(method.Name, new List<Method> { method });
			}
			foreach (var member in members)
				if (!member.IsPublic && !IsTraitImplementation(member.Type))
					AddNonGenericMethods(member.Type);
			if (Name != Base.Any)
				AddAnyMethods();
			return cachedAvailableMethods;
		}
	}

	public bool IsTraitImplementation(Type memberType) =>
		memberType.IsTrait && methods.Count >= memberType.Methods.Count &&
		memberType.Methods.All(typeMethod =>
			methods.Any(method => method.HasEqualSignature(typeMethod)));

	private Dictionary<string, List<Method>>? cachedAvailableMethods;

	private void AddNonGenericMethods(Type implementType)
	{
		foreach (var (methodName, otherMethods) in implementType.AvailableMethods)
		{
			var nonGenericMethods = new List<Method>(implementType.IsGeneric
				? otherMethods.Where(m => !m.IsGeneric && !m.Parameters.Any(p => p.Type.IsGeneric))
				: otherMethods);
			AddAvailableMethods(methodName, nonGenericMethods);
		}
	}

	private void AddAvailableMethods(string methodName, List<Method> newMethods)
	{
		if (cachedAvailableMethods!.ContainsKey(methodName))
			cachedAvailableMethods[methodName].AddRange(newMethods);
		else
			cachedAvailableMethods.Add(methodName, newMethods);
	}

	private void AddAnyMethods()
	{
		cachedAnyMethods ??= GetType(Base.Any).AvailableMethods;
		foreach (var (methodName, anyMethods) in cachedAnyMethods)
			AddAvailableMethods(methodName, anyMethods);
	}

	private static IReadOnlyDictionary<string, List<Method>>? cachedAnyMethods;

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
	}
}