namespace Strict.Language;

/// <summary>
/// .strict files contain a type or trait and must be in the correct namespace folder.
/// Strict code only contains optionally implement, then has*, then methods*. No empty lines.
/// There is no typical lexing/scoping/token splitting needed as Strict syntax is very strict.
/// </summary>
//TODO: split up into TypeParser, TypeValidator, TypeMemberFinder and TypeMethodFinder and use all of these in this Type class
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
		CreatedBy = "Package: " + package + ", file=" + file + ", StackTrace:\n" +
			StackTraceExtensions.FormatStackTraceIntoClickableMultilineText(1);
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
	/// <summary>
	/// For debugging purposes to see where this Type was initially created.
	/// </summary>
	public string CreatedBy { get; protected set; }

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

	private static bool HasGenericMethodHeader(string line) =>
		line.Contains(Base.Generic, StringComparison.Ordinal) ||
		line.Contains(Base.GenericLowercase, StringComparison.Ordinal);

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
			TryParse(parser, lineNumber);
	}

	private void TryParse(ExpressionParser parser, int rememberStartMethodLineNumber)
	{
		try
		{
			ParseLineForMembersAndMethods(parser);
		}
		catch (Context.TypeNotFound ex)
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
			throw new ParsingFailed(this, lineNumber, ex.Message.Split('\n').Take(2).ToWordList("\n"), ex);
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
					GetMemberExpression(parser, nameAndType,
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
			return GetMemberExpression(parser, nameAndType,
				remainingLine[(equalIndex + 2)..]);
		}
		constraintsSpan = remainingLine[(nameAndType.Length + 1 + Keyword.With.Length + 1)..];
		return null;
	}

	private const char EqualCharacter = '=';

	private Expression GetMemberExpression(ExpressionParser parser, string memberName,
		ReadOnlySpan<char> remainingTextSpan) =>
		parser.ParseExpression(new Body(new Method(this, 0, parser, new[] { EmptyBody })),
			GetFromConstructorCallFromUpcastableMemberOrJustEvaluate(memberName, remainingTextSpan));

	public const string EmptyBody = nameof(EmptyBody);

	private ReadOnlySpan<char> GetFromConstructorCallFromUpcastableMemberOrJustEvaluate(
		string memberName, ReadOnlySpan<char> remainingTextSpan)
	{
		var memberNameWithFirstLetterCaps = memberName.MakeFirstLetterUppercase();
		return FindType(memberNameWithFirstLetterCaps) != null &&
			!remainingTextSpan.StartsWith(memberNameWithFirstLetterCaps)
				? string.Concat(memberNameWithFirstLetterCaps, "(", remainingTextSpan, ")").AsSpan()
				: remainingTextSpan.StartsWith(Name) && !char.IsUpper(memberName[0])
					? throw new CurrentTypeCannotBeInstantiatedAsMemberType(this, lineNumber,
						remainingTextSpan.ToString())
					: remainingTextSpan;
	}

	public sealed class CurrentTypeCannotBeInstantiatedAsMemberType : ParsingFailed
	{
		public CurrentTypeCannotBeInstantiatedAsMemberType(Type type, int lineNumber, string typeName)
			: base(type, lineNumber, typeName) { }
	}

	private static string GetMemberType(SpanSplitEnumerator nameAndExpression)
	{
		var memberType = nameAndExpression.Current.ToString();
		while (memberType.Contains('(') && !memberType.Contains(')'))
		{
			nameAndExpression.MoveNext();
			memberType += " " + nameAndExpression.Current.ToString();
		}
		return memberType;
	}

	private static bool
		HasConstraints(string wordAfterName, ref SpanSplitEnumerator nameAndExpression) =>
		wordAfterName == Keyword.With || nameAndExpression.MoveNext() &&
		nameAndExpression.Current.ToString() == Keyword.With;

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
		var memberLimit = IsEnum
			? Limit.MemberCountForEnums
			: Limit.MemberCount;
		if (members.Count > memberLimit)
			throw new MemberCountShouldNotExceedLimit(this, memberLimit);
		if (IsDataType || IsEnum)
			return;
		if (methods.Count == 0 && members.Count < 2 && !IsNoneAnyOrBoolean() &&
			Name != Base.Name)
			throw new NoMethodsFound(this, lineNumber);
		if (methods.Count > Limit.MethodCount && Package.Name != nameof(Base))
			throw new MethodCountMustNotExceedLimit(this);
	}

	public bool IsDataType =>
		methods.Count == 0 &&
		(members.Count > 1 || members.Count == 1 && members[0].Value is not null);
	public bool IsEnum =>
		methods.Count == 0 && members.Count > 1 && members.All(m => m.Value is not null);

	public sealed class MemberCountShouldNotExceedLimit : ParsingFailed
	{
		public MemberCountShouldNotExceedLimit(Type type, int limit) : base(type, 0,
			$"{type.Name} type has {type.members.Count} members, max: {limit}") { }
	}

	private bool IsNoneAnyOrBoolean() => Name is Base.None or Base.Any or Base.Boolean or Base.Mutable;

	public sealed class NoMethodsFound : ParsingFailed
	{
		public NoMethodsFound(Type type, int lineNumber) : base(type, lineNumber,
			"Each type must have at least two members (datatypes and enums) or at least one method, " +
			"otherwise it is useless") { }
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
		if (nonImplementedTraitMethods.Count > 0 && nonImplementedTraitMethods.Count !=
			trait.Methods.Count(traitMethod => traitMethod.Name != Method.From))
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
			$"Type {type.Name} has more than {Limit.NestingLevel} levels of nesting in line: " +
			$"{lineNumber + 1}") { }
	}

	public sealed class CharacterCountMustBeWithinLimit : ParsingFailed
	{
		public CharacterCountMustBeWithinLimit(Type type, int lineLength, int lineNumber) :
			base(type, lineNumber,
				$"Type {type.Name} has character count {lineLength} in line: {lineNumber + 1} but limit is " +
				$"{Limit.CharacterCount}") { }
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
				MergeMultiLineListIntoSingleLine(',');
			else if (lines[lineNumber - 1].EndsWith('+'))
				MergeMultiLineListIntoSingleLine('+');
			if (listStartLineNumber != -1 && listEndLineNumber != -1)
				SetNewLinesAndLineNumbersAfterMerge();
		}
	}

	private void MergeMultiLineListIntoSingleLine(char endCharacter)
	{
		if (listStartLineNumber == -1)
			listStartLineNumber = lineNumber - 1;
		lines[listStartLineNumber] += ' ' + lines[lineNumber].TrimStart();
		if (lines[lineNumber].EndsWith(endCharacter))
			return;
		listEndLineNumber = lineNumber;
		if (lines[listStartLineNumber].Length < Limit.MultiLineCharacterCount)
			throw new MultiLineExpressionsAllowedOnlyWhenLengthIsMoreThanHundred(this,
				listStartLineNumber - 1, lines[listStartLineNumber].Length);
	}

	private int listStartLineNumber = -1;
	private int listEndLineNumber = -1;

	public sealed class MultiLineExpressionsAllowedOnlyWhenLengthIsMoreThanHundred : ParsingFailed
	{
		public MultiLineExpressionsAllowedOnlyWhenLengthIsMoreThanHundred(Type type, int lineNumber,
			int length) : base(type, lineNumber,
			"Current length: " + length + $", Minimum Length for Multi line expressions: {
				Limit.MultiLineCharacterCount
			}") { }
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
		return cachedGenericTypes.TryGetValue(key, out var genericType)
			? genericType
			: null;
	}

	private Dictionary<string, GenericTypeImplementation>? cachedGenericTypes;

	/// <summary>
	/// Most often called for List (or the Iterator trait), which we want to optimize for
	/// </summary>
	private GenericTypeImplementation CreateGenericImplementation(string key, IReadOnlyList<Type> implementationTypes)
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
		FindMethod(Method.From, implementationTypes) != null;

	private Method? FindMethod(string methodName, IReadOnlyList<Type> implementationTypes) =>
		!AvailableMethods.TryGetValue(methodName, out var matchingMethods)
			? null
			: matchingMethods.FirstOrDefault(method =>
				method.Parameters.Count == implementationTypes.Count &&
				IsMethodWithMatchingParametersType(method, implementationTypes));

	public sealed class CannotGetGenericImplementationOnNonGeneric : Exception
	{
		public CannotGetGenericImplementationOnNonGeneric(string name, string key) :
			base("Type: " + name + ", Generic Implementation: " + key) { }
	}

	public string FilePath => Path.Combine(Package.FolderPath, Name) + Extension;
	public const string Extension = ".strict";
	public Member? FindMember(string name) => Members.FirstOrDefault(member => member.Name == name);

	public Method GetMethod(string methodName, IReadOnlyList<Expression> arguments, ExpressionParser parser) =>
		FindMethod(methodName, arguments, parser) ??
		throw new NoMatchingMethodFound(this, methodName, AvailableMethods);

	public Method? FindMethod(string methodName, IReadOnlyList<Expression> arguments,
		ExpressionParser parser)
	{
		if (IsGeneric)
			throw new GenericTypesCannotBeUsedDirectlyUseImplementation(this,
				"Type is Generic and cannot be used directly");
		if (!AvailableMethods.TryGetValue(methodName, out var matchingMethods))
			return FindAndCreateFromBaseMethod(methodName, arguments, parser);
		foreach (var method in matchingMethods)
		{
			if (method.Parameters.Count == arguments.Count)
			{
				//TODO: clean up, optimized this a bit
				if (arguments.Count == 1)
				{
					if (IsMethodParameterMatchingArgument(method, 0, arguments[0].ReturnType))
						return method;
				}
				else if (IsMethodWithMatchingParametersType(method,
					arguments.Select(argument => argument.ReturnType).ToList()))
					return method;
			}
			//TODO: not sure about this, looks very slow to do on every possible method
			if (method.Parameters.Count == 1 && arguments.Count > 0)
			{
				var parameter = method.Parameters[0];
				if (IsParameterTypeList(parameter) && CanAutoParseArgumentsIntoList(arguments) &&
					IsMethodParameterMatchingWithArgument(arguments,
						(GenericTypeImplementation)parameter.Type))
					return method;
			}
		}
		return FindAndCreateFromBaseMethod(methodName, arguments, parser) ??
			throw new ArgumentsDoNotMatchMethodParameters(arguments, this, matchingMethods);
	}

	private static bool IsParameterTypeList(NamedType parameter) =>
		parameter.Type is GenericTypeImplementation { Generic.Name: Base.List };

	private static bool CanAutoParseArgumentsIntoList(IReadOnlyList<Expression> arguments) =>
		arguments.All(a => a.ReturnType == arguments[0].ReturnType);

	private static bool IsMethodParameterMatchingWithArgument(IReadOnlyList<Expression> arguments,
		GenericTypeImplementation genericType) =>
		genericType.ImplementationTypes[0] == arguments[0].ReturnType;

	public class GenericTypesCannotBeUsedDirectlyUseImplementation : Exception
	{
		public GenericTypesCannotBeUsedDirectlyUseImplementation(Type type, string extraInformation) :
			base(type + " " + extraInformation) { }
	}

	//TODO: got two usages, but they are different and can be optimized each
	private static bool IsMethodWithMatchingParametersType(Method method,
		IReadOnlyList<Type> argumentReturnTypes)
	{
		for (var index = 0; index < method.Parameters.Count; index++)
			if (!IsMethodParameterMatchingArgument(method, index, argumentReturnTypes[index]))
				return false;
		return true;
	}

	private static bool IsMethodParameterMatchingArgument(Method method, int index,
		Type argumentReturnType)
	{
		var methodParameterType = method.Parameters[index].Type;
		if (argumentReturnType == methodParameterType || method.IsGeneric ||
			methodParameterType.Name == Base.Any ||
			IsArgumentImplementationTypeMatchParameterType(argumentReturnType, methodParameterType))
			return true;
		if (methodParameterType.IsEnum && methodParameterType.Members[0].Type == argumentReturnType)
			return true;
		if (methodParameterType.Name == Base.Iterator && method.Type == argumentReturnType)
			return true;
		if (methodParameterType.IsGeneric)
			throw new GenericTypesCannotBeUsedDirectlyUseImplementation(
				methodParameterType, //ncrunch: no coverage
				"(parameter " + index + ") is not usable with argument " + argumentReturnType + " in " +
				method);
		return argumentReturnType.IsCompatible(methodParameterType);
	}

	private static bool
		IsArgumentImplementationTypeMatchParameterType(Type argumentType, Type parameterType) =>
		argumentType is GenericTypeImplementation argumentGenericType &&
		argumentGenericType.ImplementationTypes.Any(t => t == parameterType);

	/// <summary>
	/// Any non public member is automatically iteratable if it has Iterator, for example Text.strict
	/// or Error.strict have public members you have to iterate over yourself.
	/// If there are two private iterators, then pick the first member automatically
	/// </summary>
	public bool IsIterator => Name == Base.Iterator || HasAnyIteratorMember();

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
			var isIterator = !member.IsPublic && member.Type.IsIterator;
			cachedEvaluatedMemberTypes.Add(member.Type.Name, isIterator);
			if (isIterator)
				return true;
		}
		return false;
	}

	private bool? cachedIteratorResult;
	private Dictionary<string, bool> cachedEvaluatedMemberTypes = new();


	private Method? FindAndCreateFromBaseMethod(string methodName,
		IReadOnlyList<Expression> arguments, ExpressionParser parser)
	{
		if (methodName != Method.From)
			return null;
		var fromMethod = "from(";
		fromMethod += GetMatchingMemberParametersIfExist(arguments);
		return fromMethod.Length > 5 && (fromMethod.Split(',').Length - 1 == arguments.Count ||
			fromMethod.Split(',').Length - 1 == PrivateMembersCount)
			? BuildMethod($"{fromMethod[..^2]})", parser)
			: IsDataType
				? BuildMethod(fromMethod[..^1], parser)
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
		return parameters == null && arguments.Count > 1 && PrivateMembersCount == 1 &&
			CanUpcastAllArgumentsToMemberType(arguments, members[0], arguments[0].ReturnType)
				? FormatMemberAsParameterWithType()
				: parameters;
	}

	private int PrivateMembersCount => members.Count(member => !member.IsPublic);

	private static bool CanUpcastAllArgumentsToMemberType(IEnumerable<Expression> arguments,
		NamedType member, Type firstArgumentReturnType) =>
		member.Type is GenericTypeImplementation { Generic.Name: Base.List } genericType &&
		genericType.ImplementationTypes[0] == firstArgumentReturnType &&
		arguments.All(a => a.ReturnType == firstArgumentReturnType);

	private string FormatMemberAsParameterWithType() =>
		$"{members[0].Name.MakeFirstLetterLowercase()} {members[0].Type.Name}, ";

	private Method BuildMethod(string fromMethod, ExpressionParser parser) => new(this, 0, parser, new[] { fromMethod });

	public bool IsCompatible(Type sameOrBaseType) =>
		this == sameOrBaseType || HasAnyCompatibleMember(sameOrBaseType) ||
		CanUpCast(sameOrBaseType) || sameOrBaseType.IsMutableAndHasMatchingImplementation(this) || CanUpCastCurrentTypeToOther(sameOrBaseType) || IsCompatibleOneOfType(sameOrBaseType);

	private bool IsCompatibleOneOfType(Type sameOrBaseType) =>
		sameOrBaseType is OneOfType oneOfType && oneOfType.Types.Any(IsCompatible);

	private bool CanUpCastCurrentTypeToOther(Type sameOrBaseType) =>
		sameOrBaseType.members.Count == 1 && sameOrBaseType.methods.Count == 0 && sameOrBaseType.members[0].Type == this &&
		ValidateMemberConstraints(sameOrBaseType.members[0].Constraints);

	private static bool ValidateMemberConstraints(IReadOnlyCollection<Expression>? constraints) =>
		true; // TODO: figure out how to evaluate constraints at this point

	private bool HasAnyCompatibleMember(Type sameOrBaseType) =>
		members.Any(member =>
			member.Type == sameOrBaseType && members.Count(m => m.Type == member.Type) == 1);

	private bool CanUpCast(Type sameOrBaseType) =>
		sameOrBaseType.Name == Base.Text && Name == Base.Number || sameOrBaseType.IsIterator &&
		members.Any(member => member.Type == GetType(Base.Number));


	private bool IsMutableAndHasMatchingImplementation(Type argumentType) =>
		this is GenericTypeImplementation genericTypeImplementation &&
		genericTypeImplementation.Generic.Name == Base.Mutable &&
		genericTypeImplementation.ImplementationTypes[0] == argumentType;

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
			foreach (var member in members)
				if (!member.IsPublic && !IsTraitImplementation(member.Type))
					AddNonGenericMethods(member.Type);
			if (Name != Base.Any)
				AddAnyMethods();
			return cachedAvailableMethods;
		}
	}
	private Dictionary<string, List<Method>>? cachedAvailableMethods;

	private void AddAvailableMethod(Method method)
	{
		if (cachedAvailableMethods!.ContainsKey(method.Name))
		{
			var methodsWithThisName = cachedAvailableMethods[method.Name];
			foreach (var existingMethod in methodsWithThisName)
				if (existingMethod.IsSameMethodNameReturnTypeAndParameters(method))
					return;
			methodsWithThisName.Add(method);
		}
		else
			cachedAvailableMethods.Add(method.Name, new List<Method> { method });
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
					AddAvailableMethod(otherMethod);
	}

	private void AddAnyMethods()
	{
		cachedAnyMethods ??= GetType(Base.Any).AvailableMethods;
		foreach (var (_, anyMethods) in cachedAnyMethods)
		foreach (var anyMethod in anyMethods)
			AddAvailableMethod(anyMethod);
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
		public ArgumentsDoNotMatchMethodParameters(IReadOnlyList<Expression> arguments, Type type,
			IEnumerable<Method> allMethods) : base((arguments.Count == 0
				? "No arguments does "
				: (arguments.Count == 1
					? "Argument: "
					: "Arguments: ") + arguments.Select(a => a.ToStringWithType()).ToWordList() + " do ") +
			"not match these " + type + " method(s):\n" + string.Join("\n", allMethods)) { }
	}

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
		Console.WriteLine(this + " GetGenericTypeArguments: " + genericArguments.ToWordList());
		return genericArguments;
	}

	public class InvalidGenericTypeWithoutGenericArguments : Exception
	{
		public InvalidGenericTypeWithoutGenericArguments(Type type) : base(
			"This type is broken and needs to be fixed, check the creation: " + type + ", CreatedBy: " +
			type.CreatedBy) { }
	}
}