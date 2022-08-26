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
// ReSharper disable once HollowTypeName
public class Type : Context
{
	public Type(Package package, TypeLines file) : base(package, file.Name)
	{
		if (package.FindDirectType(Name) != null)
			throw new TypeAlreadyExistsInPackage(Name, package);
		package.Add(this);
		lines = file.Lines;
		for (lineNumber = 0; lineNumber < lines.Length; lineNumber++)
			if (ValidateCurrentLineIsNonEmptyAndTrimmed().
				StartsWith(Implement, StringComparison.Ordinal))
				implements.Add(ParseImplement(lines[lineNumber][Implement.Length..]));
			else
				break;
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

	public sealed class TypeAlreadyExistsInPackage : Exception
	{
		public TypeAlreadyExistsInPackage(string name, Package package) : base(
			name + " in package: " + package) { }
	}

	private readonly string[] lines;
	private int lineNumber;
	public string FilePath => Path.Combine(Package.FolderPath, Name) + Extension;
	public Package Package => (Package)Parent;

	private void CheckIfTraitIsImplemented(Type trait)
	{
		var nonImplementedTraitMethods = trait.Methods.
			Where(traitMethod => traitMethod.Name != Method.From &&
				methods.All(implementedMethod => traitMethod.Name != implementedMethod.Name)).ToList();
		if (nonImplementedTraitMethods.Count > 0)
			throw new MustImplementAllTraitMethods(this, nonImplementedTraitMethods);
	}

	private Type ParseImplement(string remainingLine)
	{
		if (members.Count > 0 || methods.Count > 0)
			throw new ImplementMustComeBeforeMembersAndMethods(this, lineNumber, remainingLine); //ncrunch: no coverage this condition never pass since this method is always called before any members or methods are parsed
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
		if (methods.Count == 0 && members.Count + implements.Count < 2 && !IsNoneAnyOrBoolean())
			throw new NoMethodsFound(this, lineNumber);
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

	private void ParseAllRemainingLinesIntoMembersAndMethods(ExpressionParser parser)
	{
		for (; lineNumber < lines.Length; lineNumber++)
		{
			var rememberStartMethodLineNumber = lineNumber;
			try
			{
				ParseLineForMembersAndMethods(parser);
			}
			catch (TypeNotFound ex)
			{
				throw new ParsingFailed(this, rememberStartMethodLineNumber, ex.Message, ex);
			}
		}
	}

	private bool IsNoneAnyOrBoolean() =>
		Name == Base.None || Name == Base.Any && Name == Base.Boolean;

	private void ParseLineForMembersAndMethods(ExpressionParser parser)
	{
		if (ValidateCurrentLineIsNonEmptyAndTrimmed().StartsWith(Has, StringComparison.Ordinal))
			members.Add(ParseMember(parser, lines[lineNumber].AsSpan(Has.Length)));
		else if (lines[lineNumber].StartsWith(Implement, StringComparison.Ordinal))
			throw new ImplementMustComeBeforeMembersAndMethods(this, lineNumber, lines[lineNumber]);
		else
			methods.Add(new Method(this, lineNumber, parser, GetAllMethodLines()));
	}

	// ReSharper disable once MethodTooLong
	private Member ParseMember(ExpressionParser parser, ReadOnlySpan<char> remainingLine)
	{
		if (methods.Count > 0)
			throw new MembersMustComeBeforeMethods(this, lineNumber, remainingLine.ToString());
		var nameAndExpression = remainingLine.Split();
		nameAndExpression.MoveNext();
		var nameAndType = nameAndExpression.Current.ToString();
		if (nameAndExpression.MoveNext() && nameAndExpression.Current[0] != '=')
			nameAndType += " " + nameAndExpression.Current.ToString();
		try
		{
			var expression = nameAndExpression.MoveNext()
				? parser.ParseExpression(
					//TODO: dummy!
					new Body(new Method(this, 0, parser, new[] { "Dummy" })),
					remainingLine[(nameAndType.Length + 3)..])
				: null;
			return new Member(this, nameAndType, expression);
		}
		catch (ParsingFailed) // Unit test should be added after Member parsing logic is finished
		{
			throw;
		}
		catch (Exception ex)
		{
			throw new ParsingFailed(this, lineNumber, ex.Message, ex);
		}
	}

	public sealed class MembersMustComeBeforeMethods : ParsingFailed
	{
		public MembersMustComeBeforeMethods(Type type, int lineNumber, string line) : base(type,
			lineNumber, line) { }
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

	public sealed class NoMethodsFound : ParsingFailed
	{
		public NoMethodsFound(Type type, int lineNumber) : base(type, lineNumber,
			"Each type must have at least one method, otherwise it is useless") { }
	}

	public sealed class MustImplementAllTraitMethods : ParsingFailed
	{
		public MustImplementAllTraitMethods(Type type, IEnumerable<Method> missingTraitMethods) :
			base(type, type.lineNumber, "Missing methods: " + string.Join(", ", missingTraitMethods)) { }
	}

	private string[] GetAllMethodLines()
	{
		if (IsTrait && IsNextLineValidMethodBody())
			throw new TypeHasNoMembersAndThusMustBeATraitWithoutMethodBodies(this);
		if (!IsTrait && !IsNextLineValidMethodBody())
			throw new MethodMustBeImplementedInNonTraitType(this, lines[lineNumber]);
		var methodLineNumber = lineNumber;
		while (IsNextLineValidMethodBody())
			lineNumber++;
		return lines[methodLineNumber..(lineNumber + 1)];
	}

	private bool IsNextLineValidMethodBody()
	{
		if (lineNumber + 1 >= lines.Length)
			return false;
		var line = lines[lineNumber + 1];
		if (line.StartsWith('\t'))
			return true;
		if (line.Length != line.TrimStart().Length)
			throw new ExtraWhitespacesFoundAtBeginningOfLine(this, lineNumber, line);
		return false;
	}

	public sealed class TypeHasNoMembersAndThusMustBeATraitWithoutMethodBodies : ParsingFailed
	{
		public TypeHasNoMembersAndThusMustBeATraitWithoutMethodBodies(Type type) : base(type, 0) { }
	}

	// ReSharper disable once HollowTypeName
	public sealed class MethodMustBeImplementedInNonTraitType : ParsingFailed
	{
		public MethodMustBeImplementedInNonTraitType(Type type, string definitionLine) : base(type,
			type.lineNumber, definitionLine) { }
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
	public const string Extension = ".strict";

	public Method GetMethod(string methodName, IReadOnlyList<Expression> arguments) =>
		FindMethod(methodName, arguments) ??
		throw new NoMatchingMethodFound(this, methodName, AvailableMethods);

	public Method? FindMethod(string methodName, IReadOnlyList<Expression> arguments)
	{
		if (!AvailableMethods.TryGetValue(methodName, out var matchingMethods))
			return FindAndCreateFromBaseMethod(methodName, arguments);
		foreach (var method in matchingMethods)
			if (method.Parameters.Count == arguments.Count &&
				IsMethodWithMatchingParameters(arguments, method))
				return method;
		throw new ArgumentsDoNotMatchMethodParameters(arguments, matchingMethods);
	}

	private static bool IsMethodWithMatchingParameters(IReadOnlyList<Expression> arguments, Method method)
	{
		for (var index = 0; index < method.Parameters.Count; index++)
			if (!arguments[index].ReturnType.IsCompatible(method.Parameters[index].Type))
				return false;
		return true;
	}

	private Method? FindAndCreateFromBaseMethod(string methodName,
		IReadOnlyList<Expression> arguments)
	{
		if (methodName == Method.From && arguments.Count == 1)
			foreach (var implementType in implements)
				if (implementType == arguments[0].ReturnType)
					return new Method(this, 0, null!,
						new[] { "from(" + implementType.Name.MakeFirstLetterLowercase() + ")" });
		//TODO: also allow creation from any of the has members available, e.g. Stacktrace.strict
		//from(method)
		//	Method = method
		return null;
	}

	private bool IsCompatible(Type sameOrBaseType) =>
		this == sameOrBaseType || sameOrBaseType.Name == Base.Any ||
		implements.Contains(sameOrBaseType) || CanUpCast(sameOrBaseType);

	private bool CanUpCast(Type sameOrBaseType)
	{
		if (sameOrBaseType.Name is Base.List)
			return Name == Base.Number || implements.Contains(GetType(Base.Number)) || Name == Base.Text;
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
				AddAvailableMethods(implementType);
			if (Name != Base.Any)
				AddAvailableMethods(GetType(Base.Any));
			return cachedAvailableMethods;
		}
	}

	private void AddAvailableMethods(Type implementType)
	{
		foreach (var (methodName, otherMethods) in implementType.AvailableMethods)
			if (cachedAvailableMethods!.ContainsKey(methodName))
				cachedAvailableMethods[methodName].AddRange(otherMethods);
			else
				cachedAvailableMethods.Add(methodName, otherMethods);
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
					: "Arguments: ") + arguments.Select(a => a.ReturnType + " " + a).ToWordList() +
				" do ") +
			"not match:\n" + string.Join('\n', allMethods)) { }
	}
}