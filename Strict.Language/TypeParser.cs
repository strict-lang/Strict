using System.Globalization;

namespace Strict.Language;

public sealed partial class TypeParser(Type type, string[] lines)
{
	private string[] lines = lines;

	public void ParseMembersAndMethods(ExpressionParser parser)
	{
		for (LineNumber = 0; LineNumber < lines.Length; LineNumber++)
			TryParse(parser, LineNumber);
	}

	/// <summary>
	/// Should be a property, but that is way slower in debug mode when this is most useful!
	/// </summary>
	internal int LineNumber = -1;

	private void TryParse(ExpressionParser parser, int rememberStartMethodLineNumber)
	{
		try
		{
			ParseLineForMembersAndMethods(parser);
		}
		catch (Context.TypeNotFound ex)
		{
			type.Dispose();
			throw new ParsingFailed(type, rememberStartMethodLineNumber, ex.Message, ex);
		}
		catch (ParsingFailed)
		{
			type.Dispose();
			throw;
		}
		catch (Exception ex)
		{
			type.Dispose();
			throw new ParsingFailed(type, rememberStartMethodLineNumber, string.IsNullOrEmpty(ex.Message)
				? ex.GetType().Name
				: ex.Message, ex);
		}
	}

	private void ParseLineForMembersAndMethods(ExpressionParser parser)
	{
		var line = ValidateCurrentLineIsNonEmptyAndTrimmed();
		if (line.StartsWith(Type.HasWithSpaceAtEnd, StringComparison.Ordinal))
		{
			type.Members.Add(GetNewMember(parser));
		}
		else if (line.StartsWith(Type.MutableWithSpaceAtEnd, StringComparison.Ordinal) &&
			!(LineNumber + 1 < lines.Length && lines[LineNumber + 1].StartsWith('\t')))
		{
			type.Members.Add(GetNewMember(parser, Keyword.Mutable));
		}
		else if (line.StartsWith(Type.ConstantWithSpaceAtEnd, StringComparison.Ordinal))
		{
			type.Members.Add(GetNewMember(parser, Keyword.Constant));
		}
		else
		{
			var methodFirstLineNumber = LineNumber;
			var methodLines = GetAllMethodLines(methodFirstLineNumber);
			DetectTrivialEndlessRecursionInFrom(methodLines);
			DetectSelfRecursionWithSameArguments(methodLines);
			DetectHugeConstantRange(methodLines);
			var method = new Method(type, methodFirstLineNumber, parser, methodLines);
			DetectRedundantReturn(methodLines, method);
			var existingMethod = type.Methods.Find(m =>
				m.Name == method.Name && m.ReturnType == method.ReturnType && m.Parameters.
					Select(p => p.Type).SequenceEqual(method.Parameters.Select(p => p.Type)));
			if (existingMethod != null)
				throw new MethodWithSameNameAndParameterCountAlreadyExists(type, //ncrunch: no coverage
					methodFirstLineNumber, method, existingMethod);
			type.Methods.Add(method);
		}
	}

	public sealed class MethodWithSameNameAndParameterCountAlreadyExists(Type type,
		int lineNumber,
		Method method,
		Method existingMethod)
		: ParsingFailed(type, lineNumber, method.ToString(),
			existingMethod.ToString()); //ncrunch: no coverage

	public sealed class
		MemberNameMustNotStartWithTypeName(Type type, int lineNumber, string memberName)
		: ParsingFailed(type, lineNumber,
			$"Member name {memberName} must not start with type name {type.Name}");

	private string ValidateCurrentLineIsNonEmptyAndTrimmed()
	{
		var line = lines[LineNumber];
		if (line.Length == 0)
			throw new EmptyLineIsNotAllowed(type, LineNumber);
		return char.IsWhiteSpace(line[0])
			? throw new ExtraWhitespacesFoundAtBeginningOfLine(type, LineNumber, line)
			: char.IsWhiteSpace(line[^1])
				? throw new ExtraWhitespacesFoundAtEndOfLine(type, LineNumber, line)
				: line;
	}

	public sealed class EmptyLineIsNotAllowed(Type type, int lineNumber)
		: ParsingFailed(type, lineNumber);

	public sealed class ExtraWhitespacesFoundAtBeginningOfLine(Type type,
		int lineNumber,
		string message,
		string method = "") : ParsingFailed(type, lineNumber,
		message + " (strict always requires tab for indentation)", method);

	public sealed class ExtraWhitespacesFoundAtEndOfLine(Type type,
		int lineNumber,
		string message,
		string method = "") : ParsingFailed(type, lineNumber, message, method);

	private Dictionary<Member, string>? rememberToInitializeMemberInitialValues;

	private static bool
		StartsWithConstructorCall(ReadOnlySpan<char> valueOnly, string declaredTypeName) =>
		valueOnly.StartsWith(declaredTypeName, StringComparison.Ordinal) &&
		valueOnly.Length > declaredTypeName.Length && valueOnly[declaredTypeName.Length] == '(';

	private const char EqualCharacter = '=';

	private static bool
		HasConstraints(string wordAfterName, ref SpanSplitEnumerator nameAndExpression) =>
		wordAfterName is Keyword.With || (nameAndExpression.MoveNext() &&
			nameAndExpression.Current.ToString() is Keyword.With);

	public sealed class
		RedundantExplicitMemberTypeName(Type type, int lineNumber, string memberName, string typeName)
		: ParsingFailed(type, lineNumber,
			$"Member '{memberName}' already infers type '{typeName}' from its name, remove the " +
			"redundant explicit type");

	private string[] GetAllMethodLines(int methodFirstLineNumber)
	{
		if (type.MustUseBodylessTraitMethods && IsNextLineValidMethodBody())
			throw new Type.TypeHasNoMembersAndThusMustBeATraitWithoutMethodBodies(type);
		if (!type.IsTrait && !IsNextLineValidMethodBody())
			throw new MethodMustBeImplementedInNonTrait(type, lines[LineNumber], LineNumber);
		IncrementLineNumberTillMethodEnd();
		return listStartLineNumber != -1
			? throw new UnterminatedMultiLineListFound(type, listStartLineNumber - 1,
				lines[listStartLineNumber])
			: lines[methodFirstLineNumber..(LineNumber + 1)];
	}

	private bool IsNextLineValidMethodBody()
	{
		if (LineNumber + 1 >= lines.Length)
			return false;
		var line = lines[LineNumber + 1];
		ValidateNestingAndLineCharacterCountLimit(line);
		if (line.StartsWith('\t'))
			return true;
		return line.Length != line.TrimStart().Length
			? throw new ExtraWhitespacesFoundAtBeginningOfLine(type, LineNumber, line)
			: false;
	}

	private void ValidateNestingAndLineCharacterCountLimit(string line)
	{
		if (line.StartsWith(SixTabs, StringComparison.Ordinal))
			throw new NestingMoreThanFiveLevelsIsNotAllowed(type, LineNumber + 1);
		if (line.Length > Limit.CharacterCount)
			throw new CharacterCountMustBeWithinLimit(type, line.Length, LineNumber + 1);
	}

	private const string SixTabs = "\t\t\t\t\t\t";

	public sealed class NestingMoreThanFiveLevelsIsNotAllowed(Type type, int lineNumber)
		: ParsingFailed(type, lineNumber,
			$"Type {type.Name} has more than {Limit.NestingLevel} levels of nesting in line: " +
			$"{lineNumber + 1}");

	public sealed class CharacterCountMustBeWithinLimit(Type type, int lineLength, int lineNumber)
		: ParsingFailed(type, lineNumber,
			$"Type {type.Name} has character count {lineLength} in line: {lineNumber + 1} but limit is " +
			$"{Limit.CharacterCount}");

	public sealed class MethodMustBeImplementedInNonTrait(Type type,
		string definitionLine,
		int lineNumber) : ParsingFailed(type, lineNumber, definitionLine);

	private void IncrementLineNumberTillMethodEnd()
	{
		while (IsNextLineValidMethodBody())
		{
			LineNumber++;
			if (lines[LineNumber - 1].EndsWith(','))
				MergeMultiLineListIntoSingleLine(',');
			else if (lines[LineNumber - 1].EndsWith('+'))
				MergeMultiLineListIntoSingleLine('+');
			if (listStartLineNumber != -1 && listEndLineNumber != -1)
				SetNewLinesAndLineNumbersAfterMerge();
		}
	}

	private void MergeMultiLineListIntoSingleLine(char endCharacter)
	{
		if (listStartLineNumber == -1)
			listStartLineNumber = LineNumber - 1;
		lines[listStartLineNumber] += ' ' + lines[LineNumber].TrimStart();
		if (lines[LineNumber].EndsWith(endCharacter))
			return;
		listEndLineNumber = LineNumber;
		if (lines[listStartLineNumber].Length < Limit.MultiLineCharacterCount)
			throw new MultiLineExpressionsAllowedOnlyWhenLengthIsMoreThanHundred(type,
				listStartLineNumber - 1, lines[listStartLineNumber].Length);
	}

	private int listStartLineNumber = -1;

	private int listEndLineNumber = -1;

	public sealed class MultiLineExpressionsAllowedOnlyWhenLengthIsMoreThanHundred(Type type,
		int lineNumber,
		int length) : ParsingFailed(type, lineNumber,
		"Current length: " + length + $", Minimum Length for Multi line expressions: {
			Limit.MultiLineCharacterCount
		}");

	private void SetNewLinesAndLineNumbersAfterMerge()
	{
		var newLines = new List<string>(lines[..(listStartLineNumber + 1)]);
		newLines.AddRange(lines[(listEndLineNumber + 1)..]);
		lines = newLines.ToArray();
		LineNumber = listStartLineNumber;
		listStartLineNumber = -1;
		listEndLineNumber = -1;
	}

	public sealed class UnterminatedMultiLineListFound(Type type, int lineNumber, string line)
		: ParsingFailed(type, lineNumber, line);
}
