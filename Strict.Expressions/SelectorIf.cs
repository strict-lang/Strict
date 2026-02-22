using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

public sealed class SelectorIf(Expression selector, IReadOnlyList<SelectorIf.Case> cases,
	int lineNumber, Expression? optionalElse = null, Body? bodyForErrorMessage = null)
	: Expression(GetMatchingReturnType(cases, optionalElse, bodyForErrorMessage), lineNumber)
{
	public Expression Selector { get; } = selector;
	public IReadOnlyList<Case> Cases { get; } = cases;
	public Expression? OptionalElse { get; } = optionalElse;
	public override bool IsConstant =>
		Selector.IsConstant && //ncrunch: no coverage
		Cases.All(@case => @case.Pattern.IsConstant && @case.Then.IsConstant) && //ncrunch: no coverage
		(OptionalElse?.IsConstant ?? true);

	public override string ToString() =>
		"if " + Selector + " is" + Environment.NewLine +
		string.Join(Environment.NewLine, Cases.Select(FormatCase)) + (OptionalElse != null
			? Environment.NewLine + FormatElse(OptionalElse)
			: "");

	private static string FormatCase(Case @case)
	{
		var thenText = @case.Then.ToString();
		if (thenText.Contains(Environment.NewLine, StringComparison.Ordinal))
			thenText = thenText.Replace(Environment.NewLine, Environment.NewLine + "\t"); //ncrunch: no coverage
		return "\t" + @case.Pattern + ThenSeparator + thenText;
	}

	private static string FormatElse(Expression optionalElse)
	{
		var elseText = optionalElse.ToString();
		if (elseText.Contains(Environment.NewLine, StringComparison.Ordinal))
			elseText = elseText.Replace(Environment.NewLine, Environment.NewLine + "\t"); //ncrunch: no coverage
		return "\telse " + elseText;
	}

	public sealed class Case(Expression pattern, Expression then, Expression condition)
	{
		public Expression Pattern { get; } = pattern;
		public Expression Then { get; } = then;
		public Expression Condition { get; } = condition;
	}

	private static Type GetMatchingReturnType(IReadOnlyList<Case> cases, Expression? optionalElse,
		Body? bodyForErrorMessage)
	{
		if (cases.Count == 0)
			throw new InvalidOperationException("SelectorIf requires at least one case"); //ncrunch: no coverage
		var returnType = cases[0].Then.ReturnType;
		for (var i = 1; i < cases.Count; i++)
			returnType = GetMatchingType(returnType, cases[i].Then.ReturnType, bodyForErrorMessage);
		if (optionalElse != null)
			returnType = GetMatchingType(returnType, optionalElse.ReturnType, bodyForErrorMessage);
		return returnType;
	}

	private static Type GetMatchingType(Type thenType, Type? elseType, Body? bodyForErrorMessage) =>
		elseType == null || elseType.IsSameOrCanBeUsedAs(thenType, false) || elseType.IsError
			? thenType
			: thenType.IsSameOrCanBeUsedAs(elseType, false) || thenType.IsError
				? elseType
				: thenType.FindFirstUnionType(elseType) ??
				throw new If.ReturnTypeOfThenAndElseMustHaveMatchingType(
					bodyForErrorMessage ?? new Body(thenType.Methods[0]), thenType, elseType);

	private const string ThenSeparator = " then ";
}