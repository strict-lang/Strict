using System.Diagnostics;
using System.Runtime.CompilerServices;

[assembly: InternalsVisibleTo("Strict.Expressions.Tests")]

namespace Strict.Language;

/// <summary>
/// Every method body is just an expression, which might contain multiple expressions, which are
/// all executed, and then the final result is returned (all previous expressions must succeed).
/// Method parameters are in this context and can be used by any of the expressions nested here.
/// </summary>
public sealed class Body : Expression
{
	/// <summary>
	/// At construction time, we only know the method we are in if there is a parent Body we are in.
	/// While parsing each of the expressions, we need to check for variables as defined below. This
	/// means the expressions list can't be done yet and needs this object to exist for scope parsing
	/// </summary>
	[Log]
	public Body(Method method, int tabs = 0, Body? parent = null) : base(method.ReturnType,
		method.TypeLineNumber + method.lines.Count)
	{
		Method = method;
		Tabs = tabs;
		Parent = parent;
		parent?.children.Add(this);
	}

	public Method Method { get; private set; }
	public int Tabs { get; }
	public Body? Parent { get; private set; }
	public readonly List<Body> children = new();
	public Range LineRange { get; internal set; }
	public int ParsingLineNumber { get; set; }
	internal string CurrentLine => Method.lines[ParsingLineNumber];
	public bool IsFakeBodyForMemberInitialization => Method.Name == nameof(TypeParser.GetMemberExpression);

	/// <summary>
	/// Called when actually needed, and code needs to run, usually triggered by
	/// Method.GetBodyAndParseIfNeeded and child bodies inside. After parsing all expressions in this
	/// body, we will validate them all here. If there are multiple expressions, this body is
	/// returned, otherwise just a single expression is directly returned, and the body is discarded.
	/// The last expression return type must match our (method or caller) return type.
	/// </summary>
	[Log]
	public Expression Parse()
	{
		var expressions = new List<Expression>();
		for (ParsingLineNumber = LineRange.Start.Value; ParsingLineNumber < LineRange.End.Value;
			ParsingLineNumber++)
			try
			{
				expressions.Add(Method.ParseLine(this, CurrentLine));
			}
			catch (ParsingFailed)
			{
				StartDebuggerInDebugModeIfNotAttached();
				throw;
			}
			catch (Exception ex)
			{
				StartDebuggerInDebugModeIfNotAttached();
				throw new ParsingFailed((Type)Method.Parent, Method.TypeLineNumber + ParsingLineNumber,
					CurrentLine, ex);
			}
		SetExpressions(expressions);
		return Expressions.Count == 1
			? Expressions[0]
			: this;
	}

	private static void StartDebuggerInDebugModeIfNotAttached()
	{
#if DEBUG
		if (!Debugger.IsAttached && !StartedFromNCrunch && !StartedFromDotnetTestConsole &&
			!StartedFromReSharper)
			Debugger.Launch(); //ncrunch: no coverage
#endif
	}

	//ncrunch: no coverage start
	public static bool StartedFromNCrunch
	{
		get
		{
			if (wasStartedFromNCrunch is not null)
				return wasStartedFromNCrunch.Value;
			wasStartedFromNCrunch = Environment.GetEnvironmentVariable("NCrunch") == "1";
			return wasStartedFromNCrunch.Value;
		}
	}
	private static bool? wasStartedFromNCrunch;
	private static bool StartedFromDotnetTestConsole
	{
		get
		{
			if (wasStartedFromDotnetTestConsole is not null)
				return wasStartedFromDotnetTestConsole.Value;
			wasStartedFromDotnetTestConsole = AppDomain.CurrentDomain.FriendlyName.ToLower().
				StartsWith("test", StringComparison.Ordinal);
			return wasStartedFromDotnetTestConsole.Value;
		}
	}
	private static bool? wasStartedFromDotnetTestConsole;
	public static bool StartedFromReSharper =>
		AppDomain.CurrentDomain.FriendlyName.ToLower().
			StartsWith("resharpertest", StringComparison.Ordinal);
	//ncrunch: no coverage end

	public Body SetExpressions(IReadOnlyList<Expression> expressions)
	{
		Expressions = expressions;
		if (Expressions.Count == 0)
			throw new SpanExtensions.EmptyInputIsNotAllowed();
		ParsingLineNumber--;
		var lastExpression = Expressions[^1];
		var isLastExpressionReturn = lastExpression.GetType().Name == Base.Return;
		if (Method.ReturnType.Name != Base.None && (isLastExpressionReturn || IsMethodReturn()) &&
			Method.Name != Base.Run && Method.Name != Method.From && !ChildHasMatchingMethodReturnType(
				Parent == null
					? Method.ReturnType
					: Parent.ReturnType, lastExpression))
			throw new ChildBodyReturnTypeMustMatchMethod(this, lastExpression);
		return !isLastExpressionReturn
			? this
			: Parent != null
				? this
				: throw new ReturnAsLastExpressionIsNotNeeded(this);
	}

	private bool IsMethodReturn() =>
		ParsingLineNumber > 0 && ParsingLineNumber < LineRange.End.Value &&
		!CurrentLine.StartsWith("\t\t", StringComparison.Ordinal);

	public IReadOnlyList<Expression> Expressions { get; private set; } = [];

	/// <summary>
	/// We don't have access to the specific expressions here, so we need to do GetType().Name checks
	/// </summary>
	private static bool
		ChildHasMatchingMethodReturnType(Type parentType, Expression lastExpression) =>
		lastExpression.GetType().Name == Base.Declaration && parentType.Name == Base.None ||
		lastExpression.ReturnType.Name == Base.Error ||
		lastExpression.ReturnType.IsSameOrCanBeUsedAs(parentType);

	public sealed class ChildBodyReturnTypeMustMatchMethod(Body body, Expression lastExpression)
		: ParsingFailed(body,
			$"Last expression {
				lastExpression
			} return type: {
				lastExpression.ReturnType
			} is not matching with expected method return type:" + $" {
				(body.Parent == null
					? body.Method.ReturnType
					: body.Parent.ReturnType)
			} in method line: {
				body.ParsingLineNumber
			}");

	public sealed class ReturnAsLastExpressionIsNotNeeded(Body body) : ParsingFailed(body);
	public List<Variable>? Variables { get; private set; }

	public Body AddVariable(string name, Expression value, bool isMutable)
	{
		if (name.IsKeyword())
			throw new NamedType.CannotUseKeywordsAsName(name);
		if (!name.Length.IsWithinLimit())
			throw new NamedType.NameLengthIsNotWithinTheAllowedLimit(name);
		CheckForNameWithDifferentTypeUsage(name, value);
		var oldVariable = FindVariable(name.AsSpan());
		if (oldVariable is not null)
			throw new VariableNameIsAlreadyInUse(this, oldVariable, value);
		(Variables ??= new List<Variable>()).Add(new Variable(name, isMutable, value, this));
		return this;
	}

	private void CheckForNameWithDifferentTypeUsage(string name, Expression value)
	{
		var nameType = value.ReturnType.TryGetType(name.MakeFirstLetterUppercase());
		if (nameType != null && nameType != value.ReturnType)
			throw new VariableNameCannotHaveDifferentTypeNameThanValue(this, name,
				value.ReturnType.Name);
	}

	public class VariableNameIsAlreadyInUse(Body body, Variable oldVariable, Expression newValue)
		: ParsingFailed(body,
			$"Variable {
				oldVariable
			} was already declared before and cannot be re-declared here with: {
				newValue
			}");

	public class VariableNameCannotHaveDifferentTypeNameThanValue(Body body,
		string variableNameType, string valueType) : ParsingFailed(body,
		$"Variable name {variableNameType} " + $"denotes different type than its value type {
			valueType
		}. Prefer using a different name");

	public sealed class ValueIsNotMutableAndCannotBeChanged(Body body, string name)
		: ParsingFailed(body, name);

	public void CheckIfWeCouldUpdateMutableParameterOrVariable(Type contextType, string name, Expression value)
	{
		foreach (var member in contextType.Members)
			if (member.Name == name)
			{
				member.CheckIfWeCouldUpdateValue(value, this);
				return;
			}
		foreach (var parameter in Method.Parameters)
			if (parameter.Name == name)
			{
				parameter.CheckIfWeCouldUpdateValue(value, this);
				return;
			}
		var variable = FindVariable(name) ?? throw new IdentifierNotFound(this, name);
		variable.CheckIfWeCouldUpdateValue(value);
	}

	public sealed class IdentifierNotFound(Body body, string name) : ParsingFailed(body,
		name + ", Variables in scope: " + body.GetAllVariables().ToWordList());

	private List<Variable> GetAllVariables()
	{
		var allVariables = Variables ?? new List<Variable>();
		if (Parent != null)
			allVariables.AddRange(Parent.GetAllVariables());
		return allVariables;
	}

	public Variable? FindVariable(ReadOnlySpan<char> searchFor)
	{
		if (Variables != null)
			foreach (var variable in Variables)
				if (searchFor.Equals(variable.Name, StringComparison.Ordinal))
					return variable;
		return Parent?.FindVariable(searchFor);
	}

	public override bool IsConstant => Expressions.All(e => e.IsConstant);
	public override string ToString() => string.Join(Environment.NewLine, Expressions);
	public string GetLine(int lineNumber) => Method.lines[lineNumber];

	public Body? FindCurrentChild()
	{
		// ReSharper disable once ForCanBeConvertedToForeach, not done for performance reasons
		for (var index = 0; index < children.Count; index++)
		{
			var child = children[index];
			if (child.LineRange.Start.Value <= ParsingLineNumber)
				continue;
			if (child.LineRange.Start.Value > ParsingLineNumber + 1)
				break;
			ParsingLineNumber += child.LineRange.End.Value - child.LineRange.Start.Value;
			return child;
		}
		return null;
	}

	/// <summary>
	/// When cloning methods, we have to be careful to also clone the body and update it and all
	/// children bodies to link to the new cloned method and not longer the original method.
	/// Like the base from-method or generic method.
	/// </summary>
	public Body CloneAndUpdateMethod(Method newClonedMethod)
	{
		var clone = (Body)MemberwiseClone();
		clone.UpdateCurrentAndChildrenMethod(newClonedMethod);
		return clone;
	}

	private void UpdateCurrentAndChildrenMethod(Method newClonedMethod)
	{
		Method = newClonedMethod;
		foreach (var child in children)
			child.UpdateCurrentAndChildrenMethod(newClonedMethod);
	}

	public Body GetInnerBodyAndUpdateHierarchy(int currentLineNumber, Body child)
	{
		var innerForBody = new Body(Method, Tabs, this)
		{
			LineRange = new Range(currentLineNumber + 1, child.LineRange.End)
		};
		innerForBody.children.Add(child);
		child.Parent = innerForBody;
		children.Remove(child);
		return innerForBody;
	}
}