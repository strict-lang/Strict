using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Validators;

public sealed class MethodValidator : Visitor
{
	public override void VisitBody(Expression expression, object? context = null)
	{
		if (expression is not Body body)
		{
			base.VisitBody(expression, context);
			return;
		}
		for (var index = body.LineRange.Start.Value; index < body.LineRange.End.Value; index++)
		{
			var line = body.GetLine(index);
			if (line.Contains("((") && line.Contains("))") && line.Count(t => t == '(') < 3)
				throw new ListArgumentCanBeAutoParsedWithoutDoubleBrackets(body, line);
		}
		if (body.Variables is null)
		{
			base.VisitBody(expression, context);
			return;
		}
		var usedVariables = new HashSet<string>();
		base.VisitBody(expression, usedVariables);
		ValidateUnusedVariables(body, usedVariables);
		ValidateMethodVariablesHidesAnyTypeMember(body, body.Method.Type.Members);
	}

	private static void ValidateUnusedVariables(Body body, IReadOnlySet<string> usedVariables)
	{
		foreach (var variable in body.Variables!)
			if (!usedVariables.Contains(variable.Name))
				throw new UnusedMethodVariableMustBeRemoved(body.Method.Type, variable.Name);
		var mutableReassignments = body.Expressions.OfType<MutableReassignment>().ToList();
		foreach (var mutableVariable in body.Variables.Where(variable => variable.IsMutable))
			if (IsVariableValueUnchanged(mutableVariable, mutableReassignments))
				throw new VariableDeclaredAsMutableButValueNeverChanged(body, mutableVariable);
	}

	public sealed class UnusedMethodVariableMustBeRemoved(Type type, string name)
		: ParsingFailed(type, 0, name);

	private static bool IsVariableValueUnchanged(Variable mutableVariable,
		IEnumerable<MutableReassignment> mutableReassignments) =>
		mutableReassignments.FirstOrDefault(m => m.Name == mutableVariable.Name) == null;

	public sealed class VariableDeclaredAsMutableButValueNeverChanged(Body body, Variable variable)
		: ParsingFailed(body, variable.Name);

	protected override void VisitExpression(Expression expression, object? context)
	{
		if (context is not HashSet<string> usedVariables)
			return;
		if (expression is VariableCall variableCall)
			usedVariables.Add(variableCall.Variable.Name);
		else if (expression is MutableReassignment reassignment)
			usedVariables.Add(reassignment.Name);
	}

	public sealed class ListArgumentCanBeAutoParsedWithoutDoubleBrackets(Body body, string line)
		: ParsingFailed(body, line);

	private static void ValidateMethodVariablesHidesAnyTypeMember(Body body,
		IEnumerable<Member> members)
	{
		foreach (var member in members)
			if (body.Variables != null && body.FindVariable(member.Name) != null)
				throw new VariableHidesMemberUseDifferentName(body, body.Method.Name, member.Name);
	}

	public class VariableHidesMemberUseDifferentName(Body body, string methodName, string variableName)
		: ParsingFailed(body, $"Method name {methodName}, Variable name {variableName}");

	private static void ValidateMethodParameters(Method method)
	{
		foreach (var parameter in method.Parameters)
		{
			ValidateUnusedParameter(method, parameter.Name);
			ValidateUnchangedMutableParameter(method, parameter);
			ValidateMethodParameterHidesAnyTypeMember(parameter.Name, method);
		}
	}

	private static void ValidateUnusedParameter(Method method, string name)
	{
		if (method.GetParameterUsageCount(name) < 2)
			throw new UnusedMethodParameterMustBeRemoved(method.Type, name);
	}

	public sealed class UnusedMethodParameterMustBeRemoved(Type type, string name)
		: ParsingFailed(type, 0, name);

	private static void ValidateUnchangedMutableParameter(Method method, Parameter parameter)
	{
#if TODO
		if (parameter is { IsMutable: true } &&
			new MutableAssignmentVisitor(parameter).Visit(method.GetBodyAndParseIfNeeded()) is null)
			throw new ParameterDeclaredAsMutableButValueNeverChanged(method.Type, parameter.Name);
#endif
	}

	public sealed class ParameterDeclaredAsMutableButValueNeverChanged(Type type, string name)
		: ParsingFailed(type, 0, name);

	private static void ValidateMethodParameterHidesAnyTypeMember(string parameterName, Method method)
	{
		if (method.Type.Members.Any(member => member.Name == parameterName))
			throw new ParameterHidesMemberUseDifferentName(method.Type, method.Name, parameterName);
	}

	public sealed class ParameterHidesMemberUseDifferentName(Type type, string methodName, string parameterName)
		: ParsingFailed(type, 0, $"Method name {methodName}, Parameter name {parameterName}");
}