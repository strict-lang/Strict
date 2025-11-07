using Type = Strict.Language.Type;

namespace Strict.Validators;

/// <summary>
/// Checks if any variable is unused, if a mutable variable is never changed or if a method
/// parameter hides a member variable, all of which are not allowed and will cause an error.
/// </summary>
public sealed class TypeValidator : Visitor
{
	public override void Visit(Type type, object? context = null)
	{
		if (!type.IsDataType)
			foreach (var member in type.Members)
				if (!IsReservedName(member.Name) && !member.IsPublic &&
					type.CountMemberUsage(member.Name) < 2)
					throw new UnusedMemberMustBeRemoved(type, member.Name);
		base.Visit(type, context);
	}

	private static bool IsReservedName(string name) =>
		name is Base.ValueLowercase or "iterator" or "elements" or Base.GenericLowercase;

	public sealed class UnusedMemberMustBeRemoved(Type type, string memberName)
		: ParsingFailed(type, 0, memberName);

	protected override void Visit(Body body, object? context = null)
	{
		for (var index = body.LineRange.Start.Value; index < body.LineRange.End.Value; index++)
		{
			var line = body.GetLine(index);
			if (line.Contains("((") && line.Contains("))") && line.Count(t => t == '(') < 3)
				throw new ListArgumentCanBeAutoParsedWithoutDoubleBrackets(body, line);
		}
		if (body.Variables is null)
		{
			base.Visit(body, context);
			return;
		}
		context ??= new VariableUsages();
		base.Visit(body, context);
		ValidateUnusedVariables(body, context);
		ValidateMethodVariablesHidesAnyTypeMember(body, body.Method.Type.Members);
	}

	private sealed class VariableUsages
	{
		public readonly HashSet<string> used = new();
		public readonly HashSet<string> reassignedMutables = new();
	}

	private static void ValidateUnusedVariables(Body body, object? context)
	{
		if (context is not VariableUsages variables)
			return;
		foreach (var variable in body.Variables!)
			if (!variables.used.Contains(variable.Name))
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

	protected override Expression? Visit(Expression? expression, Body? body, object? context = null)
	{
		if (context is not VariableUsages variables)
			return expression;
		if (expression is ParameterCall parameterCall)
			variables.used.Add(parameterCall.Parameter.Name);
		else if (expression is VariableCall variableCall)
			variables.used.Add(variableCall.Variable.Name);
		else if (expression is MutableReassignment reassignment)
			variables.reassignedMutables.Add(reassignment.Name);
		return expression;
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

	public override void Visit(Method method, bool forceParsingBody = false, object? context = null)
	{
		if (method.Parameters.Any(p => p.IsMutable))
			context ??= new VariableUsages();
		base.Visit(method, forceParsingBody, context);
		foreach (var parameter in method.Parameters)
		{
			ValidateUnusedParameter(method, parameter.Name);
			ValidateUnchangedMutableParameter(method, parameter, context);
			ValidateMethodParameterHidesAnyTypeMember(parameter.Name, method);
		}
	}

	private static void ValidateUnusedParameter(Method method, string name)
	{
		if (method.Name != Method.From && !method.Type.IsTrait && method.GetParameterUsageCount(name) < 2)
			throw new UnusedMethodParameterMustBeRemoved(method, name);
	}

	public sealed class UnusedMethodParameterMustBeRemoved(Method method, string name)
		: ParsingFailed(method.Type, method.TypeLineNumber, name);

	private static void ValidateUnchangedMutableParameter(Method method, Parameter parameter, object? context)
	{
		if (context is VariableUsages variables && parameter is { IsMutable: true } &&
			!variables.reassignedMutables.Contains(parameter.Name))
			throw new ParameterDeclaredAsMutableButValueNeverChanged(method.Type, parameter.Name);
	}

	public sealed class ParameterDeclaredAsMutableButValueNeverChanged(Type type, string name)
		: ParsingFailed(type, 0, name);

	private static void ValidateMethodParameterHidesAnyTypeMember(string parameterName, Method method)
	{
		if (method.Name != Method.From && method.Type.Members.Any(member => member.Name == parameterName))
			throw new ParameterHidesMemberUseDifferentName(method.Type, method.Name, parameterName);
	}

	public sealed class ParameterHidesMemberUseDifferentName(Type type, string methodName, string parameterName)
		: ParsingFailed(type, 0, $"Method name {methodName}, Parameter name {parameterName}");
}