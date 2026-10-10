using Strict.Language;

namespace Strict.Expressions;

public partial class MethodExpressionParser
{
	public override bool IsVariableMutated(Body body, string variableName)
	{
		foreach (var expression in body.Expressions)
		{
			if (IsMutationOfVariable(expression, variableName) ||
				(expression is If ifExpression &&
					CheckForVariableMutationInIf(variableName, ifExpression)) ||
				(expression is For forExpression &&
					(IsForCustomVariableMutation(forExpression, variableName) ||
						forExpression.Body is MutableReassignment ||
						IsMutableMethodCallOnVariable(forExpression.Body, variableName) ||
						(forExpression.Body is Body forBody && IsVariableMutated(forBody, variableName)) ||
						(forExpression.Body is If forIfBody &&
							CheckForVariableMutationInIf(variableName, forIfBody)))))
				return true;
			if (IsMutableMethodCallOnVariable(expression, variableName))
				return true;
		}
		return false;
	}

	private static bool IsForCustomVariableMutation(For forExpression, string variableName)
	{
		for (var index = 0; index < forExpression.CustomVariables.Length; index++)
			if (forExpression.CustomVariables[index] is VariableCall variableCall &&
				variableCall.Variable.Name == variableName)
				return true;
		return false;
	}

	private static bool IsMutationOfVariable(Expression expression, string variableName) =>
		expression is MutableReassignment reassignment && (reassignment.Name == variableName ||
			(reassignment.Target is ListCall { List: VariableCall listCall } &&
				listCall.Variable.Name == variableName));

	private bool CheckForVariableMutationInIf(string variableName, If ifExpression)
	{
		if (IsMutationOfVariable(ifExpression.Then, variableName) ||
			(ifExpression.Then is Body thenBody && IsVariableMutated(thenBody, variableName)) ||
			(ifExpression.Then is If ifBody && CheckForVariableMutationInIf(variableName, ifBody)) ||
			IsMutableMethodCallOnVariable(ifExpression.Then, variableName))
			return true;
		return ifExpression.OptionalElse != null &&
			(IsMutationOfVariable(ifExpression.OptionalElse, variableName) ||
				(ifExpression.OptionalElse is Body elseBody && IsVariableMutated(elseBody, variableName)) ||
				(ifExpression.OptionalElse is If elseIfBody &&
					CheckForVariableMutationInIf(variableName, elseIfBody)) ||
				IsMutableMethodCallOnVariable(ifExpression.OptionalElse, variableName));
	}

	private static bool IsMutableMethodCallOnVariable(Expression expression, string variableName) =>
		expression is MethodCall methodCall && (IsMutableInstanceMethodCall(methodCall, variableName) ||
			IsVariablePassedToMutableParameter(methodCall, variableName));

	private static bool IsMutableInstanceMethodCall(MethodCall methodCall, string variableName) =>
		methodCall is { Instance: VariableCall varCall, IsMutable: true } &&
		varCall.Variable.Name == variableName;

	private static bool IsVariablePassedToMutableParameter(MethodCall methodCall, string variableName)
	{
		for (var index = 0;
			index < methodCall.Arguments.Count && index < methodCall.Method.Parameters.Count; index++)
			if (methodCall.Method.Parameters[index].IsMutable &&
				GetRootVariableName(methodCall.Arguments[index]) == variableName)
				return true;
		return false;
	}

	private static string? GetRootVariableName(Expression expression)
	{
		var current = expression;
		while (current is ListCall listCall)
			current = listCall.List;
		while (current is MemberCall memberCall && memberCall.Instance != null)
			current = memberCall.Instance;
		return current is VariableCall variableCall
			? variableCall.Variable.Name
			: null;
	}
}
