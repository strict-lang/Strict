﻿using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Validators;

public sealed record MethodValidator(IEnumerable<Method> Methods) : Validator
{
	public void Validate()
	{
		foreach (var method in Methods)
			Validate(method);
	}

	private static void Validate(Method method)
	{
		if (method.GetBodyAndParseIfNeeded() is Body body)
		{
			ValidateUnchangedMutableVariables(body);
			ValidateUnusedVariables(body);
			ValidateMethodCall(body);
			ValidateMethodVariablesHidesAnyTypeMember(body, method.Type.Members);
		}
		ValidateMethodParameters(method);
	}

	private static void ValidateUnchangedMutableVariables(Body body)
	{
		var mutableVariables = body.Variables?.Where(variable => variable.IsMutable);
		var mutableReassignments = body.Expressions.OfType<MutableReassignment>().ToList();
		if (mutableVariables != null)
			foreach (var mutableVariable in mutableVariables)
				if (IsVariableValueUnchanged(mutableVariable, mutableReassignments))
					throw new VariableDeclaredAsMutableButValueNeverChanged(body, mutableVariable);
	}

	private static bool IsVariableValueUnchanged(Variable mutableVariable,
		IEnumerable<MutableReassignment> mutableReassignments) =>
		mutableReassignments.FirstOrDefault(m => m.Name == mutableVariable.Name) == null;

	public sealed class VariableDeclaredAsMutableButValueNeverChanged(Body body, Variable variable)
		: ParsingFailed(body, variable.Name);

	private static void ValidateUnusedVariables(Body body)
	{
		if (body.Variables != null)
			foreach (var variable in body.Variables)
				ValidateUnusedVariable(body.Method, variable.Name);
	}

	private static void ValidateUnusedVariable(Method method, string name)
	{
		if (method.GetVariableUsageCount(name) < 2)
			throw new UnusedMethodVariableMustBeRemoved(method.Type, name);
	}

	public sealed class UnusedMethodVariableMustBeRemoved(Type type, string name)
		: ParsingFailed(type, 0, name);

	private static void ValidateMethodCall(Body body)
	{
		for (var index = body.LineRange.Start.Value; index < body.LineRange.End.Value; index++)
		{
			var line = body.GetLine(index);
			if (line.Contains("((") && line.Contains("))") && line.Count(t => t == '(') < 3)
				throw new ListArgumentCanBeAutoParsedWithoutDoubleBrackets(body, line);
		}
	}

	public sealed class ListArgumentCanBeAutoParsedWithoutDoubleBrackets(Body type, string line)
		: ParsingFailed(type, line);

	private static void ValidateMethodVariablesHidesAnyTypeMember(Body body,
		IEnumerable<Member> members)
	{
		foreach (var member in members)
			if (body.Variables != null && body.FindVariable(member.Name) != null)
				throw new VariableHidesMemberUseDifferentName(body, body.Method.Name, member.Name);
	}

	public class
		VariableHidesMemberUseDifferentName(Body body, string methodName, string variableName)
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
		if (parameter is { IsMutable: true } &&
			new MutableAssignmentVisitor(parameter).Visit(method.GetBodyAndParseIfNeeded()) is null)
			throw new ParameterDeclaredAsMutableButValueNeverChanged(method.Type, parameter.Name);
	}

	public sealed class ParameterDeclaredAsMutableButValueNeverChanged(Type type, string name)
		: ParsingFailed(type, 0, name);

	private static void ValidateMethodParameterHidesAnyTypeMember(string parameterName, Method method)
	{
		if (method.Type.Members.Any(member => member.Name == parameterName))
			throw new ParameterHidesMemberUseDifferentName(method.Type, method.Name, parameterName);
	}

	public sealed class
		ParameterHidesMemberUseDifferentName(Type type, string methodName, string parameterName)
		: ParsingFailed(type, 0, $"Method name {methodName}, Parameter name {parameterName}");
}