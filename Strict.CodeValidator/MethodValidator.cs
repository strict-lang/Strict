using Strict.Language;
using Strict.Language.Expressions;
using Type = Strict.Language.Type;

namespace Strict.CodeValidator;

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
			ValidateMethodCall(body);
		}
		ValidateMethodParameters(method);
	}

	private static void ValidateMethodCall(Body body)
	{
		for (var index = body.LineRange.Start.Value; index < body.LineRange.End.Value; index++)
		{
			var line = body.GetLine(index);
			if (line.Contains("((") && line.Contains("))") && line.Count(t => t == '(') < 3)
				throw new ListArgumentCanBeAutoParsedWithoutDoubleBrackets(body, line);
		}
	}

	private static void ValidateUnchangedMutableVariables(Body body)
	{
		var mutableVariables = body.Variables?.Where(variable => variable.Value.IsMutable);
		var mutableDeclarations = body.Expressions.OfType<MutableDeclaration>().ToList();
		if (mutableVariables != null)
			foreach (var mutableVariable in mutableVariables)
				if (IsVariableValueUnchanged(mutableVariable, mutableDeclarations))
					throw new VariableDeclaredAsMutableButValueNeverChanged(body, mutableVariable.Key);
	}

	private static bool IsVariableValueUnchanged(KeyValuePair<string, Expression> mutableVariable,
		IEnumerable<MutableDeclaration> mutableDeclarations) =>
		mutableVariable.Value.Equals(GetDeclarationValue(mutableDeclarations, mutableVariable));

	private static Expression? GetDeclarationValue(
		IEnumerable<MutableDeclaration> mutableDeclarations,
		KeyValuePair<string, Expression> mutableVariable) =>
		mutableDeclarations.FirstOrDefault(m => m.Name == mutableVariable.Key)?.Value;

	public sealed class VariableDeclaredAsMutableButValueNeverChanged : ParsingFailed
	{
		public VariableDeclaredAsMutableButValueNeverChanged(Body body, string name) : base(body,
			name) { }
	}

	private static void ValidateMethodParameters(Method method)
	{
		foreach (var parameter in method.Parameters)
		{
			ValidateUnusedParameter(method, parameter);
			ValidateUnchangedMutableParameter(method, parameter);
		}
	}

	private static void ValidateUnusedParameter(Method method, NamedType parameter)
	{
		if (method.GetParameterUsageCount(parameter.Name) < 2)
			throw new UnusedMethodParameterMustBeRemoved(method.Type, parameter.Name);
	}

	public sealed class UnusedMethodParameterMustBeRemoved : ParsingFailed
	{
		public UnusedMethodParameterMustBeRemoved(Type type, string name) : base(type, 0, name) { }
	}

	private static void ValidateUnchangedMutableParameter(Method method, Parameter parameter)
	{
		if (parameter is { IsMutable: true, DefaultValue: null })
			throw new ParameterDeclaredAsMutableButValueNeverChanged(method.Type, parameter.Name);
	}

	public sealed class ParameterDeclaredAsMutableButValueNeverChanged : ParsingFailed
	{
		public ParameterDeclaredAsMutableButValueNeverChanged(Type type, string name) : base(type, 0,
			name) { }
	}

	public sealed class ListArgumentCanBeAutoParsedWithoutDoubleBrackets : ParsingFailed
	{
		public ListArgumentCanBeAutoParsedWithoutDoubleBrackets(Body type, string line) : base(type,
			line) { }
	}
}