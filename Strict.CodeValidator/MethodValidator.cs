using Strict.Language;
using Strict.Language.Expressions;

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
			ValidateUnchangedMutableVariables(body);
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
}