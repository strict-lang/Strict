﻿using System;

namespace Strict.Language.Expressions;

public sealed class ParameterCall : Expression
{
	public ParameterCall(Parameter parameter) : base(parameter.Type) => Parameter = parameter;
	public Parameter Parameter { get; internal set; }

	public static Expression? TryParse(Body body, ReadOnlySpan<char> input)
	{
		foreach (var parameter in body.Method.Parameters)
			if (input.Equals(parameter.Name, StringComparison.Ordinal))
				return new ParameterCall(parameter);
		return null;
	}

	public override string ToString() => Parameter.Name;
}