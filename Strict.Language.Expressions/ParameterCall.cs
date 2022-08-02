﻿namespace Strict.Language.Expressions;

// ReSharper disable once HollowTypeName
public sealed class ParameterCall : Expression
{
	public ParameterCall(Parameter parameter) : base(parameter.Type) => Parameter = parameter;
	public Parameter Parameter { get; }
	public override string ToString() => Parameter.Name;
}