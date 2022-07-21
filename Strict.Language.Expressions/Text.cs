using System;
using System.Collections.Generic;
using System.Linq;

namespace Strict.Language.Expressions;

public class Text : Value
{
	public Text(Context context, string value) : base(context.GetType(Base.Text), value) { }
	public override string ToString() => "\"" + Data + "\"";

	public override bool Equals(Expression? other) =>
		other is Value v && (string)Data == (string)v.Data;

	//[Obsolete]
	//public static Expression? TryParse(Method.Line line, string input) =>
	//	input.Length >= 2 && input[0] == '"' && input[^1] == '"'
	//		? new Text(line.Method, input[1..^1])
	//		: null;

	public static Expression? TryParse(Method.Line line, ReadOnlySpan<char> input) =>
		input.Length >= 2 && input[0] == '"' && input[^1] == '"'
			? new Text(line.Method, input.Slice(1, input.Length - 2).ToString())
			: null;
}