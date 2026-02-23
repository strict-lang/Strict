using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

public sealed class Instance(Type type, int lineNumber = 0, bool isMutable = false)
	: Expression(type, lineNumber, isMutable)
{
	public static Expression Parse(Body body, Method method)
	{
		var isMutable = method.ReturnType.IsMutable;
		var valueInstance = new Instance((Type)method.Parent, body.CurrentFileLineNumber, isMutable);
		body.AddVariable(Type.ValueLowercase, valueInstance, isMutable);
		return valueInstance;
	}

	public override bool IsConstant => false;
	public override string ToString() => Type.ValueLowercase;
}