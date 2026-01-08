using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

public class Instance(Type type, int lineNumber = 0) : Expression(type, lineNumber)
{
	public static Expression Parse(Body body, Method method)
	{
		var valueInstance = new Instance((Type)method.Parent, body.CurrentFileLineNumber);
		body.AddVariable(Base.ValueLowercase, valueInstance, false);
		return valueInstance;
	}

	public override bool IsConstant => true;
	public override string ToString() => Base.ValueLowercase;
}