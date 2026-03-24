using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

public sealed class Instance(Type type, int lineNumber = 0, bool isMutable = false)
	: Expression(type, lineNumber, isMutable)
{
	public static Expression Parse(Body body, Method method)
	{
		var isMutable = method.ReturnType.IsMutable;
		var valueInstance = new Instance(GetUsableType((Type)method.Parent), body.CurrentFileLineNumber,
			isMutable);
		body.AddVariable(Type.ValueLowercase, valueInstance, isMutable);
		return valueInstance;
	}

	private static Type GetUsableType(Type type) =>
		type.IsGeneric && type is not GenericTypeImplementation && type.IsList
			? type.GetGenericImplementation(type.GetType(Type.GenericUppercase))
			: type;

	public override bool IsConstant => false;
	public override string ToString() => Type.ValueLowercase;

	//ncrunch: no coverage start
	public override bool Equals(Expression? other) =>
		ReferenceEquals(this, other) || other is Instance i && ReturnType == i.ReturnType;

	public override int GetHashCode() => ReturnType.GetHashCode();
}