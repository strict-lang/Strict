using Strict.Language;
using Strict.Language.Expressions;
using Type = Strict.Language.Type;

namespace Strict.VirtualMachine;

/// <summary>
/// The only place where we can have a "static" method call to one of the from methods of a type
/// before we have a type instance yet, it is the only way to create instances.
/// </summary>
public sealed class Instance
{
	public Instance(Type? type, object value)
	{
		ReturnType = type;
		if (value is Value valueObj)
			Value = valueObj.Data;
		else
			Value = value;
	}

	public Instance(object value) => Value = value;

	public Instance(Expression expression)
	{
		ReturnType = expression.ReturnType;
		if (expression is Value value)
			Value = value.Data;
		else
			Value = new object();
	}

	public Type? ReturnType { get; }
	public object Value { get; set; }
}