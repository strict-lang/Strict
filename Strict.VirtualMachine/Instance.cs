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
	private readonly string typeName = string.Empty;

	public Instance(Type? type, object value)
	{
		ReturnType = type;
		if (value is Value valueObj)
			Value = valueObj.Data;
		else
			Value = value;
	}

	public Instance(string typeName, object value)
	{
		Value = value;
		this.typeName = typeName;
	}

	public Instance(Expression expression)
	{
		ReturnType = expression.ReturnType;
		if (expression is Value value)
			Value = value.Data;
		else
			Value = new object();
	}

	public Type? ReturnType { get; }
	public string TypeName =>
		ReturnType == null
			? typeName
			: throw new NotImplementedException("ReturnType.IsMutable() does make no sense, the expression must be checked! ? ReturnType.MutableReturnType?.Name : ReturnType.Name");
	public object Value { get; set; }
}