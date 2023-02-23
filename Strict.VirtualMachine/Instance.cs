using Strict.Language;
using Strict.Language.Expressions;
using Type = Strict.Language.Type;

namespace Strict.VirtualMachine;
//TODO: Change this class so Value is always expression not an object! (LM)
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

	public Instance(Expression expression, bool isMember = false)
	{
		ReturnType = expression.ReturnType;
		if (expression is Value value)
			Value = value.Data;
		else
			Value = new object(); //ncrunch: no coverage
		IsMember = isMember;
	}

	public bool IsMember { get; set; }
	public Type? ReturnType { get; }
	public string TypeName =>
		ReturnType == null
			? typeName
			: ReturnType.Name;
	public object Value { get; set; }
}