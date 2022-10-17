//TODO: remove, put this stuff into VirtualMachine project, call it Instance

namespace Strict.Language.Expressions;

/// <summary>
/// The only place where we can have a "static" method call to one of the from methods of a type
/// before we have a type instance yet, it is the only way to create instances.
/// </summary>
public sealed class From : Value
{
	public From(Type type) : base(type, new object()/*TODO: should be container*/) { }
	public override string ToString() => ReturnType.Name;
}