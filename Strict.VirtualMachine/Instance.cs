//https://deltaengine.fogbugz.com/f/cases/26382/Complete-Instance-cs-in-VirtualMachine-project-and-add-Object-container-to-hold-content-of-the-instance-type
//namespace Strict.VirtualMachine;

///// <summary>
///// The only place where we can have a "static" method call to one of the from methods of a type
///// before we have a type instance yet, it is the only way to create instances.
///// </summary>
//public class Instance
//{
//	public Instance(Type type) : base(type, new object()) { } //here object should be a container
//	public override string ToString() => ReturnType.Name;
//}