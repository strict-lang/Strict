namespace Strict.Language;

/// <summary>
/// Simple helper to give us all the names of common base types often used like Number and Boolean
/// </summary>
public static class Base
{
	/// <summary>
	/// Has no implementation and is used for void, empty or none, which is not valid to assign.
	/// </summary>
	public const string None = nameof(None);
	/// <summary>
	/// Defines all the methods available in any type (everything automatically implements **Any**).
	/// These methods don't have to be implemented by any class, they are automatically implemented.
	/// </summary>
	public const string Any = nameof(Any);
	public const string AnyLowercase = "any";
	/// <summary>
	/// Most basic type: can only be true or false, any statement must either be None or return a
	/// Boolean (anything else is a compiler error). Any statement returning false (like a failing
	/// test) will also immediately cause an error at runtime or in the Editor via SCrunch.
	/// </summary>
	public const string Boolean = nameof(Boolean);
	/// <summary>
	/// Can be any floating point or integer number (think byte, short, int, long, float or double
	/// in other languages). Also, it can be a decimal or BigInteger, the compiler can decide and
	/// optimize this away into anything that makes sense in the current context.
	/// </summary>
	public const string Number = nameof(Number);
	public const string Character = nameof(Character);
	public const string Count = nameof(Count);
	public const string HashCode = nameof(HashCode);
	public const string Range = nameof(Range);
	public const string Text = nameof(Text);
	public const string Error = nameof(Error);
	public const string For = nameof(For);
	public const string Generic = nameof(Generic);
	public const string GenericLowercase = "generic";
	public const string Iterator = nameof(Iterator);
	public const string List = nameof(List);
	public const string Type = nameof(Type);
	public const string Method = nameof(Method);
	public const string Logger = nameof(Logger);
	public const string App = nameof(App);
	public const string System = nameof(System);
	public const string File = nameof(File);
	public const string Return = nameof(Return);
	public const string Directory = nameof(Directory);
	public const string Run = nameof(Run);
	public const string Declaration = nameof(Declaration);
	public const string MutableReassignment = nameof(MutableReassignment);
	public const string ValueLowercase = "value";
	public const string TextWriter = nameof(TextWriter);
	public const string TextReader = nameof(TextReader);
	public const string Stacktrace = nameof(Stacktrace);
	public const string Name = nameof(Name);
	public const string Mutable = nameof(Mutable);
	public const string Dictionary = nameof(Dictionary);
}