using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

public sealed class ValueInstance(Type returnType, object? value)
{
	public Type ReturnType { get; } = returnType;
	public object? Value { get; set; } = value;
	public override string ToString() => $"{ReturnType.Name}:{Value}";
}