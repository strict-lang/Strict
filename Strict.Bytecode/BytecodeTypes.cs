using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Bytecode;

/// <summary>
/// After <see cref="BytecodeGenerator"/> generates all bytecode from the parsed expressions or
/// <see cref="BytecodeDeserializer"/> loads a .strictbinary ZIP file with the same bytecode,
/// this class contains the deserialized bytecode for each type used with each method used.
/// </summary>
public sealed class BytecodeTypes
{
	/// <summary>
	/// Each key is a type.FullName (e.g. Strict/Number, Strict/ImageProcessing/Color), the Value
	/// contains all members of this type and all not stripped out methods that were actually used.
	/// </summary>
	public Dictionary<string, TypeMembersAndMethods> MethodsPerType = new();

	public sealed class TypeMembersAndMethods
	{
		public List<TypeMember> Members = new();
		public Dictionary<string, List<Instruction>> InstructionsPerMethod = new();
	}

	public sealed record TypeMember(string Name, string FullTypeName,
		Instruction? InitialValueExpression);

	public List<Instruction>? Find(Type type, Method method) =>
		Find(type.FullName, method.FullName.Name, method.Parameters.Count, method.ReturnType.Name);

	public List<Instruction>? Find(string fullTypeName, string methodName, int parametersCount,
		string returnType = "") =>
		MethodsPerType.TryGetValue(fullTypeName, out var methods) &&
		methods.InstructionsPerMethod.TryGetValue(
			GetMethodKey(methodName, parametersCount, returnType), out var instructions)
			? instructions
			: null;

	public static string GetMethodKey(string name, int parametersCount, string returnType = "") =>
		name + parametersCount + returnType;
}