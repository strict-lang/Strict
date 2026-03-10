namespace Strict.Bytecode;

/// <summary>
/// Each InstructionType corresponds to an Instruction class in the <see cref="Instructions" /> namespace
/// here. For details on what each instruction does, see the corresponding Instruction class.
/// Invoke variants (InvokeWriteToList, InvokeWriteToTable, InvokeRemove) encode special list/table
/// operations as distinct instruction types so no secondary enum is needed.
/// </summary>
public enum InstructionType : byte
{
	LoadConstantToRegister,
	LoadVariableToRegister,
	StoreConstantToVariable,
	StoreRegisterToVariable,
	Set,
	ListCall,
	StoreSeparator = 10,
	Add,
	Subtract,
	Multiply,
	Divide,
	Modulo,
	ArithmeticSeparator = 20,
	Equal,
	NotEqual,
	LessThan,
	GreaterThan,
	BinaryOperatorsSeparator = 30,
	Invoke,
	Return,
	LoopBegin,
	LoopEnd,
	Jump,
	JumpIfTrue,
	JumpIfFalse,
	JumpIfNotZero,
	JumpEnd,
	JumpToIdIfFalse,
	JumpToIdIfTrue,
	InvokeWriteToList,
	InvokeWriteToTable,
	InvokeRemove
}