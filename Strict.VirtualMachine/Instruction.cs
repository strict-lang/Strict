namespace Strict.Runtime;

/// <summary>
/// Each Instruction corresponds to a Statement class in the <see cref="Statements" /> namespace
/// here. For details on what each instruction does, see the corresponding Statement class.
/// </summary>
public enum Instruction
{
	LoadConstantToRegister,
	LoadVariableToRegister,
	StoreConstantToVariable,
	StoreRegisterToVariable,
	StoreSeparator = 100,
	Add,
	Subtract,
	Multiply,
	Divide,
	Modulo,
	ArithmeticSeparator = 200,
	Equal,
	NotEqual,
	LessThan,
	GreaterThan,
	BinaryOperatorsSeparator = 300,
	Jump,
	JumpIfTrue,
	JumpIfFalse,
	ControlFlowSeparator = 400,
	Invoke,
	Return
	/*unused, remove!
	   LoopBegin,
	   LoopBeginRange,
	   IterationEnd,
	   ListCall,
	   JumpIfTrue,
	   JumpIfFalse,
	   JumpIfNotZero,
	   JumpEnd,
	   JumpToIdIfFalse,
	   JumpToIdIfTrue,
	WriteToList,
	WriteToTable,
	RemoveFromTable,
	Remove,
	//this is quite stupid, it is just a method call! ToText,
	//this is quite stupid, it is just a method call! ToNumber,

	   public sealed class Conversion(
	   	Register storedValueRegister,
	   	Register registerToStoreConversion,
	   	Strict.Language.Type conversionType,
	   	Instruction instruction) : RegisterStatement(instruction, storedValueRegister)
	   {
	   	public Strict.Language.Type ConversionType { get; } = conversionType;
	   	public Register RegisterToStoreConversion { get; } = registerToStoreConversion;
	   }
	   
	   public abstract class InstanceStatement(Instruction instruction, Instance instance) : Statement(instruction)
	   {
	   	public Instance Instance { get; } = instance;
	   	public override string ToString() => $"{Instruction} {Instance.Value}";
	   }
	
	   public sealed class IterationEnd(int steps) : Statement
	   {
	   	public int Steps { get; } = steps;
	   	public override Instruction Instruction => Instruction.IterationEnd;
	   }
	   
	   public sealed class ListCallStatement(Register register, Register indexValueRegister, string identifier)
	   	: RegisterStatement(register, Instruction.ListCall)
	   {
	   	public Register IndexValueRegister { get; } = indexValueRegister;
	   	public string Identifier { get; } = identifier;
	   }
	   
	   public sealed class LoopBegin(Register register)
	   	: RegisterStatement(register, Instruction.LoopBegin);
	
	   public sealed class LoopRangeBegin(Register startIndex, Register endIndex) : Statement
	   {
	   	public Register StartIndex { get; set; } = startIndex;
	   	public Register EndIndex { get; set; } = endIndex;
	   	public override Instruction Instruction => Instruction.LoopBeginRange;
	   }
	   
	   public class Remove(string identifier, Register register)
	   	: RegisterStatement(register, Instruction.Remove)
	   {
	   	public string Identifier { get; } = identifier;
	   }
	   
	   public class RemoveFromTable(Register key, string identifier) : RegisterStatement(key,
	   	Instruction.RemoveFromTable)
	   {
	   	public string Identifier { get; } = identifier;
	   }
	   
	   public sealed class WriteToList(Register register, string identifier)
	   	: RegisterStatement(register, Instruction.WriteToList)
	   {
	   	public string Identifier { get; } = identifier;
	   }
	
	   public sealed class WriteToTable(Register key, Register value, string identifier)
	   	: Statement(Instruction.WriteToTable)
	   {
	   	public Register Key { get; } = key;
	   	public Register Value { get; } = value;
	   	public string Identifier { get; } = identifier;
	   }
	*/
}

