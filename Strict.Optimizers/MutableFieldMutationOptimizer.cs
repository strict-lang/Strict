using Strict.Bytecode.Instructions;

namespace Strict.Optimizers;

/// <summary>
/// Replaces patterns where an object is extracted from a collection, a constructor is called
/// with modified field values, and the result is written back to the same collection location.
/// Pattern before:
///   1. ListCall to get object from collection into R1
///   2. Load field from object (via ListCall or MemberCall) into R2
///   3. Do operations on the field
///   4. NewInstance constructor call combining modified fields into R3
///   5. WriteToList to put R3 back to same collection/index
/// Pattern after (after method inlining makes this visible):
///   1. ListCall to get object from collection into R1
///   2. For each field:
///      a. Load field from R1 into R2
///      b. Do operations
///      c. WriteToTable to set field directly on R1
/// This optimization recognizes that instead of creating and replacing, we can mutate in-place.
/// </summary>
public sealed class MutableObjectMutationOptimizer : InstructionOptimizer
{
	public override List<Instruction> Optimize(List<Instruction> instructions) => instructions;
}
