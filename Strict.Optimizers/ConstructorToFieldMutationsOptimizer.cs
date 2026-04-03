using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Expressions;
using Strict.Language;

namespace Strict.Optimizers;

/// <summary>
/// Replaces <c>Invoke(Color.from(expr1, expr2, expr3))</c> patterns (produced by
/// <see cref="MethodInliningOptimizer"/> inlining GetBrightnessAdjustedColor) with
/// <see cref="ConstructValueTypeInstruction"/> so the VirtualMachine avoids a full
/// method-dispatch constructor call in the hot loop.
/// The replacement reuses the registers that already hold the computed field values
/// in the instruction stream just before the Invoke — no new register allocation needed.
/// </summary>
public sealed class ConstructorToFieldMutationsOptimizer : InstructionOptimizer
{
  public override void Optimize(BinaryExecutable binary)
  {
    foreach (var typeEntry in binary.MethodsPerType)
    foreach (var methodGroup in typeEntry.Value.MethodGroups.Values)
    foreach (var method in methodGroup)
      method.instructions = Optimize(method.instructions);
  }

  public override List<Instruction> Optimize(List<Instruction> instructions)
  {
    for (var index = 0; index < instructions.Count; index++)
    {
      if (instructions[index] is Invoke invoke &&
        TryBuildConstructValueType(invoke, instructions, index,
          out var replacement))
        instructions[index] = replacement;
    }
    return instructions;
  }

  private static bool TryBuildConstructValueType(Invoke invoke,
    List<Instruction> instructions, int invokeIndex,
    out ConstructValueTypeInstruction replacement)
  {
    replacement = null!;
    if (invoke.Method.Method.Name != Method.From || invoke.Method.Instance != null)
      return false;
    var returnType = invoke.Method.ReturnType;
    if (returnType.IsNumber || returnType.IsText || returnType.IsBoolean || returnType.IsList ||
      returnType.IsDictionary || returnType.IsTrait || returnType.Members.Count == 0)
      return false;
    var args = invoke.Method.Arguments;
    if (args.Count == 0 || args.Count > returnType.Members.Count)
      return false;
    // Every arg must be a simple arithmetic binary (field + modifier)
    if (!AllArgsAreBinaryExpressions(args))
      return false;
    // Collect the registers that hold each arg result by scanning backwards from invokeIndex
    var argRegisters = FindArgRegisters(invoke, instructions, invokeIndex);
    if (argRegisters == null)
      return false;
    replacement = new ConstructValueTypeInstruction(invoke.Register, returnType, argRegisters);
    return true;
  }

  private static bool AllArgsAreBinaryExpressions(IReadOnlyList<Expression> args)
  {
    foreach (var arg in args)
      if (arg is not Binary binary || !IsDirectArithmetic(binary.Method.Name))
        return false;
    return true;
  }

  /// <summary>
  /// Walks backward through the instruction list to find the registers that hold each argument
  /// of the Invoke. It matches BinaryInstruction result registers (index Registers[2]) because
  /// the arg expressions are arithmetic operations whose result is the last register written.
  /// </summary>
  private static Register[]? FindArgRegisters(Invoke invoke, List<Instruction> instructions,
    int invokeIndex)
  {
    var args = invoke.Method.Arguments;
    var found = new Register[args.Count];
    var remaining = args.Count;
    // Walk backwards; pick up BinaryInstruction output registers in order
    var matchIndex = args.Count - 1;
    for (var scan = invokeIndex - 1; scan >= 0 && remaining > 0; scan--)
    {
      if (instructions[scan] is not BinaryInstruction bin || bin.Registers.Length < 3 ||
        bin.IsConditional())
        continue;
      found[matchIndex] = bin.Registers[2];
      matchIndex--;
      remaining--;
    }
    return remaining == 0
      ? found
      : null;
  }

  private static bool IsDirectArithmetic(string methodName) =>
    methodName is BinaryOperator.Plus or BinaryOperator.Minus or BinaryOperator.Multiply
      or BinaryOperator.Divide;
}
