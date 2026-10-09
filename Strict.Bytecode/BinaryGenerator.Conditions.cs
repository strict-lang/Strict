using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Bytecode;

public sealed partial class BinaryGenerator
{
	private void GenerateReturningInlineConditional(If inlineConditional)
	{
		GenerateCodeForIfCondition(inlineConditional.Condition);
		GenerateReturnInstruction(inlineConditional.Then);
		instructions.Add(new JumpToId(idStack.Pop(), InstructionType.JumpEnd));
		GenerateReturnInstruction(inlineConditional.OptionalElse!);
	}

	private void GenerateIfInstructions(If ifExpression)
	{
		GenerateCodeForIfCondition(ifExpression.Condition);
		GenerateCodeForThen(ifExpression);
		instructions.Add(new JumpToId(idStack.Pop(), InstructionType.JumpEnd));
		if (ifExpression.OptionalElse == null)
			return;
		idStack.Push(conditionalId);
		instructions.Add(new JumpToId(conditionalId++, InstructionType.JumpToIdIfTrue));
		GenerateInstructions([ifExpression.OptionalElse]);
		instructions.Add(new JumpToId(idStack.Pop(), InstructionType.JumpEnd));
	}

	private void GenerateCodeForThen(If ifExpression)
	{
		if (ifExpression.Then is Body thenBody)
			GenerateInstructions(thenBody.Expressions);
		else
			GenerateInstructions([ifExpression.Then]);
	}

	/// <summary>
	/// Number comparisons used as values (outside if conditions) become direct compare instructions,
	/// invoking Number.strict would recurse as its body is the same comparison. a >= b is not a < b.
	/// </summary>
	private bool TryGenerateNumberComparisonValue(Binary comparison)
	{
		var name = comparison.Method.Name;
		if (name is not (BinaryOperator.Smaller or BinaryOperator.Greater or BinaryOperator.SmallerOrEqual
				or BinaryOperator.GreaterOrEqual) || !comparison.Instance!.ReturnType.IsNumber)
			return false;
		GenerateInstructionFromExpression(comparison.Instance);
		var leftRegister = registry.PreviousRegister;
		GenerateInstructionFromExpression(comparison.Arguments[0]);
		instructions.Add(new BinaryInstruction(name is BinaryOperator.Greater or BinaryOperator.SmallerOrEqual
			? InstructionType.GreaterThan
			: InstructionType.LessThan, leftRegister, registry.PreviousRegister,
			registry.AllocateRegister()));
		if (name is BinaryOperator.Smaller or BinaryOperator.Greater)
			return true;
		var comparedRegister = registry.PreviousRegister;
		instructions.Add(new LoadConstantInstruction(registry.AllocateRegister(),
			new ValueInstance(comparison.ReturnType, false)));
		instructions.Add(new BinaryInstruction(InstructionType.Equal, comparedRegister,
			registry.PreviousRegister, registry.AllocateRegister()));
		return true;
	}

	private void GenerateCodeForBinary(MethodCall binaryExpression)
	{
		if (CanGenerateDirectBinaryInstruction(binaryExpression.Method.Name))
			GenerateBinaryInstruction(binaryExpression,
				GetInstructionBasedOnBinaryOperationName(binaryExpression.Method.Name));
	}

	private static bool CanGenerateDirectBinaryInstruction(string methodName) =>
		methodName is BinaryOperator.Plus or BinaryOperator.Minus or BinaryOperator.Multiply
			or BinaryOperator.Divide or BinaryOperator.Modulate or BinaryOperator.Is ||
		methodName.StartsWith("is not", StringComparison.Ordinal);

	private static InstructionType GetInstructionBasedOnBinaryOperationName(string binaryOperator) =>
		binaryOperator switch
		{
			BinaryOperator.Plus => InstructionType.Add,
			BinaryOperator.Multiply => InstructionType.Multiply,
			BinaryOperator.Minus => InstructionType.Subtract,
			BinaryOperator.Divide => InstructionType.Divide,
			BinaryOperator.Modulate => InstructionType.Modulo,
			BinaryOperator.Is => InstructionType.Equal,
			_ when binaryOperator.StartsWith("is not", StringComparison.Ordinal) => InstructionType.
				NotEqual,
			_ => throw new OperatorNotSupported(binaryOperator) //ncrunch: no coverage
		};

	private void GenerateCodeForIfCondition(Expression condition)
	{
		if (condition is MethodCall binaryCondition && IsBinaryComparison(binaryCondition))
			GenerateForBinaryIfConditionalExpression(binaryCondition);
		else
			GenerateForBooleanCallIfCondition(condition);
	}

	private static bool IsBinaryComparison(MethodCall call) =>
		call.Method.Name is BinaryOperator.Is or BinaryOperator.Greater or BinaryOperator.Smaller ||
		call.Method.Name.StartsWith("is not", StringComparison.Ordinal);

	private void GenerateForBinaryIfConditionalExpression(MethodCall condition)
	{
		var leftRegister = GenerateLeftSideForIfCondition(condition);
		var rightRegister = GenerateRightSideForIfCondition(condition);
		GenerateInstructionsFromIfCondition(GetConditionalInstruction(condition.Method), leftRegister,
			rightRegister);
	}

	private Register GenerateLeftSideForIfCondition(MethodCall condition) =>
		condition.Instance switch
		{
			MethodCall nestedMethodCall when IsBinaryOperation(nestedMethodCall.Method.Name) =>
				GenerateValueBinaryInstructions(nestedMethodCall,
					GetInstructionBasedOnBinaryOperationName(nestedMethodCall.Method.Name)),
			MethodCall nestedMethodCall => InvokeAndGetStoredRegisterForConditional(nestedMethodCall),
			_ => LoadVariableForIfConditionLeft(condition)
		};

	private static bool IsBinaryOperation(string methodName) =>
		methodName is BinaryOperator.Plus or BinaryOperator.Minus or BinaryOperator.Multiply
			or BinaryOperator.Divide or BinaryOperator.Modulate;

	private Register InvokeAndGetStoredRegisterForConditional(MethodCall condition)
	{
		GenerateInstructionFromExpression(condition);
		return registry.PreviousRegister;
	}

	private Register GenerateRightSideForIfCondition(MethodCall condition)
	{
		GenerateInstructionFromExpression(condition.Arguments[0]);
		return registry.PreviousRegister;
	}

	private void GenerateBinaryInstruction(MethodCall binaryExpression,
		InstructionType operationInstruction)
	{
		if (binaryExpression.Instance is MethodCall nestedBinary &&
			CanGenerateDirectBinaryInstruction(nestedBinary.Method.Name))
		{
			var leftRegister = GenerateValueBinaryInstructions(nestedBinary,
				GetInstructionBasedOnBinaryOperationName(nestedBinary.Method.Name));
			GenerateInstructionFromExpression(binaryExpression.Arguments[0]);
			instructions.Add(new BinaryInstruction(operationInstruction, leftRegister,
				registry.PreviousRegister, registry.AllocateRegister()));
		}
		else if (binaryExpression.Arguments[0] is MethodCall nestedBinaryArgument &&
			CanGenerateDirectBinaryInstruction(nestedBinaryArgument.Method.Name))
		{
			GenerateNestedBinaryInstructions(binaryExpression, operationInstruction,
				nestedBinaryArgument);
		}
		else
		{
			GenerateValueBinaryInstructions(binaryExpression, operationInstruction);
		}
	}

	private void GenerateNestedBinaryInstructions(MethodCall binaryExpression,
		InstructionType operationInstruction, MethodCall binaryArgument)
	{
		var right = GenerateValueBinaryInstructions(binaryArgument,
			GetInstructionBasedOnBinaryOperationName(binaryArgument.Method.Name));
		var left = registry.AllocateRegister();
		if (binaryExpression.Instance != null)
			instructions.Add(new LoadVariableToRegister(left, binaryExpression.Instance.ToString()));
		instructions.Add(new BinaryInstruction(operationInstruction, left, right,
			registry.AllocateRegister()));
	}

	private Register GenerateValueBinaryInstructions(MethodCall binaryExpression,
		InstructionType operationInstruction)
	{
		if (binaryExpression.Instance == null)
			throw new InstanceNameNotFound();
		GenerateInstructionFromExpression(binaryExpression.Instance);
		var leftValue = registry.PreviousRegister;
		GenerateInstructionFromExpression(binaryExpression.Arguments[0]);
		var rightValue = registry.PreviousRegister;
		var resultRegister = registry.AllocateRegister();
		instructions.Add(new BinaryInstruction(operationInstruction, leftValue, rightValue,
			resultRegister));
		return resultRegister;
	}

	private void GenerateSelectorIfInstructions(SelectorIf selectorIf)
	{
		foreach (var selectorCase in selectorIf.Cases)
		{
			GenerateCodeForIfCondition(selectorCase.Condition);
			GenerateInstructionFromExpression(selectorCase.Then);
			instructions.Add(new ReturnInstruction(registry.PreviousRegister));
			instructions.Add(new JumpToId(idStack.Pop(), InstructionType.JumpEnd));
		}
		if (selectorIf.OptionalElse != null)
		{
			GenerateInstructionFromExpression(selectorIf.OptionalElse);
			instructions.Add(new ReturnInstruction(registry.PreviousRegister));
		}
	}

	private void GenerateForBooleanCallIfCondition(Expression condition)
	{
		GenerateInstructionFromExpression(condition);
		var instanceCallRegister = registry.PreviousRegister;
		instructions.Add(new LoadConstantInstruction(registry.AllocateRegister(),
			new ValueInstance(condition.ReturnType, 1.0)));
		GenerateInstructionsFromIfCondition(InstructionType.Equal, instanceCallRegister,
			registry.PreviousRegister);
	}

	private void GenerateInstructionsFromIfCondition(InstructionType conditionInstruction,
		Register leftRegister, Register rightRegister)
	{
		instructions.Add(new BinaryInstruction(conditionInstruction, leftRegister, rightRegister));
		idStack.Push(conditionalId);
		instructions.Add(new JumpToId(conditionalId++, InstructionType.JumpToIdIfFalse));
	}

	private static InstructionType GetConditionalInstruction(Method condition) =>
		condition.Name switch
		{
			BinaryOperator.Greater => InstructionType.GreaterThan,
			BinaryOperator.Smaller => InstructionType.LessThan,
			_ when condition.Name.StartsWith("is not", StringComparison.Ordinal) => InstructionType.
				NotEqual,
			_ => InstructionType.Equal
		};

	private Register LoadVariableForIfConditionLeft(MethodCall condition)
	{
		if (condition.Instance != null)
			GenerateInstructionFromExpression(condition.Instance);
		return registry.PreviousRegister;
	}
}
