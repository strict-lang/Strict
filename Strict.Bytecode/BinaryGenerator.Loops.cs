using Strict.Bytecode.Instructions;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Bytecode;

public sealed partial class BinaryGenerator
{
	private bool TryGenerateSumForLoopReturn(Expression expression)
	{
		if (expression is not For forExpression || !ReturnType.IsNumber && !ReturnType.IsText)
			return false;
		GenerateInstructionForSumAggregation(forExpression);
		return true;
	}

	private bool TryGenerateListForLoopReturn(Expression expression)
	{
		if (expression is not For forExpression || !ShouldAggregateLoopToList())
			return false;
		GenerateInstructionForListAggregation(forExpression);
		instructions.Add(new ReturnInstruction(registry.PreviousRegister));
		return true;
	}

	private void GenerateInstructionForSumAggregation(For forExpression)
	{
		var resultVariable = $"forResult{forResultId++}";
		instructions.Add(
			new StoreVariableInstruction(ReturnType.IsText
				? new ValueInstance("")
				: new ValueInstance(ReturnType, 0), resultVariable));
		GenerateLoopInstructions(forExpression, resultVariable, LoopAggregation.Number);
		instructions.Add(new LoadVariableToRegister(registry.AllocateRegister(), resultVariable));
		instructions.Add(new ReturnInstruction(registry.PreviousRegister));
	}

	private void GenerateInstructionForListAggregation(For forExpression)
	{
		var resultVariable = $"forResult{forResultId++}";
		var listType = GetListType(forExpression.Body.ReturnType);
		//TODO: why does this create a new ValueInstance, no good, especially the array version!
		instructions.Add(
			new StoreVariableInstruction(new ValueInstance(listType, Array.Empty<ValueInstance>()),
				resultVariable));
		GenerateLoopInstructions(forExpression, resultVariable, LoopAggregation.List);
		instructions.Add(new LoadVariableToRegister(registry.AllocateRegister(), resultVariable));
	}

	private bool ShouldAggregateLoopToList() => ReturnType.IsIterator || ReturnType.IsList;

	private Type GetListType(Type elementType) =>
		binary.basePackage.FindType(Type.List)?.GetGenericImplementation(elementType) ??
		throw new ListTypeNotFound(elementType);

	private void GenerateLoopInstructions(For forExpression, string? aggregationTarget = null,
		LoopAggregation aggregation = LoopAggregation.None)
	{
		if (aggregation != LoopAggregation.None)
			Disown(forExpression, true);
		CopyElementWrittenListsBeforeLoop(forExpression, aggregation != LoopAggregation.None);
		var ownedAtLoopStart = ownedLists.ToHashSet();
		var forSourceLine = forExpression.LineNumber;
		var instructionCountBeforeLoopStart = instructions.Count;
		var customVariableNames =
			forExpression.CustomVariables.Select(variable => variable.ToString()).ToArray();
		var iterator = GetLoopIteratorExpression(forExpression);
		LoopBeginInstruction loopBegin;
		if (iterator.ReturnType.Name == Type.Range)
		{
			loopBegin = GenerateInstructionForRangeLoopInstruction(iterator, customVariableNames);
			loopBegin.SourceLine = forSourceLine;
		}
		else
		{
			var iteratorStart = instructions.Count;
			GenerateInstructionFromExpression(iterator);
			for (var instructionIndex = iteratorStart; instructionIndex < instructions.Count;
				instructionIndex++)
				instructions[instructionIndex].SourceLine = forSourceLine;
			loopBegin = new LoopBeginInstruction(registry.PreviousRegister, customVariableNames);
			loopBegin.SourceLine = forSourceLine;
			instructions.Add(loopBegin);
		}
		var bodyAggregatedDirectly =
			GenerateInstructionsForLoopBody(forExpression, aggregationTarget, aggregation);
		if (!string.IsNullOrWhiteSpace(aggregationTarget) && !bodyAggregatedDirectly)
			AddLoopAggregation(aggregationTarget, aggregation);
		var loopEnd = new LoopEndInstruction(instructions.Count - instructionCountBeforeLoopStart)
		{
			Begin = loopBegin, SourceLine = forSourceLine
		};
		instructions.Add(loopEnd);
		ownedLists.IntersectWith(ownedAtLoopStart);
	}

	private void AddLoopAggregation(string aggregationTarget, LoopAggregation aggregation)
	{
		switch (aggregation)
		{
		case LoopAggregation.None:
			break;
		case LoopAggregation.Number:
			AddNumberAggregation(aggregationTarget);
			break;
		case LoopAggregation.List:
			AddListAggregation(aggregationTarget);
			break;
		case LoopAggregation.Any:
			ReturnTrueIfIterationIsTrue();
			break;
		}
	}

	private void ReturnTrueIfIterationIsTrue()
	{
		var iterationRegister = registry.PreviousRegister;
		instructions.Add(new LoadConstantInstruction(registry.AllocateRegister(),
			new ValueInstance(ReturnType, true)));
		var trueRegister = registry.PreviousRegister;
		GenerateInstructionsFromIfCondition(InstructionType.Equal, iterationRegister, trueRegister);
		instructions.Add(new ReturnInstruction(trueRegister));
		instructions.Add(new JumpToId(idStack.Pop(), InstructionType.JumpEnd));
	}

	private void AddNumberAggregation(string aggregationTarget)
	{
		var loopValueRegister = registry.PreviousRegister;
		instructions.Add(new LoadVariableToRegister(registry.AllocateRegister(), aggregationTarget));
		var accumulatorRegister = registry.PreviousRegister;
		instructions.Add(new BinaryInstruction(InstructionType.Add, accumulatorRegister,
			loopValueRegister, registry.AllocateRegister()));
		instructions.Add(
			new StoreFromRegisterInstruction(registry.PreviousRegister, aggregationTarget));
	}

	private void AddListAggregation(string aggregationTarget) =>
		instructions.Add(new WriteToListInstruction(registry.PreviousRegister, aggregationTarget));

	/// <summary>
	/// Custom iterators build all their elements, loops only using index count to Length instead.
	/// </summary>
	private static Expression GetLoopIteratorExpression(For forExpression)
	{
		var iterator = forExpression.Iterator;
		var type = iterator.ReturnType;
		if (type.IsList || type.IsText || type.IsNumber || type.Name == Type.Range)
			return iterator;
		var iteratorMethod = type.Methods.FirstOrDefault(method =>
			method.Name == Keyword.For && method.ReturnType.IsIterator);
		if (iteratorMethod == null)
			return iterator;
		var lengthMethod = type.Methods.FirstOrDefault(method =>
			method is { Name: "Length", Parameters.Count: 0, ReturnType.IsNumber: true });
		if (lengthMethod != null && forExpression.CustomVariables.Length == 0 &&
			!UsesLoopValue(forExpression.Body))
			iteratorMethod = lengthMethod;
		return new MethodCall(iteratorMethod, iterator, lineNumber: iterator.LineNumber);
	}

	/// <summary>
	/// Unknown expressions count as using the loop value, only proven index-only loops change.
	/// </summary>
	private static bool UsesLoopValue(Expression expression) =>
		expression switch
		{
			VariableCall variableCall => variableCall.Variable.Name is Type.ValueLowercase
				or Type.OuterLowercase,
			List list => list.Values.Any(UsesLoopValue),
			Value or ParameterCall => false,
			MemberCall memberCall => memberCall.Instance != null && UsesLoopValue(memberCall.Instance),
			MethodCall methodCall => methodCall.Instance != null && UsesLoopValue(methodCall.Instance) ||
				methodCall.Arguments.Any(UsesLoopValue),
			Body body => body.Expressions.Any(UsesLoopValue),
			Declaration declaration => UsesLoopValue(declaration.Value),
			MutableReassignment reassignment => UsesLoopValue(reassignment.Target) ||
				UsesLoopValue(reassignment.Value),
			If ifExpression => UsesLoopValue(ifExpression.Condition) ||
				UsesLoopValue(ifExpression.Then) ||
				ifExpression.OptionalElse != null && UsesLoopValue(ifExpression.OptionalElse),
			ListCall listCall => UsesLoopValue(listCall.List) || UsesLoopValue(listCall.Index),
			_ => true
		};

	private LoopBeginInstruction GenerateInstructionForRangeLoopInstruction(
		Expression range, params string[] customVariableNames)
	{
		var (startIndexRegister, endIndexRegister) =
			range is MethodCall { Method.Name: Method.From } creation
				? (GenerateRegister(creation.Arguments[0]), GenerateRegister(creation.Arguments[1]))
				: GenerateRangeFieldLoads(GenerateRegister(range));
		var loopBegin = new LoopBeginInstruction(startIndexRegister, endIndexRegister,
			customVariableNames);
		instructions.Add(loopBegin);
		return loopBegin;
	}

	private Register GenerateRegister(Expression expression)
	{
		GenerateInstructionFromExpression(expression);
		return registry.PreviousRegister;
	}

	private (Register Start, Register End) GenerateRangeFieldLoads(Register rangeRegister)
	{
		instructions.Add(new FieldLoadInstruction(registry.AllocateRegister(), rangeRegister,
			"Start"));
		var startRegister = registry.PreviousRegister;
		instructions.Add(new FieldLoadInstruction(registry.AllocateRegister(), rangeRegister,
			"ExclusiveEnd"));
		return (startRegister, registry.PreviousRegister);
	}

	private bool GenerateInstructionsForLoopBody(For forExpression, string? aggregationTarget,
		LoopAggregation aggregation)
	{
		if (aggregation == LoopAggregation.List && !string.IsNullOrWhiteSpace(aggregationTarget) &&
			forExpression.Body is For directNestedFor)
		{
			GenerateLoopInstructions(directNestedFor, aggregationTarget, aggregation);
			return true;
		}
		// for x; if cond; value  → only aggregate when the if-then branch runs
		if (aggregation is LoopAggregation.List or LoopAggregation.Number &&
			!string.IsNullOrWhiteSpace(aggregationTarget) && forExpression.Body is If ifInLoop)
		{
			GenerateIfThenAggregation(ifInLoop, aggregationTarget, aggregation);
			return true;
		}
		if (forExpression.Body is Body forExpressionBody)
			for (var expressionIndex = 0; expressionIndex < forExpressionBody.Expressions.Count;
				expressionIndex++)
			{
				var expression = forExpressionBody.Expressions[expressionIndex];
				if (aggregation == LoopAggregation.List &&
					expressionIndex == forExpressionBody.Expressions.Count - 1 &&
					expression is For nestedFor && !string.IsNullOrWhiteSpace(aggregationTarget))
				{
					GenerateLoopInstructions(nestedFor, aggregationTarget, aggregation);
					return true;
				}
				if (aggregation is LoopAggregation.List or LoopAggregation.Number &&
					expressionIndex == forExpressionBody.Expressions.Count - 1 &&
					expression is If ifExpression && !string.IsNullOrWhiteSpace(aggregationTarget))
				{
					GenerateIfThenAggregation(ifExpression, aggregationTarget, aggregation);
					return true;
				}
				Disown(expression, false);
				GenerateInstructionFromExpression(expression);
			}
		else
			GenerateInstructionFromExpression(forExpression.Body);
		return false;
	}

	/// <summary>
	/// Emits if-condition + then-body + aggregation only on the then path, so filtered loops
	/// (`for xs; if cond; map(value)` or a counting `1`) do not aggregate on false branches.
	/// </summary>
	private void GenerateIfThenAggregation(If ifExpression, string aggregationTarget,
		LoopAggregation aggregation)
	{
		GenerateCodeForIfCondition(ifExpression.Condition);
		GenerateCodeForThen(ifExpression);
		AddLoopAggregation(aggregationTarget, aggregation);
		instructions.Add(new JumpToId(idStack.Pop(), InstructionType.JumpEnd));
		if (ifExpression.OptionalElse == null)
			return;
		idStack.Push(conditionalId);
		instructions.Add(new JumpToId(conditionalId++, InstructionType.JumpToIdIfTrue));
		GenerateInstructions([ifExpression.OptionalElse]);
		instructions.Add(new JumpToId(idStack.Pop(), InstructionType.JumpEnd));
	}
}
