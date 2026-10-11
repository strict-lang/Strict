using Strict.Bytecode.Instructions;
using Strict.Expressions;
using Strict.Language;

namespace Strict.Bytecode;

public sealed partial class BinaryGenerator
{
	private void GenerateListExpression(List list)
	{
		if (list.TryGetConstantData() is { } constantList)
		{
			instructions.Add(new LoadConstantInstruction(registry.AllocateRegister(), constantList));
			return;
		}
		var listVariable = $"listResult{listResultId++}";
		instructions.Add(new StoreVariableInstruction(
			new ValueInstance(list.ReturnType, Array.Empty<ValueInstance>()), listVariable));
		for (var valueIndex = 0; valueIndex < list.Values.Count; valueIndex++)
		{
			GenerateInstructionFromExpression(list.Values[valueIndex]);
			instructions.Add(new WriteToListInstruction(registry.PreviousRegister, listVariable));
		}
		instructions.Add(new LoadVariableToRegister(registry.AllocateRegister(), listVariable));
	}

	private static string ExtractTextPrefix(Expression? expression) =>
		expression switch
		{
			Value value when value.Data.IsText => value.Data.Text,
			To { Instance: { } inner } => ExtractTextPrefix(inner),
			_ => ""
		};

	private static Expression UnwrapToConversion(Expression expression) =>
		expression is To { Instance: { } inner, ConversionType.IsText: true }
			? inner
			: expression;

	private bool TryGenerateInstructionForCollectionManipulation(MethodCall methodCall)
	{
		switch (methodCall.Method.Name)
		{
		case "Add" when methodCall.Instance?.ReturnType.IsList == true ||
			methodCall.Instance?.ReturnType.IsDictionary == true:
			GenerateListChange(methodCall, false);
			return true;
		case "Remove" when methodCall.Instance?.ReturnType.IsList == true:
			GenerateListChange(methodCall, true);
			return true;
		case "Increment":
		case "Decrement":
			GenerateIncrementDecrementInvoke(methodCall);
			return true;
		default:
			return false;
		}
	}

	private void GenerateIncrementDecrementInvoke(MethodCall methodCall)
	{
		Register? instanceRegister = null;
		if (methodCall.Instance != null)
		{
			GenerateInstructionFromExpression(methodCall.Instance);
			instanceRegister = registry.PreviousRegister;
		}
		var parameterNames = new string[methodCall.Method.Parameters.Count];
		for (var paramIndex = 0; paramIndex < methodCall.Method.Parameters.Count; paramIndex++)
			parameterNames[paramIndex] = methodCall.Method.Parameters[paramIndex].Name;
		var methodInfo = new InvokeMethodInfo(methodCall.Method.Type.FullName, methodCall.Method.Name,
			parameterNames, GetBinaryTypeName(methodCall.ReturnType, methodCall.Method.Type), [],
			instanceRegister);
		var resultRegister = registry.AllocateRegister();
		instructions.Add(new Invoke(resultRegister, methodInfo));
		if (methodCall.Instance != null)
			instructions.Add(new StoreFromRegisterInstruction(resultRegister,
				methodCall.Instance.ToString()));
	}

	private void GenerateListChange(MethodCall methodCall, bool isRemove)
	{
		if (TryGenerateAddForTable(methodCall))
			return;
		var listName = methodCall.Instance!.ToString();
		if (OwnsList(methodCall.Instance))
			GenerateInPlaceListChange(listName, methodCall.Arguments[0], isRemove);
		else
			GenerateCopyingListChange(isRemove
				? InstructionType.Subtract
				: InstructionType.Add, listName, methodCall.Arguments[0]);
	}

	/// <summary>
	/// Lists are values: a variable only changes its list in place while it owns it, it was declared
	/// with a new list that was not shared with any variable, member, argument or list since then.
	/// </summary>
	private readonly HashSet<string> ownedLists = new(StringComparer.Ordinal);

	private bool OwnsList(Expression list) =>
		list is not VariableCall variableCall || ownedLists.Contains(variableCall.Variable.Name);

	private static bool IsNewList(Expression value) => value is Value or Binary;

	/// <summary>
	/// Called before each statement, sharing a list variable or assigning another list to it ends
	/// its ownership. Loops are checked as a whole, their later lines run before earlier ones too.
	/// </summary>
	private void Disown(Expression expression, bool isValueUsed)
	{
		switch (expression)
		{
		case VariableCall variableCall when isValueUsed:
			ownedLists.Remove(variableCall.Variable.Name);
			break;
		case Body body:
			for (var index = 0; index < body.Expressions.Count; index++)
				Disown(body.Expressions[index], isValueUsed && index == body.Expressions.Count - 1);
			break;
		case Declaration declaration:
			ownedLists.Remove(declaration.Name);
			// ReSharper disable TailRecursiveCall
			Disown(declaration.Value, true);
			break;
		case MutableReassignment reassignment:
			if (!IsNewList(reassignment.Value))
				ownedLists.Remove(reassignment.Name);
			Disown(reassignment.Value, true);
			break;
		case Return returnExpression:
			Disown(returnExpression.Value, true);
			break;
		case For forExpression:
			Disown(forExpression.Iterator, true);
			Disown(forExpression.Body, isValueUsed);
			break;
		case If ifExpression:
			Disown(ifExpression.Condition, false);
			Disown(ifExpression.Then, isValueUsed);
			if (ifExpression.OptionalElse != null)
				Disown(ifExpression.OptionalElse, isValueUsed);
			break;
		case SelectorIf selectorIf:
			Disown(selectorIf.Selector, false);
			foreach (var selectorCase in selectorIf.Cases)
				Disown(selectorCase.Then, isValueUsed);
			if (selectorIf.OptionalElse != null)
				Disown(selectorIf.OptionalElse, isValueUsed);
			break;
		case ListCall listCall:
			Disown(listCall.List, false);
			Disown(listCall.Index, false);
			break;
		case MemberCall { Instance: { } instance }:
			Disown(instance, false);
			break;
		case MethodCall methodCall:
			if (methodCall.Instance != null)
				Disown(methodCall.Instance, isValueUsed && methodCall.Method.Name is "Add" or "Remove");
			foreach (var argument in methodCall.Arguments)
				Disown(argument, true);
			break;
		case List list:
			foreach (var element in list.Values)
				Disown(element, true);
			break;
		}
	}

	private void GenerateCopyingListChange(InstructionType operation, string listName,
		Expression element)
	{
		instructions.Add(new LoadVariableToRegister(registry.AllocateRegister(), listName));
		var list = registry.PreviousRegister;
		GenerateInstructionFromExpression(element);
		instructions.Add(new BinaryInstruction(operation, list, registry.PreviousRegister,
			registry.AllocateRegister()));
		instructions.Add(new StoreFromRegisterInstruction(registry.PreviousRegister, listName));
	}

	/// <summary>
	/// An element write changes the list in place, a list the variable does not own is still used
	/// by the variable, member or list it came from: it is copied first and owned afterwards.
	/// </summary>
	private void CopyListIfNotOwned(Expression target)
	{
		if (target is not ListCall { List: VariableCall listVariable } || OwnsList(listVariable))
			return;
		instructions.Add(new CopyListInstruction(listVariable.Variable.Name));
		ownedLists.Add(listVariable.Variable.Name);
	}

	/// <summary>
	/// Element writes in a loop copy such a list once before the loop, unless the loop shares the
	/// list itself, then every write copies it (bad code, the old version is used again).
	/// </summary>
	private void CopyElementWrittenListsBeforeLoop(For forExpression, bool isValueUsed)
	{
		var copied = ElementWrittenLists(forExpression.Body).Distinct().
			Where(name => !ownedLists.Contains(name)).ToList();
		ownedLists.UnionWith(copied);
		Disown(forExpression, isValueUsed);
		foreach (var name in copied.Where(ownedLists.Contains))
			instructions.Add(new CopyListInstruction(name));
	}

	private static IEnumerable<string> ElementWrittenLists(Expression expression) =>
		expression switch
		{
			MutableReassignment { Target: ListCall { List: VariableCall listVariable } } =>
				[listVariable.Variable.Name],
			Body body => body.Expressions.SelectMany(ElementWrittenLists),
			For nestedFor => ElementWrittenLists(nestedFor.Body),
			If ifExpression => ElementWrittenLists(ifExpression.Then).Concat(
				ifExpression.OptionalElse is { } optionalElse
					? ElementWrittenLists(optionalElse)
					: []),
			_ => []
		};

	private void GenerateInPlaceListChange(string listName, Expression element, bool isRemove)
	{
		GenerateInstructionFromExpression(element);
		instructions.Add(isRemove
			? new RemoveInstruction(registry.PreviousRegister, listName)
			: new WriteToListInstruction(registry.PreviousRegister, listName));
		instructions.Add(new LoadVariableToRegister(registry.AllocateRegister(), listName));
	}

	/// <summary>
	/// list = list + element is what List.Add does, so it appends in place instead of copying.
	/// A list shared with another variable is never changed in place, see <see cref="OwnsList"/>.
	/// </summary>
	private bool IsAppendToSameList(MutableReassignment reassignment) =>
		reassignment.Value is Binary
		{
			Method.Name: BinaryOperator.Plus, Instance: VariableCall or ParameterCall
		} plusBinary && plusBinary.Instance.ReturnType.IsList &&
		plusBinary.Instance.ToString() == reassignment.Name &&
		!plusBinary.Method.Parameters[0].Type.IsList && OwnsList(plusBinary.Instance);

	private bool TryGenerateAddForTable(MethodCall methodCall)
	{
		if (methodCall.Arguments.Count != 2 || methodCall.Instance == null)
			return false;
		GenerateInstructionFromExpression(methodCall.Arguments[0]);
		var key = registry.PreviousRegister;
		GenerateInstructionFromExpression(methodCall.Arguments[1]);
		var value = registry.PreviousRegister;
		instructions.Add(new WriteToTableInstruction(key, value, methodCall.Instance.ToString()));
		return true;
	}

	private void GenerateForAssignmentOrDeclaration(Expression declarationOrAssignment, string name)
	{
		if (declarationOrAssignment is Value declarationOrAssignmentValue &&
			(declarationOrAssignment is not List list || list.TryGetConstantData() != null))
		{
			TryGenerateInstructionsForAssignmentValue(declarationOrAssignmentValue, name);
		}
		else
		{
			GenerateInstructionFromExpression(declarationOrAssignment);
			instructions.Add(new StoreFromRegisterInstruction(registry.PreviousRegister, name));
		}
	}

	private void TryGenerateInstructionsForAssignmentValue(Value assignmentValue, string variableName)
	{
		var data = assignmentValue.ReturnType.IsDictionary
			? new ValueInstance(assignmentValue.ReturnType,
				new Dictionary<ValueInstance, ValueInstance>())
			: GetValueInstanceFromExpression(assignmentValue);
		instructions.Add(new StoreVariableInstruction(data, variableName));
	}
}
