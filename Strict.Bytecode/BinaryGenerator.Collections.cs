using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

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
		expression is To { Instance: { } inner }
			? inner
			: expression;

	private bool TryGenerateInstructionForCollectionManipulation(MethodCall methodCall)
	{
		switch (methodCall.Method.Name)
		{
		case "Add" when methodCall.Instance?.ReturnType.IsList == true ||
			methodCall.Instance?.ReturnType.IsDictionary == true:
			GenerateInstructionsForAddMethod(methodCall);
			return true;
		case "Remove" when methodCall.Instance?.ReturnType.IsList == true:
			GenerateInstructionsForRemoveMethod(methodCall);
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

	private void GenerateInstructionsForRemoveMethod(MethodCall methodCall)
	{
		if (methodCall.Instance == null)
			return;
		if (!OwnsList(methodCall.Instance))
			GenerateCopyingListChange(InstructionType.Subtract, methodCall.Instance.ToString(),
				methodCall.Arguments[0]);
		else
		{
			GenerateInstructionFromExpression(methodCall.Arguments[0]);
			instructions.Add(new RemoveInstruction(registry.PreviousRegister,
				methodCall.Instance.ToString()));
		}
	}

	private void GenerateInstructionsForAddMethod(MethodCall methodCall)
	{
		if (TryGenerateAddForTable(methodCall) || methodCall.Instance == null)
			return;
		if (OwnsList(methodCall.Instance))
			GenerateAppendToList(methodCall.Instance.ToString(), methodCall.Arguments[0]);
		else
			GenerateCopyingListChange(InstructionType.Add, methodCall.Instance.ToString(),
				methodCall.Arguments[0]);
	}

	/// <summary>
	/// In place changes are only allowed on lists the variable created itself, a variable initialized
	/// from another variable, parameter or member shares that list and gets a changed copy instead.
	/// </summary>
	private static bool OwnsList(Expression list) =>
		list is not VariableCall variableCall || !IsNamedValue(variableCall.Variable.InitialValue);

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

	private void GenerateAppendToList(string listName, Expression element)
	{
		GenerateInstructionFromExpression(element);
		instructions.Add(new WriteToListInstruction(registry.PreviousRegister, listName));
		instructions.Add(new LoadVariableToRegister(registry.AllocateRegister(), listName));
	}

	/// <summary>
	/// list = list + element is what List.Add does, so it appends in place instead of copying.
	/// A list shared with another variable is never changed in place, see <see cref="OwnsList"/>.
	/// </summary>
	private static bool IsAppendToSameList(MutableReassignment reassignment) =>
		reassignment.Value is Binary
		{
			Method.Name: BinaryOperator.Plus, Instance: VariableCall or ParameterCall
		} binary && binary.Instance.ReturnType.IsList && binary.Instance.ToString() == reassignment.Name &&
		!binary.Method.Parameters[0].Type.IsList && OwnsList(binary.Instance);

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
