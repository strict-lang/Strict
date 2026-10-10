using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

public sealed partial class MethodCallEvaluator(Interpreter interpreter)
{
	public ValueInstance EvaluateListCall(ListCall call, ExecutionContext ctx)
	{
		interpreter.Statistics.ListCallCount++;
		var directOuter = TryGetDirectOuterValue(call.List, ctx);
		var listInstance = directOuter ?? interpreter.RunExpression(call.List, ctx);
		var index = (int)interpreter.RunExpression(call.Index, ctx).GetArithmeticNumber();
		if (listInstance.IsList || listInstance.IsText ||
			listInstance.TryGetValueTypeInstance()?.ReturnType.IsList == true)
		{
			var length = listInstance.GetIteratorLength();
			if (index < -length || index >= length)
				throw new Interpreter.ListIndexOutOfRange(ctx.Method, call.ToString(), index, length);
			return listInstance.GetIteratorValue(interpreter.characterType, index);
		}
		if (directOuter != null)
		{
			var typeInst = listInstance.TryGetValueTypeInstance();
			if (typeInst != null)
				for (var i = 0; i < typeInst.Values.Length; i++)
					if (typeInst.Values[i].IsText)
						return typeInst.Values[i].GetIteratorValue(interpreter.characterType, index);
		}
		var listInstanceText = listInstance.GetType().Name;
		try
		{
			listInstanceText = listInstance.ToString();
		}
		catch (Exception ex)
		{
			listInstanceText += ".ToString failed: " + ex;
		}
		throw new InterpreterExecutionFailed(ctx.Method, call.LineNumber,
			InterpreterExecutionFailed.BuildContextMessage(ctx.Method, call, ctx,
				"List call needs a list, got: " + listInstanceText), null, true);
	}

	/// <summary>
	/// isAssignedToInstance is list = list + element, only then a Mutable list changes in place.
	/// </summary>
	public ValueInstance Evaluate(MethodCall call, ExecutionContext ctx,
		bool isAssignedToInstance = false)
	{
		interpreter.Statistics.MethodCallCount++;
		var operatorType = GetOperatorCategory(call.Method.Name);
		if (operatorType != OperatorCategory.None)
			return EvaluateArithmeticOrCompareOrLogical(call, ctx, operatorType, isAssignedToInstance);
		var instance = call.Instance != null
			? TryGetDirectOuterValue(call.Instance, ctx) ?? interpreter.RunExpression(call.Instance, ctx)
			: call.Method.Name != Method.From
				? GetImplicitInstance(call.Method.Type, ctx)
				: null;
		return ExecuteMethodCall(call, instance, ctx);
	}

	private ValueInstance? GetImplicitInstance(Type methodType, ExecutionContext ctx)
	{
		var instance = ctx.This.HasValue && !ctx.This.Value.Equals(interpreter.noneInstance)
			? ctx.This
			: ctx.Parent?.Get(Type.ValueLowercase, interpreter.Statistics);
		return instance?.IsSameOrCanBeUsedAs(methodType) == true
			? instance
			: null;
	}

	private ValueInstance? TryGetDirectOuterValue(Expression expression, ExecutionContext ctx) =>
		expression switch
		{
			VariableCall { Variable.Name: Type.OuterLowercase } => ctx.Parent?.Get(Type.ValueLowercase,
				interpreter.Statistics),
			MemberCall
			{
				Instance: VariableCall { Variable.Name: Type.OuterLowercase },
				Member.Name: Type.ValueLowercase
			} => ctx.Parent?.Get(Type.ValueLowercase, interpreter.Statistics),
			_ => null
		};

	private enum OperatorCategory : byte
	{
		None,
		Arithmetic,
		Comparison,
		Logical
	}

	private static OperatorCategory GetOperatorCategory(string name) =>
		name switch
		{
			BinaryOperator.Plus or BinaryOperator.Minus or BinaryOperator.Multiply
				or BinaryOperator.Divide or BinaryOperator.Modulate
				or BinaryOperator.Power => OperatorCategory.Arithmetic,
			BinaryOperator.Greater or BinaryOperator.Smaller or BinaryOperator.Is
				or BinaryOperator.GreaterOrEqual
				or BinaryOperator.SmallerOrEqual => OperatorCategory.Comparison,
			BinaryOperator.And or BinaryOperator.Or or BinaryOperator.Xor or UnaryOperator.Not =>
				OperatorCategory.Logical,
			_ => OperatorCategory.None
		};

	private ValueInstance EvaluateArithmeticOrCompareOrLogical(MethodCall call, ExecutionContext ctx,
		OperatorCategory operatorType, bool isAssignedToInstance)
	{
		interpreter.Statistics.BinaryCount++;
		if (call.Instance == null || call.Arguments.Count != 1)
			throw new InterpreterExecutionFailed(ctx.Method, //ncrunch: no coverage
				InterpreterExecutionFailed.BuildContextMessage(ctx.Method, call, ctx,
					"Binary call must have instance and 1 argument"));
		var leftInstance = interpreter.RunExpression(call.Instance, ctx);
		if (IsDecidedByLeftSide(call.Method.Name, leftInstance))
			return leftInstance;
		var rightInstance = interpreter.RunExpression(call.Arguments[0], ctx);
		return operatorType switch
		{
			OperatorCategory.Arithmetic => ExecuteArithmeticOperation(call, ctx,
				isAssignedToInstance
					? leftInstance
					: Interpreter.CopyIfMutableList(leftInstance), rightInstance),
			OperatorCategory.Comparison => ExecuteComparisonOperation(call, ctx, leftInstance,
				rightInstance),
			OperatorCategory.Logical => ExecuteLogicalBinaryOperation(call, ctx, leftInstance,
				rightInstance),
			_ => throw new InterpreterExecutionFailed(ctx.Method, //ncrunch: no coverage
				InterpreterExecutionFailed.BuildContextMessage(ctx.Method, call, ctx,
					"Unknown operator category"))
		};
	}

	/// <summary>
	/// "and" is false when the left side is false, "or" true when it is true, the right is skipped.
	/// </summary>
	private static bool IsDecidedByLeftSide(string operatorName, ValueInstance left) =>
		operatorName is BinaryOperator.And or BinaryOperator.Or && left.GetType().IsBoolean &&
		left.Boolean == (operatorName == BinaryOperator.Or);

	private ValueInstance ExecuteArithmeticOperation(MethodCall call, ExecutionContext ctx,
		ValueInstance left, ValueInstance right)
	{
		while (true)
		{
			interpreter.Statistics.ArithmeticCount++;
			var op = call.Method.Name;
			if (op == BinaryOperator.Plus && left.IsPrimitiveType(interpreter.characterType) &&
				right.IsPrimitiveType(interpreter.characterType))
				return new ValueInstance(left.ToExpressionCodeString() + right.ToExpressionCodeString());
			if (op == BinaryOperator.Plus && left.IsPrimitiveType(interpreter.characterType) &&
				right.IsText)
				return new ValueInstance(left.ToExpressionCodeString() + right.Text);
			if (IsNumberLike(left) && IsNumberLike(right))
			{
				var l = left.GetArithmeticNumber();
				var r = right.GetArithmeticNumber();
				return op switch
				{
					BinaryOperator.Plus => new ValueInstance(interpreter.numberType, l + r),
					BinaryOperator.Minus => new ValueInstance(interpreter.numberType, l - r),
					BinaryOperator.Multiply => new ValueInstance(interpreter.numberType, l * r),
					BinaryOperator.Divide => new ValueInstance(interpreter.numberType, l / r),
					BinaryOperator.Modulate => new ValueInstance(interpreter.numberType, l % r),
					BinaryOperator.Power => new ValueInstance(interpreter.numberType, Math.Pow(l, r)),
					_ => ExecuteMethodCall(call, left, ctx) //ncrunch: no coverage
				};
			}
			if (left.IsText && right.IsText)
				return op == BinaryOperator.Plus
					? new ValueInstance(left.Text + right.Text)
					: throw new InterpreterExecutionFailed(ctx.Method,
						InterpreterExecutionFailed.BuildContextMessage(ctx.Method, call, ctx,
							"Only + operator is supported for Text, got: " + op));
			if (op == BinaryOperator.Plus && left.IsText && right.TryGetValueTypeInstance() != null)
				return new ValueInstance(left.Text + ConvertToTextWithOwnToMethod(right, ctx));
			if (left.IsText && IsNumberLike(right))
				return op == BinaryOperator.Plus
					? right.IsPrimitiveType(interpreter.characterType)
						? new ValueInstance(left.Text + right.ToExpressionCodeString())
						: new ValueInstance(left.Text + right.Number)
					: throw new InterpreterExecutionFailed(ctx.Method,
						InterpreterExecutionFailed.BuildContextMessage(ctx.Method, call, ctx,
							"Only + operator is supported for Text+Number, got: " + op));
			var leftList = ConvertToListValue(left);
			var rightList = ConvertToListValue(right);
			if (leftList.HasValue && rightList.HasValue)
			{
				if (op is BinaryOperator.Multiply or BinaryOperator.Divide &&
					leftList.Value.List.Items.Count != rightList.Value.List.Items.Count)
					return Error(ListsHaveDifferentDimensions, ctx, call);
				return op switch
				{
					BinaryOperator.Plus => CombineLists(leftList.Value,
						rightList.Value.List.Items, ctx, call),
					BinaryOperator.Minus => SubtractLists(leftList.Value, rightList.Value.List.Items),
					BinaryOperator.Multiply => MultiplyLists(leftList.Value.List.ReturnType,
						interpreter.numberType, leftList.Value.List.Items, rightList.Value.List.Items),
					BinaryOperator.Divide => DivideLists(leftList.Value.List.ReturnType,
						interpreter.numberType, leftList.Value.List.Items,
						rightList.Value.List.Items),
					_ => throw new InterpreterExecutionFailed(ctx.Method, //ncrunch: no coverage
						InterpreterExecutionFailed.BuildContextMessage(ctx.Method, call, ctx,
							"Only +, -, *, / operators are supported for Lists, got: " + op))
				};
			}
			if (leftList.HasValue && right.IsPrimitiveType(interpreter.numberType))
			{
				if (op == BinaryOperator.Plus)
					return AddToList(leftList.Value, right);
				if (op == BinaryOperator.Minus)
					return RemoveFromList(leftList.Value, right);
				if (op == BinaryOperator.Multiply)
					return MultiplyList(leftList.Value.List.ReturnType, leftList.Value.List.Items,
						right.Number);
				if (op == BinaryOperator.Divide)
					return DivideList(leftList.Value.List.ReturnType,
						leftList.Value.List.Items, right.Number);
				throw new InterpreterExecutionFailed(ctx.Method, //ncrunch: no coverage
					InterpreterExecutionFailed.BuildContextMessage(ctx.Method, call, ctx,
						"Only +, -, *, / operators are supported for List and Number, got: " + op));
			}
			if (leftList.HasValue && op == BinaryOperator.Plus)
				return AddToList(leftList.Value, right);
			if (leftList.HasValue && op == BinaryOperator.Minus)
				return RemoveFromList(leftList.Value, right);
			var unwrappedLeft = UnwrapValueMember(left);
			var unwrappedRight = UnwrapValueMember(right);
			if (!unwrappedLeft.Equals(left) || !unwrappedRight.Equals(right))
			{
				left = unwrappedLeft;
				right = unwrappedRight;
				continue;
			}
			if (IsCoreRuntimeType(call.Method.Type))
				throw new InterpreterExecutionFailed(ctx.Method,
					InterpreterExecutionFailed.BuildContextMessage(ctx.Method, call, ctx,
						BuildCoreTypeFallbackMessage(call, ctx, left, right)));
			return ExecuteMethodCall(call, left, ctx); //ncrunch: no coverage
		}
	}

	/// <summary>
	/// Text + any typed value uses that type's own "to Text" when it has one.
	/// </summary>
	private string ConvertToTextWithOwnToMethod(ValueInstance value, ExecutionContext ctx)
	{
		foreach (var method in value.GetType().Methods)
			if (method.Name == BinaryOperator.To && method.ReturnType.IsText)
				return interpreter.Execute(method, value, [], ctx).Text;
		return value.ToExpressionCodeString();
	}

	private static ValueInstance UnwrapValueMember(ValueInstance value)
	{
		if (value.TryGetValueTypeInstance() is not { } typeInstance)
			return value;
		return typeInstance.TryGetValue(Type.ValueLowercase, out var inner) &&
			(inner.IsText || inner.GetType().IsNumber || inner.GetType().IsBoolean)
				? inner
				: value;
	}

	private static bool IsCoreRuntimeType(Type type) =>
		type.IsList || type.IsDictionary || type.IsNumber || type.IsText || type.IsBoolean ||
		type.IsCharacter || type.Name == Type.Range;

	private bool IsNumberLike(ValueInstance value) => value.IsNumberLike(interpreter.numberType);

	public const string ListsHaveDifferentDimensions = "listsHaveDifferentDimensions";

	private ValueInstance ExecuteComparisonOperation(MethodCall call, ExecutionContext ctx,
		ValueInstance left, ValueInstance right)
	{
		interpreter.Statistics.CompareCount++;
		var op = call.Method.Name;
		if (op is BinaryOperator.Is)
		{
			var rightInstance = right.TryGetValueTypeInstance();
			if (rightInstance is { ReturnType.IsError: true })
			{
				var leftInstance = left.TryGetValueTypeInstance();
				var matches = leftInstance != null && leftInstance.ReturnType.IsError &&
					leftInstance.ReturnType.IsSameOrCanBeUsedAs(rightInstance.ReturnType);
				return interpreter.ToBoolean(matches);
			}
			if (left.IsPrimitiveType(interpreter.characterType) && right.IsText)
				right = new ValueInstance(interpreter.characterType, right.Text[0]);
			if (left.IsText && (right.IsPrimitiveType(interpreter.numberType) ||
				right.IsPrimitiveType(interpreter.characterType)))
				right = new ValueInstance(right.ToExpressionCodeString());
			if (ctx.IsTestAtCurrentLine && !left.IsText && left.GetType().IsNumber && !right.IsText &&
				right.GetType().IsNumber)
				return interpreter.ToBoolean(Math.Abs(left.Number - right.Number) < TestComparisonEpsilon);
			if (HasListType(left) && HasListType(right) && IsEmptyListTypeCheck(left, right))
				return interpreter.ToBoolean(left.GetType().FullName == right.GetType().FullName ||
					left.GetType().IsSameOrCanBeUsedAs(right.GetType()) ||
					right.GetType().IsSameOrCanBeUsedAs(left.GetType()));
			return interpreter.ToBoolean(left.Equals(right));
		}
		var l = left.GetArithmeticNumber();
		var r = right.GetArithmeticNumber();
		return op switch
		{
			BinaryOperator.Greater => interpreter.ToBoolean(l > r),
			BinaryOperator.Smaller => interpreter.ToBoolean(l < r),
			BinaryOperator.GreaterOrEqual => interpreter.ToBoolean(l >= r),
			BinaryOperator.SmallerOrEqual => interpreter.ToBoolean(l <= r),
			_ when IsCoreRuntimeType(call.Method.Type) => throw new InterpreterExecutionFailed(ctx.Method,
				InterpreterExecutionFailed.BuildContextMessage(ctx.Method, call, ctx,
					BuildCoreTypeFallbackMessage(call, ctx, left, right))),
			_ => ExecuteMethodCall(call, left, ctx) //ncrunch: no coverage
		};
	}

	private const double TestComparisonEpsilon = 0.00001;

	private ValueInstance ExecuteLogicalBinaryOperation(MethodCall call, ExecutionContext ctx,
		ValueInstance left, ValueInstance right)
	{
		interpreter.Statistics.LogicalOperationCount++;
		return call.Method.Name switch
		{
			BinaryOperator.And => interpreter.ToBoolean(left.Boolean && right.Boolean),
			BinaryOperator.Or => interpreter.ToBoolean(left.Boolean || right.Boolean),
			BinaryOperator.Xor => interpreter.ToBoolean(left.Boolean ^ right.Boolean),
			_ when IsCoreRuntimeType(call.Method.Type) => throw new InterpreterExecutionFailed(ctx.Method,
				InterpreterExecutionFailed.BuildContextMessage(ctx.Method, call, ctx,
					BuildCoreTypeFallbackMessage(call, ctx, left, right))),
			_ => ExecuteMethodCall(call, left, ctx) //ncrunch: no coverage
		};
	}

	private ValueInstance ExecuteMethodCall(MethodCall call, ValueInstance? instance,
		ExecutionContext ctx)
	{
		ValueInstance[] args;
		if (call.Arguments.Count == 0)
		{
			args = [];
		}
		else
		{
			args = new ValueInstance[call.Arguments.Count];
			for (var i = 0; i < call.Arguments.Count; i++)
			{
				var argument = interpreter.RunExpression(call.Arguments[i], ctx);
				args[i] = i < call.Method.Parameters.Count && call.Method.Parameters[i].IsMutable
					? argument
					: Interpreter.CopyIfMutableList(argument);
			}
		}
		if (instance is { IsDictionary: true } && args.Length > 0 && call.Method.Name == "Add")
		{
			if (args.Length == 2)
				instance.Value.GetDictionaryItems()[args[0]] = args[1];
			return instance.Value;
		}
		if (instance.HasValue && TryExecuteBuiltInMathMethod(call, instance.Value, out var mathResult))
			return mathResult;
		var capturedMutableParameters = TryRentMutableParameterCapture(call);
		var result = interpreter.Execute(call.Method, instance ?? interpreter.noneInstance, args, ctx,
			capturedMutableParameters: capturedMutableParameters);
		if (capturedMutableParameters != null)
			WriteBackMutableParameterValues(call, ctx, capturedMutableParameters);
		if (call.Method.ReturnType.IsMutable && !instance.Equals(interpreter.noneInstance))
		{
			if (call.Instance is VariableCall variableCall)
				ctx.Set(variableCall.Variable.Name, result);
			else if (call.Instance is ParameterCall parameterCall && parameterCall.Parameter.IsMutable)
				ctx.Set(parameterCall.Parameter.Name, result);
		}
		return result;
	}

	private static ValueInstance[]? TryRentMutableParameterCapture(MethodCall call)
	{
		for (var index = 0; index < call.Arguments.Count && index < call.Method.Parameters.Count;
			index++)
			if (call.Method.Parameters[index].IsMutable &&
				call.Arguments[index] is VariableCall { Variable.IsMutable: true })
				return new ValueInstance[call.Method.Parameters.Count];
		return null;
	}

	private static void WriteBackMutableParameterValues(MethodCall call, ExecutionContext ctx,
		ValueInstance[] capturedMutableParameters)
	{
		for (var index = 0; index < call.Arguments.Count && index < call.Method.Parameters.Count;
			index++)
			if (call.Method.Parameters[index].IsMutable && call.Arguments[index] is VariableCall
				{
					Variable.IsMutable: true
				} callerVariable)
				ctx.Set(callerVariable.Variable.Name, capturedMutableParameters[index]);
	}

	private bool TryExecuteBuiltInMathMethod(MethodCall call, ValueInstance instance,
		out ValueInstance result)
	{
		var typeName = call.Method.Type.Name;
		var methodName = call.Method.Name;
		if (typeName == "Degrees" && methodName is "Sin" or "Cos" or "Tan")
		{
			var radians = instance.GetArithmeticNumber() * Math.PI / 180.0;
			var value = methodName switch
			{
				"Sin" => Math.Sin(radians),
				"Cos" => Math.Cos(radians),
				_ => Math.Tan(radians)
			};
			result = new ValueInstance(interpreter.numberType, SnapNearInteger(value));
			return true;
		}
		if (typeName == "Ratio" && methodName is "Asin" or "Acos" or "Atan")
		{
			var inputValue = instance.GetArithmeticNumber();
			var degrees = methodName switch
			{
				"Asin" => Math.Asin(inputValue) * 180.0 / Math.PI,
				"Acos" => Math.Acos(inputValue) * 180.0 / Math.PI,
				_ => Math.Atan(inputValue) * 180.0 / Math.PI
			};
			result = new ValueInstance(interpreter.numberType, SnapNearInteger(degrees));
			return true;
		}
		if (typeName == "Vector2" && methodName == "Atan")
		{
			var typeInst = instance.TryGetValueTypeInstance();
			if (typeInst != null)
			{
				var numbersValue = typeInst.Values[0];
				if (numbersValue.IsList && numbersValue.List.Items.Count >= 2)
				{
					var x = numbersValue.List.Items[0].Number;
					var y = numbersValue.List.Items[1].Number;
					var degrees = Math.Atan2(x, y) * 180.0 / Math.PI;
					result = new ValueInstance(interpreter.numberType, SnapNearInteger(degrees));
					return true;
				}
			}
		}
		result = default;
		return false;
	}

	private static double SnapNearInteger(double value)
	{
		var rounded = Math.Round(value);
		return Math.Abs(value - rounded) < 1e-10
			? rounded
			: value;
	}
}
