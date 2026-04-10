using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict;

public sealed partial class VirtualMachine
{
	/// <summary>
	/// Evaluates an arbitrary expression to a ValueInstance using the current VM state.
	/// Handles values, variables, member calls, binary operations, and method calls.
	/// </summary>
	//TODO: this is called 4 million times, stupid!
	private ValueInstance EvaluateExpression(Expression expression)
	{
		if (expression is Value value)
			return value.Data;
		if (expression is VariableCall variableCall && variableCall.IsConstant &&
			variableCall.Variable.InitialValue is Value constantValue)
			return constantValue.Data;
		if (expression is VariableCall directVariable)
			return EvaluateVariableCall(directVariable);
		if (expression is ParameterCall parameterCall)
			return Memory.Frame.Get(CallFrame.ResolveSymbolId(parameterCall.Parameter.Name));
		if (expression is Instance)
			return Memory.Frame.Get(ValueSymbolId);
		if (expression is MemberCall memberCall)
			return EvaluateMemberCall(memberCall);
		if (expression is Binary binary)
			return EvaluateBinary(binary); //TODO: eats up 75% of the time here! 0.7m calls
		if (expression is MethodCall methodCall)
			return EvaluateMethodCall(methodCall);
		if (expression is ListCall listCall)
			return EvaluateListCallExpression(listCall); //TODO: another almost 25% here, 0.23m calls
		throw new InvalidOperationException("Could not evaluate expression " + expression + " (" +
			expression.GetType().Name + ")");
	}

	private ValueInstance EvaluateVariableCall(VariableCall variableCall) =>
		Memory.Frame.Get(variableCall.Variable.Name == Type.OuterLowercase
			? OuterSymbolId
			: CallFrame.ResolveSymbolId(variableCall.Variable.Name));

	private ValueInstance EvaluateListCallExpression(ListCall listCall)
	{
		var listValue = EvaluateExpression(listCall.List);
		var indexValue = EvaluateExpression(listCall.Index);
		var index = (int)indexValue.Number;
		if (listValue.IsList || listValue.IsText ||
			listValue.TryGetValueTypeInstance()?.ReturnType.IsList == true)
			return listValue.GetIteratorValue(executable.characterType, index);
		if (listValue.TryGetValueTypeInstance() is { } typeInstance)
		{
			if (typeInstance.TryGetValue(Type.ElementsLowercase, out var elementsValue) &&
				(elementsValue.IsList || elementsValue.IsText))
				return elementsValue.GetIteratorValue(executable.characterType, index);
			for (var valueIndex = 0; valueIndex < typeInstance.Values.Length; valueIndex++)
				if (typeInstance.Values[valueIndex].IsText)
					return typeInstance.Values[valueIndex].
						GetIteratorValue(executable.characterType, index);
		}
		throw new InvalidOperationException("Could not evaluate list call " + listCall);
	}

	private ValueInstance EvaluateMemberCall(MemberCall memberCall)
	{
		if (memberCall.Instance == null)
		{
			if (memberCall.Member.InitialValue is Value enumValue)
				return enumValue.Data;
			return TryGetFrameValue(CallFrame.ResolveSymbolId(memberCall.Member.Name),
				out var scopedMemberValue)
				? scopedMemberValue
				: throw new InvalidOperationException("Could not resolve member " + memberCall.Member.Name);
		}
		var instanceValue = EvaluateExpression(memberCall.Instance);
		if (TryGetNativeLength(instanceValue, memberCall.Member.Name, out var lengthValue))
			return lengthValue;
		if (instanceValue.TryGetFlatNumericMember(memberCall.Member.Name, out var flatMemberValue))
			return flatMemberValue;
		var typeInstance = instanceValue.TryGetValueTypeInstance();
		if (typeInstance != null &&
			typeInstance.TryGetValue(memberCall.Member.Name, out var memberValue))
			return memberValue;
		if (instanceValue.IsText && memberCall.Member.Name is "characters" or "elements")
			return instanceValue;
		throw new InvalidOperationException("Could not evaluate member call " + memberCall);
	}

	//TODO: cumbersome, simplify in a few lines
	private bool TryGetNativeLength(ValueInstance instance, string memberName, out ValueInstance result)
	{
		if (memberName is "Length" or "Count")
		{
			if (instance.IsText)
			{
				result = new ValueInstance(executable.numberType, instance.Text.Length);
				return true;
			}
			if (instance.IsList)
			{
				result = new ValueInstance(executable.numberType, instance.List.Count);
				return true;
			}
		}
		result = default;
		return false;
	}

	private ValueInstance EvaluateBinary(Binary binary)
	{
		var left = EvaluateExpression(binary.Instance!);
		var right = EvaluateExpression(binary.Arguments[0]);
		return binary.Method.Name switch
		{
			BinaryOperator.Plus => AddValueInstances(left, right),
			BinaryOperator.Minus => SubtractValueInstances(left, right),
			BinaryOperator.Multiply => new ValueInstance(right.GetType(),
				left.Number * right.Number),
			BinaryOperator.Divide => new ValueInstance(right.GetType(),
				left.Number / right.Number),
			_ => throw new InvalidOperationException("Unsupported binary operator: " + binary.Method.Name)
		};
	}

	private ValueInstance EvaluateMethodCall(MethodCall call)
	{
		if (call.Method.Name == Method.From)
			return EvaluateFromConstructor(call);
		if (call.Method.Name == BinaryOperator.To && call.Instance != null)
		{
			var rawValue = EvaluateExpression(call.Instance);
			if (call.ReturnType.IsText)
				return ConvertToText(rawValue);
			if (call.ReturnType.IsNumber)
				return rawValue.IsText
					? new ValueInstance(call.ReturnType, Convert.ToDouble(rawValue.Text))
					: rawValue;
		}
		var precompiledInstructions = GetPrecompiledMethodInstructions(call.Method);
		if (precompiledInstructions != null)
		{
			var evaluatedArguments = call.Arguments.Select(EvaluateExpression).ToArray();
			var evaluatedInstance = call.Instance != null
				? EvaluateExpression(call.Instance)
				: (ValueInstance?)null;
			var childScope = InitializeChildScope();
			InitializeMethodCallScopeFromExpressions(call, evaluatedArguments, evaluatedInstance);
			RunInstructions(precompiledInstructions);
			var precompiledResult = Returns;
			CleanupChildScope(childScope);
			return precompiledResult ?? new ValueInstance(call.Method.ReturnType, 0);
		}
		return TryEvaluateMethodCallFromBody(call) ??
			throw new InvalidOperationException(
				"No precompiled method instructions found for method call: " + call);
	}

	private ValueInstance? TryEvaluateMethodCallFromBody(MethodCall call)
	{
		var methodBody = call.Method.GetBodyAndParseIfNeeded();
		if (methodBody is not Body body)
			return null;
		var evaluatedInstance = call.Instance != null
			? EvaluateExpression(call.Instance)
			: (ValueInstance?)null;
		var savedFrame = Memory.Frame;
		Memory.Frame = new CallFrame(savedFrame);
		try
		{
			if (evaluatedInstance.HasValue)
			{
				Memory.Frame.Set(Type.ValueLowercase, evaluatedInstance.Value, isMember: true);
				var flatNumericInstance = evaluatedInstance.Value.TryGetFlatNumericArrayInstance();
				if (flatNumericInstance != null)
				{
					var flatMembers = flatNumericInstance.ReturnType.Members;
					for (var memberIndex = 0; memberIndex < flatMembers.Count &&
						memberIndex < flatNumericInstance.FlatWidth; memberIndex++)
						if (!flatMembers[memberIndex].Type.IsTrait)
							Memory.Frame.Set(flatMembers[memberIndex].Name,
								new ValueInstance(flatMembers[memberIndex].Type,
									flatNumericInstance.GetFlat(memberIndex)), isMember: true);
				}
				else
				{
					var typeInstance = evaluatedInstance.Value.TryGetValueTypeInstance();
					if (typeInstance != null)
						TrySetScopeMembersFromTypeMembers(typeInstance);
				}
			}
			for (var paramIndex = 0;
				paramIndex < call.Method.Parameters.Count && paramIndex < call.Arguments.Count;
				paramIndex++)
				Memory.Frame.Set(call.Method.Parameters[paramIndex].Name,
					EvaluateExpression(call.Arguments[paramIndex]));
			var lastExpression = body.Expressions[^1];
			return EvaluateExpression(lastExpression);
		}
		finally
		{
			Memory.Frame = savedFrame;
		}
	}

	private ValueInstance EvaluateFromConstructor(MethodCall call)
	{
		var targetType = call.ReturnType;
		if (TryCreateCurrentAdjustBrightnessDefaultColorImage(targetType, out var colorImage))
			return colorImage;
		var members = targetType.Members;
		if (members.Count == 0 && TryGetBinaryMembers(targetType, out var binaryMembers))
			return new ValueInstance(targetType,
				CreateConstructorValuesFromExpressions(targetType, call.Arguments, binaryMembers));
		var values = new ValueInstance[members.Count];
		var argumentIndex = 0;
		for (var memberIndex = 0; memberIndex < members.Count; memberIndex++)
			if (members[memberIndex].Type.IsTrait)
				values[memberIndex] = CreateTraitInstance(members[memberIndex].Type);
			else if (argumentIndex < call.Arguments.Count)
				values[memberIndex] = EvaluateExpression(call.Arguments[argumentIndex++]);
			else
				values[memberIndex] = CreateDefaultValue(members[memberIndex].Type);
		return new ValueInstance(targetType, values);
	}

	private bool TryCreateCurrentAdjustBrightnessDefaultColorImage(Type targetType,
		out ValueInstance colorImage)
	{
		if (targetType.Name != "ColorImage")
		{
			colorImage = default;
			return false;
		}
		var frame = Memory.Frame;
		if (!frame.TryGet(CallFrame.ResolveSymbolId("width"), out var width) ||
			!frame.TryGet(CallFrame.ResolveSymbolId("height"), out var height))
		{
			colorImage = default;
			return false;
		}
		var members = targetType.Members;
		if (members.Count < 2)
		{
			colorImage = default;
			return false;
		}
		var sizeValue = new ValueInstance(members[0].Type, [width, height]);
		var colorType = executable.basePackage.FindType("Color");
		if (colorType == null)
		{
			colorImage = default;
			return false;
		}
		var defaultColor = new ValueInstance(colorType, [
			new ValueInstance(executable.numberType, 0),
			new ValueInstance(executable.numberType, 0),
			new ValueInstance(executable.numberType, 0),
			new ValueInstance(executable.numberType, 1)
		]);
		var colorCount = (int)(width.Number * height.Number);
		var colors = new ValueInstance[colorCount];
		for (var colorIndex = 0; colorIndex < colorCount; colorIndex++)
			colors[colorIndex] = defaultColor;
		var colorList = new ValueInstance(members[1].Type, colors);
		colorImage = new ValueInstance(targetType, [sizeValue, colorList]);
		return true;
	}

	private static ValueInstance ConvertToText(ValueInstance rawValue)
	{
		if (rawValue.IsText)
			return rawValue;
		if (rawValue.TryGetValueTypeInstance() is { } typeInstance)
			return new ValueInstance(typeInstance.ToAutomaticText());
		return new ValueInstance(rawValue.ToExpressionCodeString());
	}

	/// <summary>
	/// Legacy expression-based scope initialization used by EvaluateMethodCall fallback path.
	/// Will be removed when all expression evaluation is eliminated from the VM.
	/// </summary>
	private void InitializeMethodCallScopeFromExpressions(MethodCall methodCall,
		IReadOnlyList<ValueInstance> evaluatedArguments, ValueInstance? evaluatedInstance)
	{
		for (var parameterIndex = 0; parameterIndex < methodCall.Method.Parameters.Count &&
			parameterIndex < methodCall.Arguments.Count; parameterIndex++)
			Memory.Frame.Set(methodCall.Method.Parameters[parameterIndex].Name,
				evaluatedArguments[parameterIndex]);
		if (methodCall.Instance == null)
			return;
		var instance = evaluatedInstance ?? EvaluateExpression(methodCall.Instance);
		Memory.Frame.Set(Type.ValueLowercase, instance, isMember: true);
		if (instance.IsText)
		{
			Memory.Frame.Set("elements", instance, isMember: true);
			Memory.Frame.Set("characters", instance, isMember: true);
			return;
		}
		var flatNumeric = instance.TryGetFlatNumericArrayInstance();
		if (flatNumeric != null)
		{
			var flatMembers = flatNumeric.ReturnType.Members;
			for (var memberIndex = 0; memberIndex < flatMembers.Count &&
				memberIndex < flatNumeric.FlatWidth; memberIndex++)
				if (!flatMembers[memberIndex].Type.IsTrait)
					Memory.Frame.Set(flatMembers[memberIndex].Name,
						new ValueInstance(flatMembers[memberIndex].Type, flatNumeric.GetFlat(memberIndex)),
						isMember: true);
			return;
		}
		var typeInst = instance.TryGetValueTypeInstance();
		if (typeInst != null && TrySetScopeMembersFromTypeMembers(typeInst))
			return;
		var firstNonTraitMember = instance.GetType().Members.FirstOrDefault(member =>
			!member.Type.IsTrait);
		if (firstNonTraitMember != null)
			Memory.Frame.Set(firstNonTraitMember.Name, instance, isMember: true);
	}

	/// <summary>
	/// Legacy expression-based constructor value creation used by EvaluateFromConstructor fallback.
	/// Will be removed when all expression evaluation is eliminated from the VM.
	/// </summary>
	private ValueInstance[] CreateConstructorValuesFromExpressions(Type targetType,
		IReadOnlyList<Expression> arguments, IReadOnlyList<Bytecode.Serialization.BinaryMember> binaryMembers)
	{
		var values = new ValueInstance[binaryMembers.Count];
		var argumentIndex = 0;
		for (var memberIndex = 0; memberIndex < binaryMembers.Count; memberIndex++)
		{
			var memberType = targetType.FindType(binaryMembers[memberIndex].FullTypeName) ??
				targetType.FindType(GetShortTypeName(binaryMembers[memberIndex].FullTypeName));
			if (memberType is { IsTrait: true })
				values[memberIndex] = CreateTraitInstance(memberType);
			else if (argumentIndex < arguments.Count)
				values[memberIndex] = EvaluateExpression(arguments[argumentIndex++]);
			else if (memberType != null)
				values[memberIndex] = CreateDefaultComplexValue(memberType);
			else
				values[memberIndex] = new ValueInstance(executable.numberType, 0);
		}
		return values;
	}
}
