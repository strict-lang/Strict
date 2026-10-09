using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Bytecode;

/// <summary>
/// Converts an expression into a <see cref="BinaryExecutable"/>, mostly from calling the Run
/// method of a .strict type, but can be any expression. Will get all used types with their
/// members and used methods recursively, execution and serialization can be done independently.
/// </summary>
public sealed partial class BinaryGenerator
{
	public BinaryGenerator(MethodCall methodCall)
	{
		entryMethodCall = methodCall;
		entryTypeFullName = methodCall.Method.Type.FullName;
		if (methodCall.Instance is MethodCall instanceCall)
			AddInstanceMemberVariables(instanceCall);
		AddMethodParameterVariables(methodCall);
		//TODO: this randomly crashes VirtualMachineTests.Enum stuff .. bad anyway
		var methodBody = methodCall.Method.GetBodyAndParseIfNeeded();
		// Never emit test assertions into production bytecode (was previously "working"
		// only because Binary `is` expressions were silently dropped).
		Expressions = methodBody is Body body
			? body.Expressions.Where(expr => !methodCall.Method.Tests.Contains(expr)).ToList()
			: [methodBody];
		ReturnType = methodCall.Method.ReturnType;
		binary = new BinaryExecutable(GetBasePackage(methodCall));
	}

	private BinaryGenerator(Package basePackage, IReadOnlyList<Expression> expressions,
		Type returnType)
	{
		binary = new BinaryExecutable(basePackage);
		Expressions = expressions;
		ReturnType = returnType;
		entryTypeFullName = "";
	}

	private readonly BinaryExecutable binary;

	private readonly MethodCall? entryMethodCall;

	private readonly string entryTypeFullName;

	private readonly List<Instruction> instructions = []; //TODO: why not keep this in BinaryMethod

	private readonly Dictionary<string, Type> dependencyTypes = new(StringComparer.Ordinal);

	private readonly Registry registry = new();

	private readonly Stack<int> idStack = new();

	private readonly List<Method> discoveredInvokeMethods = [];

	internal IReadOnlyList<Method> DiscoveredInvokeMethods => discoveredInvokeMethods;

	private IReadOnlyList<Expression> Expressions { get; } //TODO: stupid, remove

	private Type ReturnType { get; } //TODO: stupid, remove

	private int conditionalId; //TODO: a bit strange

	private int forResultId;

	private int listResultId;

	private void AddInstruction(Instruction instruction, int sourceLine)
	{
		instruction.SourceLine = sourceLine;
		instructions.Add(instruction);
	}

	public BinaryExecutable Generate() =>
		entryMethodCall is { Method.Name: Method.Run, Arguments.Count: 0 }
			? Generate(entryMethodCall.Method,
				entryMethodCall.Method.Type.Methods.Where(method => method.Name == Method.Run).ToArray())
			: Generate(entryTypeFullName, Expressions, ReturnType);

	//TODO: this is convoluted and not good
	public static BinaryExecutable GenerateFromRunMethods(Method preferredEntryMethod,
		IReadOnlyList<Method> runMethods)
	{
		var generator = new BinaryGenerator(GetBasePackage(preferredEntryMethod), [],
			preferredEntryMethod.ReturnType);
		return generator.Generate(preferredEntryMethod, runMethods);
	}

	public static List<Instruction> GenerateInlineInstructions(Package basePackage,
		Expression expression) =>
		new BinaryGenerator(basePackage, [expression], expression.ReturnType).GenerateInstructionList();

	private BinaryExecutable Generate(Method preferredEntryMethod, IReadOnlyList<Method> runMethods)
	{
		var methodsByType = GenerateRunMethods(runMethods, preferredEntryMethod.Type);
		AddGeneratedTypes(methodsByType, preferredEntryMethod.Type);
		binary.SetEntryPoint(GetBinaryTypeName(preferredEntryMethod.Type, preferredEntryMethod.Type),
			preferredEntryMethod.Name, preferredEntryMethod.Parameters.Count,
			GetBinaryTypeName(preferredEntryMethod.ReturnType, preferredEntryMethod.Type));
		return binary;
	}

	private BinaryExecutable Generate(string typeFullName, IReadOnlyList<Expression> entryExpressions,
		Type runReturnType)
	{
		var methodsByType =
			CompileMethodsFromExpressions(typeFullName, entryExpressions, runReturnType);
		var entryType = FindEntryType(typeFullName);
		if (entryType == null)
		{
			foreach (var (compiledTypeFullName, methodGroups) in methodsByType)
				binary.AddType(compiledTypeFullName, [], methodGroups);
		}
		else
		{
			CollectTypeDependency(entryType, true);
			CollectTypeDependency(runReturnType, false);
			foreach (var expression in entryExpressions)
				CollectExpressionDependencies(expression);
			AddGeneratedTypes(methodsByType, entryType);
		}
		return binary;
	}

	private Type? FindEntryType(string typeFullName) =>
		string.IsNullOrEmpty(typeFullName)
			? null
			: binary.basePackage.FindFullType(typeFullName) ?? binary.basePackage.FindType(typeFullName);

	private static Package GetBasePackage(Expression expression)
	{
		Context context = expression.ReturnType;
		while (context is not Package)
			context = context.Parent;
		return (Package)context;
	}

	private static Package GetBasePackage(Method method)
	{
		Context context = method.Type;
		while (context is not Package)
			context = context.Parent;
		return (Package)context;
	}

	private static ValueInstance GetValueInstanceFromExpression(Expression expression) =>
		expression switch
		{
			List list => list.TryGetConstantData() ?? throw new ListIsNotConstant(list),
			Value val => val.Data,
			MemberCall memberCall when memberCall.Member.InitialValue != null => memberCall.Member.
				InitialValue is Value enumValue
				? enumValue.Data
				: new ValueInstance(memberCall.Member.InitialValue.ToString()),
			_ => new ValueInstance(expression.ToString()) //ncrunch: no coverage
		};

	private void AddInstanceMemberVariables(MethodCall instance)
	{
		for (var parameterIndex = 0; parameterIndex < instance.Method.Parameters.Count;
			parameterIndex++)
		{
			var parameter = instance.Method.Parameters[parameterIndex];
			var member = instance.ReturnType.Members.FirstOrDefault(typeMember =>
				typeMember.Name.Equals(parameter.Name, StringComparison.OrdinalIgnoreCase));
			if (member == null)
				continue;
			var argumentExpression = parameterIndex < instance.Arguments.Count
				? instance.Arguments[parameterIndex]
				: parameter.DefaultValue ?? member.InitialValue;
			if (argumentExpression == null)
				continue;
			if (parameter.Type.IsList && instance.Arguments is not [List])
			{
				var listItems = instance.Arguments.Select(GetValueInstanceFromExpression).ToArray();
				instructions.Add(new StoreVariableInstruction(new ValueInstance(parameter.Type, listItems),
					member.Name, true));
			}
			else
			{
				instructions.Add(new StoreVariableInstruction(
					GetValueInstanceFromExpression(argumentExpression), member.Name, true));
			}
		}
	}

	private void AddMethodParameterVariables(MethodCall methodCall)
	{
		for (var index = 0; index < methodCall.Method.Parameters.Count &&
			index < methodCall.Arguments.Count; index++)
			StoreEntryVariable(methodCall.Method.Parameters[index].Name, methodCall.Arguments[index]);
	}

	private void StoreEntryVariable(string identifier, Expression expression)
	{
		if (TryGetConstantValueInstance(expression, out var value))
		{
			instructions.Add(new StoreVariableInstruction(value, identifier));
		}
		else
		{
			GenerateInstructionFromExpression(expression);
			instructions.Add(new StoreFromRegisterInstruction(registry.PreviousRegister, identifier));
		}
	}

	private static bool TryGetConstantValueInstance(Expression expression, out ValueInstance value)
	{
		switch (expression)
		{
		case List list:
			var listValue = list.TryGetConstantData();
			if (listValue != null)
			{
				value = listValue.Value;
				return true;
			}
			break;
		case Value constantValue:
			value = constantValue.Data;
			return true;
		case MemberCall memberCall when memberCall.Member.InitialValue != null:
			value = memberCall.Member.InitialValue is Value enumValue
				? enumValue.Data
				: new ValueInstance(memberCall.Member.InitialValue.ToString());
			return true;
		}
		value = default;
		return false;
	}

	private List<Instruction> GenerateInstructions(IReadOnlyList<Expression> expressions)
	{
		for (var i = 0; i < expressions.Count; i++)
			if (ReferenceEquals(expressions[i], Expressions[^1]) &&
				expressions[i] is If { OptionalElse: not null, Then: not Body } inlineConditional)
				GenerateReturningInlineConditional(inlineConditional);
			else if ((ReferenceEquals(expressions[i], Expressions[^1]) || expressions[i] is Return) &&
				expressions[i] is not If && expressions[i] is not SelectorIf)
				GenerateReturnInstruction(expressions[i]);
			else
				GenerateInstructionFromExpression(expressions[i]);
		return instructions;
	}

	private void GenerateReturnInstruction(Expression expression)
	{
		if (expression is Return returnExpression)
			expression = returnExpression.Value;
		if (TryGenerateNumberForLoopReturn(expression))
			return;
		if (TryGenerateListForLoopReturn(expression))
			return;
		if (expression is For anyLoop && ReturnType.IsBoolean)
		{
			GenerateLoopInstructions(anyLoop, nameof(LoopAggregation.Any), LoopAggregation.Any);
			instructions.Add(new LoadConstantInstruction(registry.AllocateRegister(),
				new ValueInstance(ReturnType, false)));
			instructions.Add(new ReturnInstruction(registry.PreviousRegister));
			return;
		}
		GenerateInstructionFromExpression(expression);
		instructions.Add(new ReturnInstruction(registry.PreviousRegister));
	}

	private enum LoopAggregation
	{
		None,
		Number,
		List,
		Any
	}

	//TODO: try optimize into a expression switch
	private void GenerateInstructionFromExpression(Expression expression)
	{
		var countBefore = instructions.Count;
		switch (expression)
		{
		case Body body:
			GenerateInstructions(body.Expressions);
			return;
		case Binary binaryExpression:
			if (TryGenerateNumberComparisonValue(binaryExpression))
				break;
			if (!CanGenerateDirectBinaryInstruction(binaryExpression.Method.Name))
			{
				GenerateMethodCallInstruction(binaryExpression);
				break;
			}
			GenerateCodeForBinary(binaryExpression);
			break;
		case If ifExpression:
			GenerateIfInstructions(ifExpression);
			return;
		case SelectorIf selectorIf:
			GenerateSelectorIfInstructions(selectorIf);
			return;
		case Declaration { IsMutable: true } mutableDeclaration:
			GenerateForAssignmentOrDeclaration(mutableDeclaration.Value, mutableDeclaration.Name);
			return;
		case Declaration declaration:
			GenerateForAssignmentOrDeclaration(declaration.Value, declaration.Name);
			return;
		case For forExpression:
			GenerateLoopInstructions(forExpression);
			return;
		case MutableReassignment reassignment:
			GenerateForAssignmentOrDeclaration(reassignment.Value, reassignment.Name);
			return;
		case MemberCall memberCall:
			GenerateMemberCallInstruction(memberCall);
			break;
		case VariableCall:
		case ParameterCall:
		case Instance:
			instructions.Add(new LoadVariableToRegister(registry.AllocateRegister(),
				expression.ToString()));
			break;
		case List list:
			GenerateListExpression(list);
			break;
		case Value value:
			instructions.Add(new LoadConstantInstruction(registry.AllocateRegister(),
				GetValueInstanceFromExpression(value)));
			break;
		case MethodCall methodCall:
			GenerateMethodCallInstruction(methodCall);
			break;
		case ListCall listCall:
			// Always emit index load then ListCall. Consecutive indexes (kinds(i), numbers(i))
			// each need their own index materialization — never reuse a prior list element register.
			GenerateInstructionFromExpression(listCall.Index);
			var indexRegister = registry.PreviousRegister;
			instructions.Add(new ListCallInstruction(registry.AllocateRegister(), indexRegister,
				listCall.List.ToString()));
			break;
		default:
			throw new ExpressionNotSupported(expression); //ncrunch: no coverage
		}
		var sourceLine = expression.LineNumber;
		for (var instructionIndex = countBefore; instructionIndex < instructions.Count;
			instructionIndex++)
			if (instructions[instructionIndex].SourceLine == 0)
				instructions[instructionIndex].SourceLine = sourceLine;
	}

	private void GenerateMemberCallInstruction(MemberCall memberCall)
	{
		if (memberCall.IsConstant && memberCall.Member.InitialValue != null)
		{
			instructions.Add(new LoadConstantInstruction(registry.AllocateRegister(),
				GetValueInstanceFromExpression(memberCall)));
			return;
		}
		if (memberCall.Instance == null)
		{
			instructions.Add(new LoadVariableToRegister(registry.AllocateRegister(),
				memberCall.ToString()));
			return;
		}
		if (memberCall.Member.InitialValue != null && memberCall.Member.DefinedIn.IsEnum)
		{
			TryGenerateForEnum(memberCall.Member.DefinedIn, memberCall.Member.InitialValue);
			return;
		}
		GenerateInstructionFromExpression(memberCall.Instance);
		var objectRegister = registry.PreviousRegister;
		// Struct / value-type fields (Language Type, Member, Path, …). Not Text/List primitives.
		if (memberCall.Member.Name != Type.IndexLowercase &&
			IsStructFieldAccess(memberCall.Instance.ReturnType))
		{
			instructions.Add(new FieldLoadInstruction(registry.AllocateRegister(), objectRegister,
				memberCall.Member.Name));
			return;
		}
		// Length/Count on Text/List → Invoke so native VM handlers run
		if (memberCall.Member.Name is "Length" or "Count")
		{
			var lengthInfo = new InvokeMethodInfo(
				GetBinaryTypeName(memberCall.Instance.ReturnType, memberCall.Instance.ReturnType),
				memberCall.Member.Name, [],
				GetBinaryTypeName(memberCall.ReturnType, memberCall.Instance.ReturnType), [],
				objectRegister);
			instructions.Add(new Invoke(registry.AllocateRegister(), lengthInfo));
			return;
		}
		// Fallback: keep identifier for frame-relative member loads (e.g. text.characters)
		instructions.Add(new LoadVariableToRegister(registry.AllocateRegister(),
			memberCall.ToString()));
	}

	private static bool IsStructFieldAccess(Type type) =>
		!type.IsText && !type.IsNumber && !type.IsBoolean && !type.IsCharacter && !type.IsList &&
		!type.IsNone && !type.IsAny && !type.IsEnum && type is not GenericTypeImplementation;

	private void GenerateMethodCallInstruction(MethodCall methodCall)
	{
		if (TryGenerateInstructionForCollectionManipulation(methodCall))
			return;
		if (TryGeneratePrintInstruction(methodCall))
			return;
		if (methodCall.Method.Name != Method.From)
			discoveredInvokeMethods.Add(methodCall.Method);
		Register? instanceRegister = null;
		if (methodCall.Instance != null)
		{
			GenerateInstructionFromExpression(methodCall.Instance);
			instanceRegister = registry.PreviousRegister;
		}
		var argumentRegisters = new Register[methodCall.Arguments.Count];
		for (var argumentIndex = 0; argumentIndex < methodCall.Arguments.Count; argumentIndex++)
		{
			GenerateInstructionFromExpression(methodCall.Arguments[argumentIndex]);
			argumentRegisters[argumentIndex] = registry.PreviousRegister;
		}
		var parameterNames = new string[methodCall.Method.Parameters.Count];
		for (var paramIndex = 0; paramIndex < methodCall.Method.Parameters.Count; paramIndex++)
			parameterNames[paramIndex] = methodCall.Method.Parameters[paramIndex].Name;
		var methodInfo = new InvokeMethodInfo(methodCall.Method.Type.FullName, methodCall.Method.Name,
			parameterNames, GetBinaryTypeName(methodCall.ReturnType, methodCall.Method.Type),
			argumentRegisters, instanceRegister);
		instructions.Add(new Invoke(registry.AllocateRegister(), methodInfo));
	}

	private void TryGenerateForEnum(Type type, Expression value)
	{
		if (type.IsEnum)
		{
			var data = value is Value val
				? val.Data
				: new ValueInstance(value.ToString());
			instructions.Add(new LoadConstantInstruction(registry.AllocateRegister(), data));
		}
	}

	private bool TryGeneratePrintInstruction(MethodCall methodCall)
	{
		if (methodCall.Instance is not MemberCall memberCall)
			return false;
		if (memberCall.Member.Type.Name is not (Type.Logger or Type.TextWriter or Type.System))
			return false;
		if (methodCall.Arguments.Count == 0)
		{
			instructions.Add(new PrintInstruction(""));
			return true;
		}
		var argument = methodCall.Arguments[0];
		if (argument is Value textValue && textValue.Data.IsText)
		{
			instructions.Add(new PrintInstruction(textValue.Data.Text));
			return true;
		}
		if (argument is Binary { Method.Name: BinaryOperator.Plus, Instance: { } left } binaryExpression &&
			UnwrapToConversion(left) is Value { Data.IsText: true })
		{
			var prefix = ExtractTextPrefix(binaryExpression.Instance);
			var valueExpression = UnwrapToConversion(binaryExpression.Arguments[0]);
			GenerateInstructionFromExpression(valueExpression);
			instructions.Add(new PrintInstruction(prefix, registry.PreviousRegister,
				valueExpression.ReturnType.IsText));
			return true;
		}
		if (argument is MethodCall argumentMethodCall)
		{
			GenerateInstructionFromExpression(argumentMethodCall);
			instructions.Add(new PrintInstruction("", registry.PreviousRegister,
				argumentMethodCall.ReturnType.IsText));
			return true;
		}
		if (argument is ParameterCall or VariableCall or MemberCall or ListCall)
		{
			GenerateInstructionFromExpression(argument);
			instructions.Add(new PrintInstruction("", registry.PreviousRegister,
				argument.ReturnType.IsText));
			return true;
		}
		instructions.Add(new PrintInstruction(argument.ToString()));
		return true;
	}

	private sealed class InstanceNameNotFound : Exception;

	public sealed class ListTypeNotFound(Type elementType)
		: Exception("List type not found for loop aggregation of " + elementType);

	public sealed class ExpressionNotSupported(Expression expression)
		: Exception("Bytecode generation does not support " + expression.GetType().Name + ": " +
			expression);

	public sealed class OperatorNotSupported(string binaryOperator)
		: Exception("Bytecode generation does not support operator " + binaryOperator);

	public sealed class ListIsNotConstant(List list)
		: Exception("Only constant lists can be stored as constant data: " + list);
}
