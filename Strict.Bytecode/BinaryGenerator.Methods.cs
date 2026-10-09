using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Bytecode;

public sealed partial class BinaryGenerator
{
	private Dictionary<string, Dictionary<string, List<BinaryMethod>>> GenerateRunMethods(
		IReadOnlyList<Method> runMethods, Type entryType)
	{
		var methodsByType =
			new Dictionary<string, Dictionary<string, List<BinaryMethod>>>(StringComparer.Ordinal);
		var methodsToCompile = new Queue<Method>();
		var compiledMethodKeys = new HashSet<string>(StringComparer.Ordinal);
		foreach (var runMethod in runMethods)
		{
			CollectMethodDependencies(runMethod);
			EnqueueConstraintMethods(methodsToCompile, compiledMethodKeys);
			var methodBody = runMethod.GetBodyAndParseIfNeeded();
			var methodExpressions = methodBody is Body body
				? body.Expressions.Where(expr => !runMethod.Tests.Contains(expr)).ToList()
				: [methodBody];
			var childGenerator = new BinaryGenerator(binary.basePackage, methodExpressions,
				runMethod.ReturnType);
			var methodInstructions = childGenerator.GenerateInstructions(childGenerator.Expressions);
			var parameters = CreateBinaryMembers(runMethod.Parameters, entryType);
			AddCompiledMethod(methodsByType, runMethod.Type.FullName, runMethod.Name, parameters,
				GetBinaryTypeName(runMethod.ReturnType, entryType), methodInstructions);
			compiledMethodKeys.Add(BuildMethodKey(runMethod));
			EnqueueDiscoveredMethods(childGenerator.DiscoveredInvokeMethods, methodsToCompile,
				compiledMethodKeys);
			EnqueueConstraintMethods(methodsToCompile, compiledMethodKeys);
		}
		while (methodsToCompile.Count > 0)
		{
			var method = methodsToCompile.Dequeue();
			if (method.Type.IsTrait)
				continue;
			CollectMethodDependencies(method);
			EnqueueConstraintMethods(methodsToCompile, compiledMethodKeys);
			var body = method.GetBodyAndParseIfNeeded();
			var methodExpressions = body is Body methodBody
				? methodBody.Expressions.Where(expr => !method.Tests.Contains(expr)).ToList()
				: [body];
			var childGenerator = new BinaryGenerator(binary.basePackage, methodExpressions,
				method.ReturnType);
			var methodInstructions = childGenerator.GenerateInstructions(childGenerator.Expressions);
			var parameters = CreateBinaryMembers(method.Parameters, entryType);
			AddCompiledMethod(methodsByType, method.Type.FullName, method.Name, parameters,
				GetBinaryTypeName(method.ReturnType, entryType), methodInstructions);
			EnqueueDiscoveredMethods(childGenerator.DiscoveredInvokeMethods, methodsToCompile,
				compiledMethodKeys);
			EnqueueConstraintMethods(methodsToCompile, compiledMethodKeys);
		}
		return methodsByType;
	}

	private static List<BinaryMember> CreateBinaryMembers(IReadOnlyList<Parameter> parameters,
		Type entryType) =>
		parameters.Select(parameter =>
				new BinaryMember(parameter.Name, GetBinaryTypeName(parameter.Type, entryType), null)).
			ToList();

	private void AddGeneratedTypes(
		Dictionary<string, Dictionary<string, List<BinaryMethod>>> methodsByType, Type entryType)
	{
		var orderedTypes = dependencyTypes.Values.OrderBy(type => type == entryType
			? string.Empty
			: GetBinaryTypeName(type, entryType), StringComparer.Ordinal);
		foreach (var type in orderedTypes)
		{
			AddTraitMethodSignatures(methodsByType, type, entryType);
			var members = type.Members.Where(member => !member.IsConstant || member.InitialValue != null).
				Select(member => new BinaryMember(member.Name, GetBinaryTypeName(member.Type, entryType),
					CreateInitialValueInstruction(member.InitialValue))
				{
					IsConstant = member.IsConstant
				}).ToList();
			binary.AddType(GetBinaryTypeName(type, entryType), members,
				methodsByType.TryGetValue(type.FullName, out var methodGroups)
					? methodGroups
					: new Dictionary<string, List<BinaryMethod>>(StringComparer.Ordinal), type == entryType);
		}
	}

	private static void AddTraitMethodSignatures(
		Dictionary<string, Dictionary<string, List<BinaryMethod>>> methodsByType, Type type,
		Type entryType)
	{
		if (!type.IsTrait)
			return;
		foreach (var method in type.Methods)
		{
			if (method.lines.Count != 1)
				continue;
			if (methodsByType.TryGetValue(type.FullName, out var groups) && groups.ContainsKey(method.Name))
				continue;
			AddCompiledMethod(methodsByType, type.FullName, method.Name,
				CreateBinaryMembers(method.Parameters, entryType),
				GetBinaryTypeName(method.ReturnType, entryType), []);
		}
	}

	private static Instruction? CreateInitialValueInstruction(Expression? initialValue) =>
		initialValue switch
		{
			Value { ConstantData: { } constantData } => new SetInstruction(constantData, Register.R0),
			MethodCall { Method.Name: Method.From, ReturnType.IsNumber: true,
				Arguments: [Value { ReturnType.IsNumber: true } number] } constructor =>
				new SetInstruction(new ValueInstance(constructor.ReturnType, number.Data.Number), Register.R0),
			_ => null
		};

	private void CollectMethodDependencies(Method method)
	{
		CollectTypeDependency(method.Type, true);
		CollectTypeDependency(method.ReturnType, false);
		foreach (var parameter in method.Parameters)
			CollectTypeDependency(parameter.Type, false);
		if (method.Type.IsTrait)
			return;
		var body = method.GetBodyAndParseIfNeeded();
		if (body is Body methodBody)
			foreach (var expression in methodBody.Expressions)
				CollectExpressionDependencies(expression);
		else
			CollectExpressionDependencies(body);
	}

	private void CollectExpressionDependencies(Expression expression)
	{
		CollectTypeDependency(expression.ReturnType, false);
		switch (expression)
		{
		case Body body:
			foreach (var child in body.Expressions)
				CollectExpressionDependencies(child);
			break;
		case Binary binaryExpr:
			CollectTypeDependency(binaryExpr.Method.Type, true);
			CollectTypeDependency(binaryExpr.Method.ReturnType, false);
			foreach (var parameter in binaryExpr.Method.Parameters)
				CollectTypeDependency(parameter.Type, false);
			CollectExpressionDependencies(binaryExpr.Instance!);
			// ReSharper disable TailRecursiveCall
			CollectExpressionDependencies(binaryExpr.Arguments[0]);
			break;
		case Declaration declaration:
			CollectExpressionDependencies(declaration.Value);
			break;
		case MutableReassignment reassignment:
			CollectExpressionDependencies(reassignment.Value);
			break;
		case For forExpression:
			CollectExpressionDependencies(GetLoopIteratorExpression(forExpression.Iterator));
			CollectExpressionDependencies(forExpression.Body);
			break;
		case If ifExpression:
			CollectExpressionDependencies(ifExpression.Condition);
			CollectExpressionDependencies(ifExpression.Then);
			if (ifExpression.OptionalElse != null)
				CollectExpressionDependencies(ifExpression.OptionalElse);
			break;
		case SelectorIf selectorIf:
			CollectExpressionDependencies(selectorIf.Selector);
			foreach (var @case in selectorIf.Cases)
			{
				CollectExpressionDependencies(@case.Pattern);
				CollectExpressionDependencies(@case.Then);
			}
			if (selectorIf.OptionalElse != null)
				CollectExpressionDependencies(selectorIf.OptionalElse);
			break;
		case ListCall listCall:
			CollectExpressionDependencies(listCall.List);
			CollectExpressionDependencies(listCall.Index);
			break;
		case MemberCall memberCall:
			CollectTypeDependency(memberCall.Member.Type, false);
			if (memberCall.Instance != null)
				CollectExpressionDependencies(memberCall.Instance);
			break;
		case MethodCall methodCall:
			CollectTypeDependency(methodCall.Method.Type, true);
			CollectTypeDependency(methodCall.Method.ReturnType, false);
			foreach (var parameter in methodCall.Method.Parameters)
				CollectTypeDependency(parameter.Type, false);
			if (methodCall.Instance != null)
				CollectExpressionDependencies(methodCall.Instance);
			foreach (var argument in methodCall.Arguments)
				CollectExpressionDependencies(argument);
			break;
		}
	}

	private void CollectTypeDependency(Type type, bool includeType)
	{
		if (type.IsNone || type.IsAny)
			return;
		if (type is GenericTypeImplementation genericImplementation)
		{
			foreach (var implementationType in genericImplementation.ImplementationTypes)
				CollectTypeDependency(implementationType, false);
			if (!includeType)
				return;
		}
		else if (type is GenericType genericType)
		{
			foreach (var implementation in genericType.GenericImplementations)
				CollectTypeDependency(implementation.Type, false);
			if (!includeType)
				return;
		}
		if (!dependencyTypes.TryAdd(type.FullName, type))
			return;
		foreach (var member in type.Members)
			CollectTypeDependency(member.Type, false);
	}

	private static string GetBinaryTypeName(Type type, Type entryType)
	{
		if (IsStrictBaseType(type, entryType))
			return nameof(Strict) + Context.ParentSeparator + type.Name;
		var entryPackagePrefix = entryType.Package.FullName + Context.ParentSeparator;
		return type.FullName.StartsWith(entryPackagePrefix, StringComparison.Ordinal)
			? type.FullName[entryPackagePrefix.Length..]
			: type.FullName;
	}

	private static bool IsStrictBaseType(Type type, Type entryType) =>
		type.FullName != entryType.FullName && (type.Package.Name == nameof(Strict) ||
			(entryType.Package.Name == "TestPackage" && type.Package.Name == "TestPackage"));

	private Dictionary<string, Dictionary<string, List<BinaryMethod>>> CompileMethodsFromExpressions(
		string thisEntryTypeFullName, IReadOnlyList<Expression> entryExpressions, Type runReturnType)
	{
		var methodsByType =
			new Dictionary<string, Dictionary<string, List<BinaryMethod>>>(StringComparer.Ordinal);
		var methodsToCompile = new Queue<Method>();
		var compiledMethodKeys = new HashSet<string>(StringComparer.Ordinal);
		var runInstructions = GenerateInstructions(entryExpressions);
		AddCompiledMethod(methodsByType, thisEntryTypeFullName, Method.Run, [], runReturnType.Name,
			runInstructions);
		EnqueueDiscoveredMethods(discoveredInvokeMethods, methodsToCompile, compiledMethodKeys);
		while (methodsToCompile.Count > 0)
		{
			var method = methodsToCompile.Dequeue();
			CollectMethodDependencies(method);
			var body = method.GetBodyAndParseIfNeeded();
			var methodExpressions = body is Body methodBody
				? methodBody.Expressions.Where(expr => !method.Tests.Contains(expr)).ToList()
				: [body];
			var childGenerator = new BinaryGenerator(binary.basePackage, methodExpressions,
				method.ReturnType);
			var methodInstructions = childGenerator.GenerateInstructions(childGenerator.Expressions);
			var parameters = method.Parameters.Select(parameter =>
				new BinaryMember(parameter.Name, parameter.Type.FullName, null)).ToList();
			AddCompiledMethod(methodsByType, method.Type.FullName, method.Name, parameters,
				method.ReturnType.Name, methodInstructions);
			EnqueueDiscoveredMethods(childGenerator.DiscoveredInvokeMethods, methodsToCompile,
				compiledMethodKeys);
		}
		return methodsByType;
	}

	private void EnqueueConstraintMethods(Queue<Method> methodsToCompile,
		HashSet<string> compiledMethodKeys)
	{
		foreach (var type in dependencyTypes.Values)
		foreach (var member in type.Members)
			EnqueueConstraintMethods(type, member, methodsToCompile, compiledMethodKeys);
	}

	private static void EnqueueConstraintMethods(Type type, Member member,
		Queue<Method> methodsToCompile, HashSet<string> compiledMethodKeys)
	{
		if (member.Constraints == null)
			return;
		foreach (var constraint in member.Constraints)
			if (constraint is Binary { Method.Name: BinaryOperator.Is, Instance: { } instance } binary &&
				instance.ToString() == "Length")
			{
				var rhsText = binary.Arguments[0].ToString();
				var separatorIndex = rhsText.IndexOf('.');
				if (separatorIndex <= 0)
					continue;
				var referencedMember = type.Members.FirstOrDefault(typeMember =>
					typeMember.Name.Equals(rhsText[..separatorIndex], StringComparison.OrdinalIgnoreCase));
				var method = referencedMember?.Type.FindMethod(rhsText[(separatorIndex + 1)..], []);
				if (method != null && compiledMethodKeys.Add(BuildMethodKey(method)))
					methodsToCompile.Enqueue(method);
			}
	}

	private static string BuildMethodKey(Method method) =>
		method.Type.FullName + ":" + BinaryExecutable.BuildMethodHeader(method.Name,
			method.Parameters.Select(parameter =>
				new BinaryMember(parameter.Name, parameter.Type.FullName, null)).ToArray(),
			method.ReturnType);

	private static void EnqueueDiscoveredMethods(IReadOnlyList<Method> methods,
		Queue<Method> methodsToCompile, HashSet<string> compiledMethodKeys)
	{
		foreach (var method in methods)
		{
			var methodKey = BuildMethodKey(method);
			if (compiledMethodKeys.Add(methodKey))
				methodsToCompile.Enqueue(method);
		}
	}

	private static void AddCompiledMethod(
		Dictionary<string, Dictionary<string, List<BinaryMethod>>> methodsByType, string typeFullName,
		string methodName, List<BinaryMember> parameters, string returnTypeName,
		List<Instruction> instructionsToAdd)
	{
		if (!methodsByType.TryGetValue(typeFullName, out var methodGroups))
		{
			methodGroups = new Dictionary<string, List<BinaryMethod>>(StringComparer.Ordinal);
			methodsByType[typeFullName] = methodGroups;
		}
		if (!methodGroups.TryGetValue(methodName, out var overloads))
		{
			overloads = [];
			methodGroups[methodName] = overloads;
		}
		overloads.Add(new BinaryMethod(methodName, parameters, returnTypeName, instructionsToAdd));
	}
}
