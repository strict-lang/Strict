using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

public sealed partial class MethodCallEvaluator
{
	private static string BuildCoreTypeFallbackMessage(MethodCall call, ExecutionContext ctx,
		ValueInstance left, ValueInstance right)
	{
		var message = "Cannot " + call.Method.Name + " left=" + FormatOperand(left) + " right=" +
			FormatOperand(right) + ", method=" + ctx.Method + ", call=" + call;
		var caller = GetCallerDisplay(ctx.Parent);
		return caller.Length == 0 || caller == ctx.Method.ToString() || caller == ctx.Method.Name
			? message
			: message + ", caller=" + caller;
	}

	private static string FormatOperand(ValueInstance value)
	{
		if (value.IsText)
			return Quote(value.Text);
		if (value.GetType().IsNumber)
			return value.GetCachedNumberString();
		if (value.GetType().IsBoolean)
			return value.Boolean
				? "true"
				: "false";
		if (value.TryGetValueTypeInstance() is { } typeInstance &&
			typeInstance.TryGetValue(Type.ValueLowercase, out var inner))
			return FormatOperand(inner) + " (" + typeInstance.ReturnType.Name + ")";
		return value.ToExpressionCodeString();
	}

	private static string Quote(string text) => "\"" + text + "\"";

	private static string GetCallerDisplay(ExecutionContext? ctx)
	{
		if (ctx == null)
			return "";
		var lineNumber = ctx.CurrentExpressionLineNumber;
		if (lineNumber >= 0 && lineNumber < ctx.Type.Lines.Length)
		{
			var line = ctx.Type.Lines[lineNumber].Trim();
			if (line.Length > 0)
				return line;
		}
		return ctx.Method.ToString();
	}

	private ValueInstance Error(string name, ExecutionContext ctx, Expression? source = null)
	{
		var errorType = ctx.Method.GetType(Type.Error);
		var errorValues = new ValueInstance[errorType.Members.Count];
		for (var i = 0; i < errorType.Members.Count; i++)
			errorValues[i] = errorType.Members[i].Type.Name switch
			{
				nameof(Type.Name) or Type.Text => new ValueInstance(name),
				_ when errorType.Members[i].Type.IsList => CreateStacktrace(ctx, source),
				_ => throw new InterpreterExecutionFailed(ctx.Method, //ncrunch: no coverage
					"Error member not supported: " + errorType.Members[i])
			};
		return new ValueInstance(errorType, errorValues);
	}

	private ValueInstance CreateStacktrace(ExecutionContext ctx, Expression? source)
	{
		var stacktraceType = ctx.Method.GetType(Type.Stacktrace);
		var stackValues = new ValueInstance[stacktraceType.Members.Count];
		for (var i = 0; i < stacktraceType.Members.Count; i++)
			stackValues[i] = stacktraceType.Members[i].Type.Name switch
			{
				nameof(Method) => new ValueInstance(ctx.Method.GetType(nameof(Method)),
					CreateMethodValue(ctx.Method)),
				Type.Text or nameof(Type.Name) => new ValueInstance(ctx.Method.Type.FilePath),
				Type.Number => new ValueInstance(interpreter.numberType,
					source?.LineNumber ?? ctx.Method.TypeLineNumber),
				_ => throw new InterpreterExecutionFailed(ctx.Method, //ncrunch: no coverage
					"Stacktrace member not supported: " + stacktraceType.Members[i])
			};
		return new ValueInstance(interpreter.listType.GetGenericImplementation(stacktraceType),
			[new ValueInstance(stacktraceType, stackValues)]);
	}

	private static ValueInstance[] CreateMethodValue(Method method)
	{
		var methodType = method.GetType(nameof(Method));
		var values = new ValueInstance[methodType.Members.Count];
		for (var i = 0; i < methodType.Members.Count; i++)
			values[i] = methodType.Members[i].Type.Name switch
			{
				nameof(Type.Name) or Type.Text => new ValueInstance(method.Name),
				nameof(Type) => new ValueInstance(method.GetType(nameof(Type)),
					CreateTypeValue(method.Type)),
				_ => throw new InterpreterExecutionFailed(method, //ncrunch: no coverage
					"Method member not supported: " + methodType.Members[i])
			};
		return values;
	}

	internal static ValueInstance[] CreateTypeValue(Type type)
	{
		var typeType = type.GetType(nameof(Type));
		var values = new ValueInstance[typeType.Members.Count];
		for (var i = 0; i < typeType.Members.Count; i++)
			values[i] = typeType.Members[i].Type.Name switch
			{
				nameof(Type.Name) => new ValueInstance(type.Name),
				Type.Text => new ValueInstance(type.Package.FullName),
				_ => throw new InterpreterExecutionFailed(type.Methods[0], //ncrunch: no coverage
					"Type member not supported: " + typeType.Members[i])
			};
		return values;
	}
}
