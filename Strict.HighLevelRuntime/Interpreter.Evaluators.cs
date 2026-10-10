using System.Runtime.CompilerServices;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

[assembly: InternalsVisibleTo("Strict.HighLevelRuntime.Tests")]
[assembly: InternalsVisibleTo("Strict.TestRunner")]

namespace Strict.HighLevelRuntime;

public partial class Interpreter
{
	private void DisposeTrackedValues(ExecutionContext ctx)
	{
		var returnValue = ctx.ExitMethodAndReturnValue;
		foreach (var value in ctx.DisposableValues.ToArray())
			if (returnValue.HasValue && value.Equals(returnValue.Value))
			{
				if (ctx.Parent != null)
				{
					ctx.Parent.TrackDisposable(value);
					ctx.RemoveDisposable(value);
				}
			}
			else
			{
				DisposeTrackedValue(value);
			}
	}

	private void DisposeTrackedValue(ValueInstance value)
	{
		if (!FileValue.TryGetHandle(value, fileType, out var handle))
			return;
		NativeFileRegistry.Close(handle);
	}

	private ValueInstance EvaluateListExpression(List list, ExecutionContext context)
	{
		var constantData = list.TryGetConstantData();
		if (constantData.HasValue)
			return constantData.Value;
		var count = list.Values.Count;
		var values = new ValueInstance[count];
		for (var i = 0; i < count; i++)
			values[i] = RunExpression(list.Values[i], context);
		return new ValueInstance(list.ReturnType, values);
	}

	private ValueInstance EvaluateVariable(string name, ExecutionContext context)
	{
		Statistics.VariableCallCount++;
		return context.Find(name, Statistics) ?? name switch
		{
			Type.ValueLowercase => context.This,
			Type.OuterLowercase => context.Parent!.Get(Type.ValueLowercase, Statistics),
			_ => null
		} ?? throw new ExecutionContext.VariableNotFound(name, context.Type, context.This);
	}

	public ValueInstance EvaluateMemberCall(MemberCall member, ExecutionContext ctx)
	{
		Statistics.MemberCallCount++;
		if (member.Instance is VariableCall { Variable.Name: Type.OuterLowercase })
			return ctx.Parent!.Get(member.Member.Name, Statistics);
		if (member.Member.InitialValue != null && member.IsConstant)
			return RunExpression(member.Member.InitialValue, ctx);
		var instance = member.Instance != null
			? RunExpression(member.Instance, ctx)
			: ctx.This;
		if (instance == null && ctx.Type.Members.Contains(member.Member))
			throw new UnableToCallMemberWithoutInstance(member, ctx); //ncrunch: no coverage
		if (instance is { IsDictionary: true } &&
			member.Member.Name.Equals(Type.ElementsLowercase, StringComparison.OrdinalIgnoreCase))
		{
			var dictionaryItems = instance.Value.GetDictionaryItems();
			var pairs = new ValueInstance[dictionaryItems.Count];
			var pairType = member.Member.Type is { IsList: true, IsGeneric: true }
				? listType.GetFirstImplementation()
				: member.Member.Type;
			var index = 0;
			foreach (var pair in dictionaryItems)
				pairs[index++] = new ValueInstance(pairType, [pair.Key, pair.Value]);
			return new ValueInstance(member.Member.Type, pairs);
		}
		var typeInstance = instance?.TryGetValueTypeInstance();
		if (typeInstance != null && typeInstance.TryGetValue(member.Member.Name, out var value))
			return value;
		if (instance != null && instance.Value.Equals(noneInstance))
			return CreateDefaultMemberValue(member.Member.Type);
		if (instance != null && !member.IsConstant && member.Member.Type.Name != Type.Iterator)
			return new ValueInstance(instance.Value, member.Member.Type);
		return ctx.Get(member.Member.Name, Statistics);
	}

	public class UnableToCallMemberWithoutInstance(MemberCall member, ExecutionContext ctx)
		: Exception(member + ", context " + ctx); //ncrunch: no coverage

	private ValueInstance EvaluateMutableListElementAssignment(ListCall target, Expression value,
		ExecutionContext ctx)
	{
		Statistics.MutableUsageCount++;
		var newValue = RunExpression(value, ctx);
		var index = (int)RunExpression(target.Index, ctx).Number;
		var listInstance = RunExpression(target.List, ctx);
		listInstance.List.Items[index] = newValue;
		return newValue;
	}

	private ValueInstance EvaluateAndAssign(string name, Expression value, ExecutionContext ctx,
		bool isDeclaration)
	{
		if (isDeclaration)
			Statistics.VariableDeclarationCount++;
		if (value.IsMutable)
		{
			if (isDeclaration)
				Statistics.MutableDeclarationCount++;
			Statistics.MutableUsageCount++;
		}
		var result = RunExpression(value, ctx);
		return isDeclaration
			? ctx.Variables[name] = result
			: ctx.Set(name, result);
	}

	private ValueInstance EvaluateReturn(Return r, ExecutionContext ctx)
	{
		Statistics.ReturnCount++;
		var result = RunExpression(r.Value, ctx);
		ctx.ExitMethodAndReturnValue = result;
		return result;
	}

	private ValueInstance EvaluateNot(Not not, ExecutionContext ctx)
	{
		Statistics.UnaryCount++;
		return ToBoolean(!RunExpression(not.Instance!, ctx).Boolean);
	}
}
