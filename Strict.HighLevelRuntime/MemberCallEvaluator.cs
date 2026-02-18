using Strict.Expressions;
using Strict.Language;
using System.Collections.Generic;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

internal sealed class MemberCallEvaluator(Executor executor)
{
	public ValueInstance Evaluate(MemberCall member, ExecutionContext ctx) =>
		ctx.This == null && ctx.Type.Members.Contains(member.Member)
			? throw new Executor.UnableToCallMemberWithoutInstance(member, ctx)
			: ctx.This?.Value is Dictionary<string, object?> dict &&
			dict.TryGetValue(member.Member.Name, out var value)
				? new ValueInstance(member.ReturnType, value)
				: member.Member.InitialValue != null && member.IsConstant
					? executor.RunExpression(member.Member.InitialValue, ctx)
					: member.Instance is VariableCall { Variable.Name: Type.OuterLowercase }
						? ctx.Parent!.Get(member.Member.Name)
						: ctx.Get(member.Member.Name);
}
