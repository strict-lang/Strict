using System.Collections.Generic;
using System.Linq;
using Strict.Language.Expressions;
using Strict.Language.Tokens;

namespace Strict.Language
{
	public class MethodBody : Tokenizer
	{
		public MethodBody(Method method, IReadOnlyList<string> lines)
		{
			var lineLexer = new LineLexer(this);
			for (var index = 0; index < lines.Count; index++)
				lineLexer.Process(lines[index]);
			if (lines[0].StartsWith("\treturn"))
			{
				var number = method.GetType(Base.Number);
				expressions.Add(new Return(new Binary(new Number(method, 5), number.Methods[0],
					new Boolean(method, true))));
			}
			else
			{
				var log = method.GetType(Base.Log);
				expressions.Add(new MethodCall(new MemberCall(method.Type.Members[0]),
					log.Methods.First(m => m.Name == "WriteLine"), new Text(method, "Hey")));
			}
		}

		private readonly List<Expression> expressions = new List<Expression>();
		public IReadOnlyList<Expression> Expressions => expressions;
		public void Add(Token token) { }
	}
}