using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Strict.Language.Expressions
{
	public class Instance : Expression
	{
		public Instance(Type type) : base(type) { }
		public static Expression Parse(Method method) => new Instance((Type)method.Parent);
	}
}
