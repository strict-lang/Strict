namespace Strict.Language
{
	public class Method
	{
		public Method(string name, Parameter[] parameters, Type returnType)
		{
			Name = name;
			Parameters = parameters;
			ReturnType = returnType;
		}

		public string Name { get; }
		public Parameter[] Parameters { get; }
		public Type ReturnType { get; }
	}
}