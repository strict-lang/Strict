namespace Strict.Language
{
    /// <summary>
    /// Simple helper to give us all the names of common base types often used like Number and Boolean
    /// </summary>
    public class Base
    {
        /// <summary>
        /// Has no implementation and is used for void, empty or none, which is not valid to assign.
        /// </summary>
        public const string None = nameof(None);
        /// <summary>
        /// Defines all the methods available in any type (everything automatically implements **Any**).
        /// These methods don't have to be implemented by any class, they are automatically implemented.
        /// </summary>
        public const string Any = nameof(Any);
        /// <summary>
        /// Most basic type, can only be true or false, any statement must either be None or return a
        /// Boolean (anything else is a compiler error). Any statement returning false (like a failing
        /// test) will also immediately cause an error at runtime or in the Editor via SCrunch.
        /// </summary>
        public const string Boolean = nameof(Boolean);
        /// <summary>
        /// Can be any floating point or integer number (think byte, short, int, long, float or double in
        /// other languages), but also can be a decimal or BigInteger, the compiler can decide and
        /// optimize this away to anything that makes sense in the current context.
        /// </summary>
        public const string Number = nameof(Number);
        /// <summary>
        /// Easy way to get another instance of the class type we are currently in.
        /// </summary>
        public const string Other = nameof(Other);
        public const string Character = nameof(Character);
        public const string Computation = nameof(Computation);
        public const string Count = nameof(Count);
        public const string Function = nameof(Function);
        public const string HashCode = nameof(HashCode);
        public const string Iteration = nameof(Iteration);
        public const string Mutable = nameof(Mutable);
        public const string Range = nameof(Range);
        public const string Text = nameof(Text);
        public const string Error = nameof(Error);
        public const string Slice = nameof(Slice);
        public const string Type = nameof(Type);
        public const string Log = nameof(Log);
        public const string App = nameof(App);
    }
}