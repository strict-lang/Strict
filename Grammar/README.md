# Strict Grammar

Strict is easy to read and write, there is usually only one way to do things and it doesn't need fluff like end of line characters. Blocks are indented and have no start, end or brakets (like in Python). All lines are expressions and have to evaluate to true, otherwise the execution and even compilation stops at this point. Callers can use catch blocks to check for this.

These grammar files are not really used to generate any lexer, parser, tokenizer. They are here for informational purposes and to generate syntax highlighting like for Textmate (.tmLanguage), which can be imported to Visual Studio Code, Textmate, Atom, Ace, Sublime, etc.

* [Strict.bnf](/Strict.bnf) ([Backus-Naur form](https://en.wikipedia.org/wiki/Backus%E2%80%93Naur_form))
* [Strict.iro](/Strict.iro) ([Iro to create syntax highlighters for modern IDEs](https://medium.com/@model_train/creating-universal-syntax-highlighters-with-iro-549501698fd2))

To generate .tmLanguage (for Visual Studio Code or Textmate) or syntax highlighter files for other IDEs or tools use https://eeyo.io/iro/

## Documentation

For more details check https://strict.dev/docs/Overview and https://strict.dev/blog

## Other languages

Strict has lots of similarity with C#, Java, C++, F#, Lisp, Scheme, etc. However Lua and Python are syntax-wise probably the closest because of their simplicity and more simple look.
* https://www.lua.org/manual/5.1/manual.html
* https://docs.python.org/3/reference/grammar.html
* https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/language-specification/lexical-structure