# Strict Grammar

Strict is easy to read and write, there is usually only one way to do things. It doesn't need fluff like end-of-line characters. Blocks are indented and have no start, end or brackets (like in Python). All lines are expressions and have to evaluate to true, otherwise the execution and even compilation stop at this point. Callers can evaluate expressions to check for this. This is how tests work, every method starts with some checks calling itself that need to evaluate to true.

These grammar files are not really used to generate any lexer, parser, tokenizer. They are here for informational purposes and to generate syntax highlighting like for Textmate (.tmLanguage), which can be imported to Visual Studio Code, Textmate, Atom, Ace, Sublime, etc.

* [Strict.bnf](Strict.ebnf) ([Backus-Naur form](https://en.wikipedia.org/wiki/Backus%E2%80%93Naur_form))
* [Strict.iro](strict.iro) ([Iro to create syntax highlighters for modern IDEs](https://medium.com/@model_train/creating-universal-syntax-highlighters-with-iro-549501698fd2))

To generate .tmLanguage (for Visual Studio Code or Textmate) or syntax highlighter files for other IDEs or tools, use https://eeyo.io/iro/ See https://github.com/Strict/strict-vscode-client for the implementation.

## Documentation

For more details check https://strict-lang.org/docs/Overview and https://strict-lang.org/blog

## Other languages

Strict has lots of similarity with C#, Java, C++, F#, Lisp, Scheme, Scala, etc. However, Lua and Python are syntax-wise probably the closest because of their simplicity and more simple look.
* https://www.lua.org/manual/5.1/manual.html
* https://docs.python.org/3/reference/grammar.html
* https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/language-specification/lexical-structure