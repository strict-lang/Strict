# Bugs
Bugs to fix, most harmful first, one short line each. Remove a line once it is done, add what is found. No history or measurements (those go into commit messages). Speed work: [performance.md](performance.md). Anything large or difficult, features and far fetched things: [ideas.md](ideas.md).

- A program calling `Directory.Change` does not compile natively when nothing else uses list text: `'func.call' op 'strict_list_text_mapped' does not reference a valid function`, the generated strict_automatic_list functions need it but the runtime pruning removes it (`has logger`, `Run`, `Directory.Change("a")`).
