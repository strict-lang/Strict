# Strict.Regex
Regular expressions for texts a user or a tool hands in. Strict's own parsing (Json, Xml, the compiler) does not need them, and code should not use them when `Text` methods (`StartsWith`, `IndexOf`, `Split`, `Replace`) are enough.

```
Regex("-?\\d+(\\.\\d+)?").FindAll("1, 22 and 3.5") is ("1", "22", "3.5")
Regex("(\\w+)=(\\w*)").Groups("set name=Strict") is ("name", "Strict")
Regex("(\\w+)=(\\w*)").Replace("a=1, b=2", "$2=$1") is "1=a, 2=b"
Regex(",\\s*").Split("a, b,c") is ("a", "b", "c")
```

`IsMatch` finds a match anywhere (anchor with `^` and `$`), `Find` gives the first match or "", `FindAll` every match left to right (an empty match moves one character on), `Groups` the groups of the first match ("" for a group that did not take part), `Replace` replaces every match with `$0` to `$9` filled in, `Split` the texts between matches. An invalid pattern makes every call return the `RegexParser.InvalidPattern` Error.

The syntax is the RE2 subset: literals, `.` (anything but a new line), classes `[a-z_]`, `[^...]` (a `]` right after `[` or `[^` is literal), `\d \w \s` and `\D \W \S` outside classes, `\t \n \r`, `\` before any other character for that character, groups `(...)` and `(?:...)`, `|`, `* + ?` and `{n}`, `{n,}`, `{n,m}`, each made lazy by a trailing `?`, and the anchors `^` and `$`. Matches are leftmost first like Perl, Python and Go. There are no backreferences or lookarounds: the pattern becomes a list of `RegexStep`s (`RegexParser`, `RegexPart`) and `RegexMachine` runs all threads of it side by side (Pike VM), so matching takes time proportional to the text times the pattern, never exponential (`(a*)*b` on a long run of `a` returns at once). A `Regex` parses its pattern once: keep one for many texts. Every search starts at the next occurrence of the literal text a pattern begins with (`constant \w+ = ` starts at `IndexOf("constant ")`), so a pattern with such a prefix is nearly as fast as `IndexOf` on texts without a match (`Examples/RegexSpeed`).
