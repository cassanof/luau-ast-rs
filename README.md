# A Parser and AST for Luau and Lua5.1 written in Rust

This is a tree-sitter-based parser for Luau and Lua5.1 written in Rust.
The grammar used is a fork of [polychromatist/tree-sitter-luau](https://github.com/cassanof/tree-sitter-luau).

This crate aims to provide a **_complete_** and **_fast_** parser for Luau, while also providing
pretty-printing and AST traversal utilities.

## Why did I make this and how is this different from other parsers?

1. _MAIN REASON_: It's error tolerant. The AST is segmented per-statement, so if there's an error in one statement, the rest of the AST will still be generated and will be traversable.
2. It's fast, memory efficient, _and thread-safe_, which means that it can be used to process large datasets of Lua files in parallel.
   I've achieved this by using tree-sitter as if it was a lexer, and then assembling the AST "semi-recursively".
   This semi-recursion is done by using a queue of statements to parse, and then parsing them one by one,
   while expressions are parsed recursively. This allows for a very fast parser, as it strikes a balance
   between the speed of a recursive descent parser and the speed of a PEG parser. The recursion is safe,
   as it won't stack overflow, because that in the case of a very deep expression, the call stack will be
   automatically extended into the heap, thanks to the `stacker` crate (see the function `parse_expr` in `parser.rs`).
   The visitors use the same queue and recursion strategy, so they're also very fast.
3. It includes comments, although these are only decorated to statements, not expressions.
4. The AST is stored in an arena-like structure, so it's very compact in memory, and can be easily
   serialized and deserialized without huge overhead. See [this blog post](https://www.cs.cornell.edu/~asampson/blog/flattening.html) by Prof. Adrian Sampson
   on how this little change can drastically improve performance.

## Example Usage

See this project I'm working on: [lua-dataset-heuristics](https://github.com/cassanof/lua-dataset-heuristics),
where I use this crate to parse a large dataset of ~500,000 Lua files and score them using a heuristic.
This parser is used to parse the files, and then the AST is traversed to find the relevant information.
All the files are processed in only a few minutes, which shows how fast this parser is.

### Current Status

- AST [COMPLETE]
- Parser [COMPLETE]
- AST Traversal through visitors [COMPLETE]
- Pretty-printing [NOT STARTED]
