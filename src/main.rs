use std::io::Read;

use luau_ast_rs::parser::Parser;

// debug binary, that simply takes in a Lua/Luau program from stdin, parses it, and prints the AST to stdout
pub fn main() {
    let mut stdin = std::io::stdin();
    let mut buffer = String::new();
    stdin.read_to_string(&mut buffer).unwrap();
    let ast = Parser::new(&buffer).parse().unwrap();
    println!("{:#?}", ast);
}
