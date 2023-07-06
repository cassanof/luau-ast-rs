use lazy_static::lazy_static;

pub mod ast;
pub mod errors;
pub mod parser;
#[cfg(test)]
mod parser_tests;
pub mod visitor;

lazy_static! {
    static ref LANG: tree_sitter::Language = luau_ast_rs_grammar::language();
}
