use lazy_static::lazy_static;

pub mod ast;
pub mod parser;
pub mod errors;
pub mod visitor;

lazy_static! {
    static ref LANG: tree_sitter::Language = tree_sitter_luau::language();
}
