#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ParseError {
    #[error("tree-sitter error.")]
    TSError,
    #[error("syntax error at {0} - {1}:\n{2}")]
    SyntaxError(tree_sitter::Point, tree_sitter::Point, String),
}

pub type Result<T> = std::result::Result<T, ParseError>;
