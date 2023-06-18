#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ParseError {
    #[error("tree-sitter error.")]
    TSError,
    #[error("syntax error at {start_row}:{start_col} - {end_row}:{end_col}\n{snippet}")]
    SyntaxError {
        start_row: usize,
        start_col: usize,
        end_row: usize,
        end_col: usize,
        snippet: String,
    },
}

pub type Result<T> = std::result::Result<T, ParseError>;
