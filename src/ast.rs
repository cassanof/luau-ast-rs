use crate::errors::ParseError;

/// Represents the status of a statement in the AST.
/// A status is Some when it is parsed correctly, None when it is removed from the AST by the user,
/// PreAllocated when it is allocated but not yet parsed, and Error when it is parsed incorrectly.
#[derive(Debug, Clone, PartialEq)]
pub enum StmtStatus {
    Some(Stmt),
    None,
    PreAllocated,
    Error(ParseError),
}

impl StmtStatus {
    #[inline]
    #[track_caller]
    pub fn unwrap(self) -> Stmt {
        match self {
            StmtStatus::Some(stmt) => stmt,
            StmtStatus::None => panic!("called `StmtStatus::unwrap()` on a `None` value"),
            StmtStatus::PreAllocated => {
                panic!("called `StmtStatus::unwrap()` on a `PreAllocated` value")
            }
            StmtStatus::Error(err) => panic!(
                "called `StmtStatus::unwrap()` on an `Error` value: {:?}",
                err
            ),
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Chunk {
    pub block: Block,
    /// This vector contains all the statements in the AST. These are referenced
    /// by index as if they were pointers in a linked list. It is an option
    /// in the case that the statement is removed from the AST or it is pre-allocated.
    /// We want to keep the indices the same, so we just replace the statement with None.
    pub(crate) stmts: Vec<StmtStatus>, // NOTE: pub(crate) for testing. use methods instead of accessing directly.
}

impl Chunk {
    /// Allocates and adds a statement to the chunk.
    pub fn add_stmt(&mut self, stmt_status: StmtStatus) -> usize {
        let i = self.alloc();
        self.set_stmt(i, stmt_status);
        i
    }

    /// Removes a statement from the chunk at the given index. This will not deallocate the
    /// statement, but rather replace it with None.
    ///
    /// # Panics
    /// Panics if the index is out of bounds.
    pub fn remove_stmt(&mut self, index: usize) {
        self.set_stmt(index, StmtStatus::None);
    }

    /// Returns the statement at the given index.
    ///
    /// # Panics
    /// Panics if the index is out of bounds.
    pub fn get_stmt(&self, index: usize) -> &StmtStatus {
        &self.stmts[index]
    }

    /// Allocates space for a statement in the chunk, returning the pointer (as a index) to the space. The
    /// cell pointed to is set to StmtStatus::PreAllocated.
    pub fn alloc(&mut self) -> usize {
        // find if there are any holes in the vector
        let mut index = None;
        for (i, stmt) in self.stmts.iter().enumerate() {
            if let StmtStatus::None = stmt {
                index = Some(i);
                break;
            }
        }

        // if there are no holes, push the statement to the end
        // otherwise, replace the hole with the statement
        match index {
            Some(i) => i,
            None => {
                self.stmts.push(StmtStatus::PreAllocated);
                self.stmts.len() - 1
            }
        }
    }

    /// Sets the statement at the given index to the given statement.
    ///
    /// # Panics
    /// Panics if the index is out of bounds.
    pub fn set_stmt(&mut self, index: usize, stmt_status: StmtStatus) {
        self.stmts[index] = stmt_status;
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Block {
    pub stmts: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Binding {
    pub name: String,
    pub ty: Option<Type>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Local {
    pub bindings: Vec<Binding>,
    pub init: Vec<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Assign {
    pub lhs: Vec<Var>,
    pub rhs: Vec<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDef {
    pub name: String,
    pub params: Vec<Binding>,
    pub body: Block,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Call {
    pub func: Expr,
    pub args: Vec<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    // CompoundOp(Compound),
    Call(Call),
    // Do(Do),
    // While(While),
    // Repeat(Repeat),
    // If(If),
    // For(For),
    // ForIn(ForIn),
    FunctionDef(FunctionDef),
    // LocalFunction(LocalFunction),
    Local(Local),
    // Assign(Assign),
    // TypeDef(TypeDef),
    // \ These under are technically "last statements", but we don't care about that in AST form /
    // Return(Return),
    // Break(Break),
    // Continue(Continue),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    // \ simple expressions, essentially values /
    Number(f64),
    String(String),
    Nil,
    Bool(bool),
    Ellipsis,
    Var(Var),
    // TableConstructor(TableConstructor),
    // Function(Function),
    // PrefixExp(PrefixExp),
    // IfElse(IfElse),
    // StringInterp(StringInterp),
    // TypeAssertion(TypeAssertion),
    // Binop(Binop),
    // Unop(Unop),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    // TODO
}

#[derive(Debug, Clone, PartialEq)]
pub enum Var {
    Name(String),
    // TableAccess(TableAccess),
    // FieldAccess(FieldAccess),
}
