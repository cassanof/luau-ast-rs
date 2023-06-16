use crate::errors::ParseError;

/// Represents the status of a statement in the AST.
/// A status is Some when it is parsed correctly, None when it is removed from the AST by the user,
/// PreAllocated when it is allocated but not yet parsed, and Error when it is parsed incorrectly.
#[derive(Debug, Clone, PartialEq)]
pub enum StmtStatus {
    // TODO: annotate Some(Stmt) with comments (leading, traling)
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

/// The Chunk struct represents the root of the AST. It contains the root of the program in the
/// block field. All the statements in the AST are stored in the stmts vector. This struct
/// acts as an arena for the statements in the AST.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Chunk {
    pub block: Block,
    // TODO: benchmark different data structures for this arena
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

/// A block represents a list of statement. The statements here are pointers to the statements
/// that live in the chunk.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Block {
    pub stmt_ptrs: Vec<usize>,
}

/// Represents a bindings in the ast. e.g. the `a` in `local a = 1`, or the
/// `b: number` in `local b: number = 2`. This binding can also represent
/// the variadic arguments in a function definition. e.g. `function f(...)`
/// in that case, the name field is "...".
#[derive(Debug, Clone, PartialEq)]
pub struct Binding {
    pub name: String,
    pub ty: Option<Type>,
}

/// Represents a local variable declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct Local {
    pub bindings: Vec<Binding>,
    pub init: Vec<Expr>,
}

/// Represents an assignment statement.
#[derive(Debug, Clone, PartialEq)]
pub struct Assign {
    pub vars: Vec<Var>,
    pub exprs: Vec<Expr>,
}

/// Represents the body of a function.
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionBody {
    pub params: Vec<Binding>,
    pub generics: Vec<GenericParam>,
    pub ret_ty: Option<Type>,
    pub block: Block,
}

/// Represents a global function definition.
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDef {
    /// The table that is being assigned to. e.g. `a` in `a.b = function() end`.
    pub table: Option<String>,
    /// Whether the function is a method. e.g. `function a:b() end` is a method.
    pub is_method: bool,
    /// The name of the function. e.g. `a` in `function a() end`.
    pub name: String,
    pub body: FunctionBody,
}

/// Represents a local function definition.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalFunctionDef {
    pub name: String,
    pub body: FunctionBody,
}

/// Represents a function call.
#[derive(Debug, Clone, PartialEq)]
pub struct Call {
    pub func: Expr,
    pub args: Vec<Expr>,
}

/// Represents a binary operation. e.g. `a + b`, `a * b`, `a / b`, etc.
#[derive(Debug, Clone, PartialEq)]
pub struct BinOp {
    pub lhs: Box<Expr>,
    pub op: BinOpKind,
    pub rhs: Box<Expr>,
}

/// Represents a unary operation. e.g. `-a`, `not a`, and `#` (length operator).
#[derive(Debug, Clone, PartialEq)]
pub struct UnOp {
    pub op: UnOpKind,
    pub expr: Box<Expr>,
}

/// Represents a compound operation. e.g. `a += b`, `a -= b`, `a *= b`, etc.
#[derive(Debug, Clone, PartialEq)]
pub struct CompOp {
    pub lhs: Var,
    pub op: CompOpKind,
    pub rhs: Box<Expr>,
}

/// Represents a return statement.
#[derive(Debug, Clone, PartialEq)]
pub struct Return {
    pub exprs: Vec<Expr>,
}

/// Represents a break statement.
#[derive(Debug, Clone, PartialEq)]
pub struct Break;

/// Represents a continue statement.
#[derive(Debug, Clone, PartialEq)]
pub struct Continue;

/// Represents a statement node in the AST.
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    CompOp(CompOp),
    Call(Call),
    // Do(Do),
    // While(While),
    // Repeat(Repeat),
    // If(If),
    // For(For),
    // ForIn(ForIn),
    FunctionDef(FunctionDef),
    LocalFunctionDef(LocalFunctionDef),
    Local(Local),
    Assign(Assign),
    // TypeDef(TypeDef),
    // \ These under are technically "last statements", but we don't care about that in AST form /
    Return(Return),
    Break(Break),
    Continue(Continue),
}

/// Represents an expression node in the AST.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    // \ simple expressions, essentially values /
    Number(f64),
    String(String),
    Nil,
    Bool(bool),
    VarArg,
    Var(Var),
    Wrap(Box<Expr>),
    // TableConstructor(TableConstructor),
    // Function(Function),
    // PrefixExp(PrefixExp),
    // IfElse(IfElse),
    // StringInterp(StringInterp),
    // TypeAssertion(TypeAssertion),
    BinOp(BinOp),
    UnOp(UnOp),
}

/// The type of a binary operator.
/// binop = '+' | '-' | '*' | '/' | '^' | '%' | '..' | '<' | '<=' | '>' | '>=' | '==' | '~=' | 'and' | 'or'
#[derive(Debug, Clone, PartialEq)]
pub enum BinOpKind {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Mod,
    Concat,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    And,
    Or,
}

/// The type of a unary operator.
/// unop = '-' | 'not' | '#'
#[derive(Debug, Clone, PartialEq)]
pub enum UnOpKind {
    Neg,
    Not,
    Len,
}

/// The type of a compound operator.
/// compoundop :: '+=' | '-=' | '*=' | '/=' | '%=' | '^=' | '..='
#[derive(Debug, Clone, PartialEq)]
pub enum CompOpKind {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    Concat,
}

/// Represents a type node in the AST.
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    // TODO
}

/// Represents a generic type parameter.
#[derive(Debug, Clone, PartialEq)]
pub enum GenericParam {
    /// A type parameter with a name. e.g. `T` in `function foo<T>() end`.
    Name(String),
    /// A pack type parameter. e.g. `function foo<T...>() end`.
    Pack(String),
}

/// Represents a variable node in the AST.
#[derive(Debug, Clone, PartialEq)]
pub enum Var {
    Name(String),
    // TableAccess(TableAccess),
    // FieldAccess(FieldAccess),
}
