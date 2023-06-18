use crate::errors::ParseError;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Represents the status of a statement in the AST.
/// A status is Some when it is parsed correctly, None when it is removed from the AST by the user,
/// PreAllocated when it is allocated but not yet parsed, and Error when it is parsed incorrectly.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Block {
    pub stmt_ptrs: Vec<usize>,
}

/// Represents a bindings in the ast. e.g. the `a` in `local a = 1`, or the
/// `b: number` in `local b: number = 2`.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct Binding {
    pub name: String,
    pub ty: Option<Type>,
}

/// Represents a local variable declaration.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct Local {
    pub bindings: Vec<Binding>,
    pub init: Vec<Expr>,
}

/// Represents an assignment statement.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct Assign {
    pub vars: Vec<Var>,
    pub exprs: Vec<Expr>,
}

/// Represents the body of a function.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionBody {
    pub params: Vec<Binding>,
    pub generics: Vec<GenericParam>,
    pub vararg: bool,
    pub ret_ty: Option<Type>,
    pub block: Block,
}

/// Represents a global function definition.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDef {
    /// The table, with possible subtables, that the function is defined
    /// in. e.g. `a.b.c` is `vec!["a", "b", "c"]`.
    pub table: Vec<String>,
    /// Whether the function is a method. e.g. `function a:b() end` is a method.
    pub is_method: bool,
    /// The name of the function. e.g. `a` in `function a() end`.
    pub name: String,
    /// The body of the function.
    pub body: FunctionBody,
}

/// Represents a local function definition.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct LocalFunctionDef {
    pub name: String,
    pub body: FunctionBody,
}

/// Represents call arguments.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub enum CallArgs {
    Exprs(Vec<Expr>),
    Table(TableConstructor),
    String(String),
}

/// Represents a function call.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct Call {
    pub func: Expr,
    pub args: CallArgs,
}

/// Represents a binary operation. e.g. `a + b`, `a * b`, `a / b`, etc.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct BinOp {
    pub lhs: Expr,
    pub op: BinOpKind,
    pub rhs: Expr,
}

/// Represents a unary operation. e.g. `-a`, `not a`, and `#` (length operator).
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct UnOp {
    pub op: UnOpKind,
    pub expr: Expr,
}

/// Represents a compound operation. e.g. `a += b`, `a -= b`, `a *= b`, etc.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct CompOp {
    pub lhs: Var,
    pub op: CompOpKind,
    pub rhs: Box<Expr>,
}

/// Represents a return statement.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct Return {
    pub exprs: Vec<Expr>,
}

/// Represents a break statement.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct Break;

/// Represents a continue statement.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct Continue;

/// Represents a named type.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct NamedType {
    table: Option<String>,
    name: String,
    params: Vec<TypeParam>,
}

/// Represents a function type
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionType {
    generics: Vec<GenericParam>,
    params: Vec<Type>,
    ret_ty: Box<Type>,
}

/// Represents a do statement.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct Do {
    pub block: Block,
}

/// Represents a while statement.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct While {
    pub cond: Expr,
    pub block: Block,
}

/// Represents a repeat statement.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct Repeat {
    pub block: Block,
    pub cond: Expr,
}

/// Represents an if statement.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct If {
    pub cond: Expr,
    pub block: Block,
    pub else_if_blocks: Vec<(Expr, Block)>,
    pub else_block: Option<Block>,
}

/// Represents a for statement.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct For {
    pub var: Binding,
    pub start: Expr,
    pub end: Expr,
    pub step: Option<Expr>,
    pub block: Block,
}

/// Represents a for in statement.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct ForIn {
    pub vars: Vec<Binding>,
    pub exprs: Vec<Expr>,
    pub block: Block,
}

/// Represents a table constructor expression.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct TableConstructor {
    pub fields: Vec<TableField>,
}

/// Represents a ifelse expression.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct IfElseExp {
    pub cond: Expr,
    pub if_expr: Expr,
    pub else_expr: Expr,
    pub else_if_exprs: Vec<(Expr, Expr)>,
}

/// Represents a table access operation
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct TableAccess {
    pub expr: Expr,
    pub index: Expr,
}

/// Represents a field access operation
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct FieldAccess {
    pub expr: Expr,
    pub field: String,
}

/// Represents a statement node in the AST.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    CompOp(CompOp),
    Call(Call),
    Do(Do),
    While(While),
    Repeat(Repeat),
    If(If),
    For(For),
    ForIn(ForIn),
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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
    // NOTE: boxed because Call uses Expr
    Call(Box<Call>),
    TableConstructor(TableConstructor),
    // NOTE: boxed because FunctionBody uses Expr
    Function(Box<FunctionBody>),
    IfElse(Box<IfElseExp>),
    // StringInterp(StringInterp),
    // TypeAssertion(TypeAssertion),
    BinOp(Box<BinOp>),
    UnOp(Box<UnOp>),
}

/// The type of a binary operator.
/// binop = '+' | '-' | '*' | '/' | '^' | '%' | '..' | '<' | '<=' | '>' | '>=' | '==' | '~=' | 'and' | 'or'
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub enum UnOpKind {
    Neg,
    Not,
    Len,
}

/// The type of a compound operator.
/// compoundop :: '+=' | '-=' | '*=' | '/=' | '%=' | '^=' | '..='
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    // \ compounds, should be their own node but we don't care /
    /// ...T
    Variadic(Box<Type>),
    /// `( T, U )` or `( T, U, ...V )`
    Pack(Vec<Type>),
    // \ simple types /
    /// `( T )`
    Wrap(Box<Type>),
    /// `typeof(exp)`
    TypeOf(Expr),
    /// `T` or `T<PARAM1, PARAM2>` or `tbl.T`
    Named(NamedType),
    /// `{ [T] : U }` or `{ x : T }`
    Table(Vec<TableProp>),
    /// `( T ) -> U` or `<T, U...>( T ) -> U`
    Function(FunctionType),
    /// `T?`
    Optional(Box<Type>),
    /// `T | U`
    Union(Box<Type>),
    /// `T & U`
    Intersection(Box<Type>),
    // \ literals (singleton) /
    /// `nil`
    Nil,
    /// `"string"`
    String(String),
    /// `false` or `true`
    Bool(bool),
}

/// Represents a table property or indexer.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub enum TableProp {
    /// `[T] : U`
    Indexer { key: Type, value: Type },
    /// `x : T`
    Prop { key: String, value: Type },
}

/// Represents a type parameter node in the AST.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub enum TypeParam {
    // TODO
}

/// Represents a generic type parameter.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub enum GenericParam {
    /// A type parameter with a name. e.g. `T` in `function foo<T>() end`.
    Name(String),
    /// A pack type parameter. e.g. `function foo<T...>() end`.
    Pack(String),
}

/// Represents a variable node in the AST.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub enum Var {
    Name(String),
    TableAccess(Box<TableAccess>),
    FieldAccess(Box<FieldAccess>),
}

/// Represents a table field.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub enum TableField {
    /// Represents a field with an explicit key. e.g. `a = 1` in `{a = 1}`.
    ExplicitKey { key: String, value: Expr },
    /// Represents a field with an implicit key. e.g. `1` in `{1}`.
    ImplicitKey(Expr),
    /// Represents a field with an array-like key. e.g. `a` in `{[a] = 1}`.
    ArrayKey { key: Expr, value: Expr },
}
