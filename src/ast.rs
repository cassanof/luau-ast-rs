use crate::errors::ParseError;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Represents the status of a statement in the AST.
/// A status is Some when it is parsed correctly, None when it is removed from the AST by the user,
/// PreAllocated when it is allocated but not yet parsed, and Error when it is parsed incorrectly.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub enum StmtStatus {
    Some(Stmt, Vec<Comment>),
    None,
    PreAllocated,
    Error(ParseError),
}

/// Represents a comment in the AST. We don't care much about the exact position of the comment, so we
/// aggregate them with the statement they are attached to.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub enum Comment {
    /// Comment that is trailing the statement. e.g. `local x = 1 -- trailing comment`
    /// Comments that are inside the statement also count as trailing comments.
    /// e.g. `local x --[[ trailing comment ]] = 1`
    Trailing(String),
    /// Comment that is leading the statement.
    /// e.g.
    /// ```lua
    /// -- leading comment
    /// local x = 1
    /// ```
    Leading(String),
}

/// The Chunk struct represents the root of the AST. It contains the root of the program in the
/// block field. All the statements in the AST are stored in the stmts vector. This struct
/// acts as an arena for the statements in the AST.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Chunk {
    pub(crate) block: Block,
    // TODO: benchmark different data structures for this arena
    /// This vector contains all the statements in the AST. These are referenced
    /// by index as if they were pointers in a linked list. It is an option
    /// in the case that the statement is removed from the AST or it is pre-allocated.
    /// We want to keep the indices the same, so we just replace the statement with None.
    pub(crate) stmts: Vec<StmtStatus>, // NOTE: pub(crate) for testing. use methods instead of accessing directly.
}

impl Iterator for Chunk {
    type Item = StmtStatus;

    fn next(&mut self) -> Option<Self::Item> {
        self.stmts.pop()
    }
}

pub trait ChunkInfo {
    /// Returns the number of statements in the chunk.
    fn len(&self) -> usize;

    /// Returns true if the chunk is empty, has no statements.
    #[must_use]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub trait ChunkRetrievable {
    /// Returns the statement at the given index.
    ///
    /// # Panics
    /// Panics if the index is out of bounds.
    fn get_stmt(&self, index: usize) -> &StmtStatus;

    /// Returns the main block of the chunk.
    fn block(&self) -> &Block;

    /// Produces a new `ChunkSlice` from the given `Block`.
    fn slice_from_block(&self, block: Block) -> ChunkSlice<'_>;
}

pub trait ChunkRef: ChunkInfo + ChunkRetrievable {}

impl ChunkInfo for &Chunk {
    fn len(&self) -> usize {
        self.stmts.len()
    }
}

impl ChunkRetrievable for &Chunk {
    fn get_stmt(&self, index: usize) -> &StmtStatus {
        &self.stmts[index]
    }

    fn block(&self) -> &Block {
        &self.block
    }

    fn slice_from_block(&self, block: Block) -> ChunkSlice<'_> {
        ChunkSlice { chunk: self, block }
    }
}

impl ChunkRef for &Chunk {}

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

    /// Adds a comment to the statement at the given index.
    ///
    /// # Panics
    /// If the given index is out of bounds, or if the statement at the given index is not Some,
    /// except if it is an Error, in which case it is ignored and the comment is not added.
    pub fn add_comment(&mut self, index: usize, comment: Comment) {
        match &mut self.stmts[index] {
            StmtStatus::Some(_, comments) => {
                comments.push(comment);
            }
            // ignore error case
            StmtStatus::Error(_) => {}
            StmtStatus::None => panic!(
                "Cannot add comment to statement that is None (ptr: {})",
                index
            ),
            StmtStatus::PreAllocated => panic!(
                "Cannot add comment to statement that is PreAllocated (ptr: {})",
                index
            ),
        }
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

/// A slice of a chunk, used to traverse subsets of the AST.
/// TODO: mutable version of this
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkSlice<'a> {
    chunk: &'a Chunk,
    block: Block,
}

impl<'a> ChunkRetrievable for ChunkSlice<'a> {
    fn get_stmt(&self, index: usize) -> &StmtStatus {
        self.chunk.get_stmt(index)
    }

    fn block(&self) -> &Block {
        &self.block
    }

    fn slice_from_block(&self, block: Block) -> ChunkSlice<'_> {
        ChunkSlice {
            chunk: self.chunk,
            block,
        }
    }
}

impl ChunkInfo for ChunkSlice<'_> {
    fn len(&self) -> usize {
        self.block.stmt_ptrs.len()
    }
}

impl<'a> ChunkRef for ChunkSlice<'a> {}

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
    /// The variadic argument, if any.
    /// None: no variadic argument
    /// Some(None): variadic argument with no type
    /// Some(Some(ty)): variadic argument with type
    pub vararg: Option<Option<Type>>,
    pub ret_ty: Option<TypeOrPack>,
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
    pub method: Option<String>,
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
    pub table: Option<String>,
    pub name: String,
    pub params: Vec<TypeOrPack>,
}

/// Represents a bound type.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct BoundType {
    pub name: String,
    pub ty: Box<Type>,
}

/// Represents a function type
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionType {
    pub generics: Vec<GenericParam>,
    pub params: TypeList,
    pub ret_ty: TypeOrPack,
}

/// Represents a table type
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct TableType {
    pub props: Vec<TableProp>,
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
pub struct IfElseExpr {
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

/// Represents a string interpolation operation
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct StringInterp {
    pub parts: Vec<StringInterpPart>,
}

/// Represents a type assertion
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct TypeAssertion {
    pub expr: Expr,
    pub ty: Type,
}

/// Represents a type list. e.g. `(a, b, c)`. or `(a, b, c, ...d)`
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TypeList {
    pub types: Vec<Type>,
    pub vararg: Option<Type>,
}

/// Represents a type definition
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct TypeDef {
    pub name: String,
    pub ty: Type,
    pub generics: Vec<GenericDef>,
    pub is_exported: bool,
}

/// Represents an union type
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct UnionType {
    pub left: Type,
    pub right: Type,
}

/// Represents an intersection type
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct IntersectionType {
    pub left: Type,
    pub right: Type,
}

/// Represents a generic definition. Used in type definitions.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct GenericDef {
    pub param: GenericParam,
    pub default: Option<TypeOrPack>,
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
    TypeDef(TypeDef),
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
    // NOTE: boxed because Call uses Expr
    Call(Box<Call>),
    TableConstructor(TableConstructor),
    // NOTE: boxed because FunctionBody uses Expr
    Function(Box<FunctionBody>),
    IfElseExpr(Box<IfElseExpr>),
    StringInterp(StringInterp),
    TypeAssertion(Box<TypeAssertion>),
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
    // \ special nodes, should be their own node but we don't care /
    /// T...
    /// This can typically only occur at the last parameter of a function.
    Pack(Box<Type>),
    // \ simple types /
    /// `typeof(exp)`
    TypeOf(Expr),
    /// `T` or `T<PARAM1, PARAM2>` or `tbl.T`
    Named(NamedType),
    /// `id : T`
    /// This is used in function type parameters.
    Bound(BoundType),
    /// `{ [T] : U }` or `{ x : T }`
    Table(TableType),
    /// `( T ) -> U` or `<T, U...>( T ) -> U`
    Function(Box<FunctionType>),
    /// `T?`
    Optional(Box<Type>),
    /// `T | U`
    Union(Box<UnionType>),
    /// `T & U`
    Intersection(Box<IntersectionType>),
    // \ literals (singleton) /
    /// `nil`
    Nil,
    /// `"string"`
    String(String),
    /// `false` or `true`
    Bool(bool),
}

/// Represents a type that is either a Type or TypePack
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub enum TypeOrPack {
    Type(Type),
    Pack(TypePack),
}

/// Represents a type pack
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub enum TypePack {
    /// `( T, U )` or `( T, U, ...V )`
    Listed(TypeList),
    /// `...T`
    Variadic(Type),
    /// `T...`
    Generic(Type),
}

/// Represents a table property or indexer.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub enum TableProp {
    /// `[T] : U`
    Indexer { key: Type, value: Type },
    /// `x : T`
    Prop { key: String, value: Type },
    /// `T`
    /// NOTE: when this prop appears, it will be the only prop in the table
    Array(Type),
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

/// Represents a part of a string interpolation.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub enum StringInterpPart {
    /// A string literal.
    String(String),
    /// An expression.
    Expr(Expr),
}
