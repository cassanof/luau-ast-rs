#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    pub block: Block,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub stmts: Vec<Stmt>,
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
