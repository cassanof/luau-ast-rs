#[derive(Debug, Clone, PartialEq)]
pub enum StmtStatus {
    Some(Stmt),
    None,
    PreAllocated,
    // TODO: add syntax error node
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
    pub(crate) stmts: Vec<StmtStatus>,
}

impl Chunk {
    pub fn add_stmt(&mut self, stmt: Stmt) -> usize {
        let i = self.alloc();
        self.set_stmt(i, stmt);
        i
    }

    pub fn remove_stmt(&mut self, index: usize) {
        self.stmts[index] = StmtStatus::None;
    }

    pub fn get_stmt(&self, index: usize) -> Option<&Stmt> {
        match self.stmts[index] {
            StmtStatus::Some(ref stmt) => Some(stmt),
            StmtStatus::None => None,
            StmtStatus::PreAllocated => {
                panic!("called `Chunk::get_stmt()` on a `PreAllocated` value")
            }
        }
    }

    pub(crate) fn alloc(&mut self) -> usize {
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

    pub(crate) fn set_stmt(&mut self, index: usize, stmt: Stmt) {
        self.stmts[index] = StmtStatus::Some(stmt);
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
