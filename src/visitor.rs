use std::{
    collections::{HashMap, VecDeque},
    marker::PhantomData,
};

use crate::{ast::*, errors::ParseError};

type UnvisitedStmts = VecDeque<(
    usize, // parent stmt ptr
    usize, // unvisited stmt ptr
)>;

// TODO: add visitors for types
macro_rules! trait_visitor {
    // ref is a tt if it's &, but it's two tts if it's &mut
    ($chunk_gen:ident, $($ref:tt)+) => {
        fn visit_chunk(&mut self, _state: &VisitorDriverState, _chunk: $($ref)+ $chunk_gen) {}
        fn visit_block(&mut self, _state: &VisitorDriverState, _block: $($ref)+ Block) {}
        fn visit_parse_error(&mut self, _state: &VisitorDriverState, _error: $($ref)+ ParseError) {}
        fn visit_stmt(&mut self, _state: &VisitorDriverState, _stmt: $($ref)+ Stmt, _coms: $($ref)+ [Comment]) {}
        fn visit_comp_op(&mut self, _state: &VisitorDriverState, _comp_op: $($ref)+ CompOp) {}
        fn visit_bin_op(&mut self, _state: &VisitorDriverState, _bin_op: $($ref)+ BinOp) {}
        fn visit_un_op(&mut self, _state: &VisitorDriverState, _un_op: $($ref)+ UnOp) {}
        fn visit_call(&mut self, _state: &VisitorDriverState, _call: $($ref)+ Call) {}
        fn visit_callargs(&mut self, _state: &VisitorDriverState, _callargs: $($ref)+ CallArgs) {}
        fn visit_do(&mut self, _state: &VisitorDriverState, _do_: $($ref)+ Do) {}
        fn visit_while(&mut self, _state: &VisitorDriverState, _while_: $($ref)+ While) {}
        fn visit_repeat(&mut self, _state: &VisitorDriverState, _repeat: $($ref)+ Repeat) {}
        fn visit_if(&mut self, _state: &VisitorDriverState, _if_: $($ref)+ If) {}
        fn visit_if_else_expr(&mut self, _if_else_else: $($ref)+ IfElseExpr) {}
        fn visit_for(&mut self, _state: &VisitorDriverState, _for_: $($ref)+ For) {}
        fn visit_for_in(&mut self, _state: &VisitorDriverState, _for_in: $($ref)+ ForIn) {}
        fn visit_function_body(&mut self, _state: &VisitorDriverState, _function_body: $($ref)+ FunctionBody) {}
        fn visit_function_def(&mut self, _state: &VisitorDriverState, _function_def: $($ref)+ FunctionDef) {}
        fn visit_local_function_def(&mut self, _state: &VisitorDriverState, _local_function_def: $($ref)+ LocalFunctionDef) {}
        fn visit_function_expr(&mut self, _state: &VisitorDriverState, _function_expr: $($ref)+ FunctionBody) {}
        fn visit_local(&mut self, _state: &VisitorDriverState, _local: $($ref)+ Local) {}
        fn visit_assign(&mut self, _state: &VisitorDriverState, _assign: $($ref)+ Assign) {}
        fn visit_return(&mut self, _state: &VisitorDriverState, _return: $($ref)+ Return) {}
        fn visit_break(&mut self, _state: &VisitorDriverState, _break: $($ref)+ Break) {}
        fn visit_continue(&mut self, _state: &VisitorDriverState,  _continue: $($ref)+ Continue) {}
        fn visit_expr(&mut self, _state: &VisitorDriverState, _expr: $($ref)+ Expr) {}
        fn visit_number(&mut self, _state: &VisitorDriverState, _number: $($ref)+ f64) {}
        fn visit_string(&mut self, _state: &VisitorDriverState, _string: $($ref)+ str) {}
        fn visit_nil(&mut self, _state: &VisitorDriverState) {}
        fn visit_bool(&mut self, _state: &VisitorDriverState, _bool: $($ref)+ bool) {}
        fn visit_vararg(&mut self, _state: &VisitorDriverState) {}
        fn visit_var(&mut self, _state: &VisitorDriverState, _var: $($ref)+ Var) {}
        fn visit_table_access(&mut self, _state: &VisitorDriverState, _table_access: $($ref)+ TableAccess) {}
        fn visit_field_access(&mut self, _state: &VisitorDriverState, _field_access: $($ref)+ FieldAccess) {}
        fn visit_binding(&mut self, _state: &VisitorDriverState,  _binding: $($ref)+ Binding) {}
        fn visit_table_constructor(&mut self, _state: &VisitorDriverState, _table_constructor: $($ref)+ TableConstructor) {}
        fn visit_table_field(&mut self, _state: &VisitorDriverState, _table_field: $($ref)+ TableField) {}
        fn visit_string_interp(&mut self, _state: &VisitorDriverState, _string_interp: $($ref)+ StringInterp) {}
        fn visit_type_assertion(&mut self, _state: &VisitorDriverState, _type_assertion: $($ref)+ TypeAssertion) {}
        fn visit_type_def(&mut self, _state: &VisitorDriverState, _type_def: $($ref)+ TypeDef) {}
    };
}

macro_rules! impl_visitor_driver {
    // ref is a tt if it's &, but it's two tts if it's &mut
    ($($ref:tt)+) => {
        fn drive_block(&mut self, block: $($ref)+ Block, unv: &mut UnvisitedStmts) {
            self.visitor.visit_block(&self.state, block);
            let parent_ptr = self.state.curr_stmt;
            for stmt_ptr in block.stmt_ptrs.iter() {
                unv.push_back((parent_ptr, *stmt_ptr));
            }
        }

        fn drive_stmt(&mut self, stmt: $($ref)+ Stmt, coms: $($ref)+ [Comment], unv: &mut UnvisitedStmts) {
            self.visitor.visit_stmt(&self.state, stmt, coms);
            match stmt {
                Stmt::CompOp(s) => self.drive_comp_op(s, unv),
                Stmt::Call(s) => self.drive_call(s, unv),
                Stmt::Do(s) => self.drive_do(s, unv),
                Stmt::While(s) => self.drive_while(s, unv),
                Stmt::Repeat(s) => self.drive_repeat(s, unv),
                Stmt::If(s) => self.drive_if(s, unv),
                Stmt::For(s) => self.drive_for(s, unv),
                Stmt::ForIn(s) => self.drive_for_in(s, unv),
                Stmt::FunctionDef(s) => self.drive_function_def(s, unv),
                Stmt::LocalFunctionDef(s) => self.drive_local_function_def(s, unv),
                Stmt::Local(s) => self.drive_local(s, unv),
                Stmt::Assign(s) => self.drive_assign(s, unv),
                Stmt::Return(s) => self.drive_return(s, unv),
                Stmt::Break(s) => self.drive_break(s),
                Stmt::Continue(s) => self.drive_continue(s),
                Stmt::TypeDef(s) => self.drive_type_def(s, unv),
            }
        }

        fn drive_expr(&mut self, expr: $($ref)+ Expr, unv: &mut UnvisitedStmts) {
            self.visitor.visit_expr(&self.state, expr);
            stacker::maybe_grow(32 * 1024, 1024 * 1024, || {
                match expr {
                    Expr::Number(n) => self.visitor.visit_number(&self.state, n),
                    Expr::String(s) => self.visitor.visit_string(&self.state, s),
                    Expr::Nil => self.visitor.visit_nil(&self.state),
                    Expr::Bool(b) => self.visitor.visit_bool(&self.state, b),
                    Expr::VarArg => self.visitor.visit_vararg(&self.state),
                    Expr::Var(v) => self.drive_var(v, unv),
                    Expr::Call(c) => self.drive_call(c, unv),
                    Expr::TableConstructor(t) => self.drive_table_constructor(t, unv),
                    Expr::Function(f) => self.drive_function_expr(f, unv),
                    Expr::IfElseExpr(e) => self.drive_if_else_expr(e, unv),
                    Expr::BinOp(b) => self.drive_bin_op(b, unv),
                    Expr::UnOp(u) => self.drive_un_op(u, unv),
                    Expr::StringInterp(s) => self.drive_string_interp(s, unv),
                    Expr::TypeAssertion(t) => self.drive_type_assertion(t, unv),
                }
            });
        }

        fn drive_comp_op(&mut self, comp_op: $($ref)+ CompOp, unv: &mut UnvisitedStmts) {
            self.visitor.visit_comp_op(&self.state,comp_op);
            self.drive_expr($($ref)+ comp_op.rhs, unv);
            self.drive_var($($ref)+ comp_op.lhs, unv);
        }

        fn drive_call(&mut self, call: $($ref)+ Call, unv: &mut UnvisitedStmts) {
            self.visitor.visit_call(&self.state, call);
            self.drive_expr($($ref)+ call.func, unv);
            self.drive_callargs($($ref)+ call.args, unv);
        }

        fn drive_callargs(&mut self, callargs: $($ref)+ CallArgs, unv: &mut UnvisitedStmts) {
            self.visitor.visit_callargs(&self.state,callargs);
            match callargs {
                CallArgs::String(s) => self.visitor.visit_string(&self.state,s),
                CallArgs::Table(t) => self.drive_table_constructor(t, unv),
                CallArgs::Exprs(es) => {
                    for e in es {
                        self.drive_expr(e, unv);
                    }
                }
            }
        }

        fn drive_do(&mut self, do_: $($ref)+ Do, unv: &mut UnvisitedStmts) {
            self.visitor.visit_do(&self.state,do_);
            self.drive_block($($ref)+ do_.block, unv);
        }

        fn drive_while(&mut self, while_: $($ref)+ While, unv: &mut UnvisitedStmts) {
            self.visitor.visit_while(&self.state,while_);
            self.drive_expr($($ref)+ while_.cond, unv);
            self.drive_block($($ref)+ while_.block, unv);
        }

        fn drive_repeat(&mut self, repeat: $($ref)+ Repeat, unv: &mut UnvisitedStmts) {
            self.visitor.visit_repeat(&self.state,repeat);
            self.drive_block($($ref)+ repeat.block, unv);
            self.drive_expr($($ref)+ repeat.cond, unv);
        }

        fn drive_if(&mut self, if_: $($ref)+ If, unv: &mut UnvisitedStmts) {
            self.visitor.visit_if(&self.state,if_);
            self.drive_expr($($ref)+ if_.cond, unv);
            self.drive_block($($ref)+ if_.block, unv);
            for (cond, block) in $($ref)+ if_.else_if_blocks {
                self.drive_expr(cond, unv);
                self.drive_block(block, unv);
            }
            if let Some(else_) = $($ref)+ if_.else_block {
                self.drive_block(else_, unv);
            }
        }

        fn drive_for(&mut self, for_: $($ref)+ For, unv: &mut UnvisitedStmts) {
            self.visitor.visit_for(&self.state,for_);
            self.drive_binding($($ref)+ for_.var);
            self.drive_expr($($ref)+ for_.start, unv);
            self.drive_expr($($ref)+ for_.end, unv);
            if let Some(step) = $($ref)+ for_.step {
                self.drive_expr(step, unv);
            }
            self.drive_block($($ref)+ for_.block, unv);
        }

        fn drive_for_in(&mut self, for_in: $($ref)+ ForIn, unv: &mut UnvisitedStmts) {
            self.visitor.visit_for_in(&self.state,for_in);
            for v in $($ref)+ for_in.vars {
                self.drive_binding(v);
            }
            for e in $($ref)+ for_in.exprs {
                self.drive_expr(e, unv);
            }
            self.drive_block($($ref)+ for_in.block, unv);
        }

        fn drive_function_body(&mut self, function_body: $($ref)+ FunctionBody, unv: &mut UnvisitedStmts) {
            self.visitor.visit_function_body(&self.state,function_body);
            // TODO: traverse generics
            for p in $($ref)+ function_body.params {
                self.drive_binding(p);
            }
            // TODO: traverse ret type
            self.drive_block($($ref)+ function_body.block, unv);
        }

        fn drive_function_def(&mut self, function_def: $($ref)+ FunctionDef, unv: &mut UnvisitedStmts) {
            self.visitor.visit_function_def(&self.state,function_def);
            self.drive_function_body($($ref)+ function_def.body, unv);
        }

        fn drive_local_function_def(
            &mut self,
            local_function_def: $($ref)+ LocalFunctionDef,
            unv: &mut UnvisitedStmts,
        ) {
            self.visitor.visit_local_function_def(&self.state,local_function_def);
            self.drive_function_body($($ref)+ local_function_def.body, unv);
        }

        fn drive_local(&mut self, local: $($ref)+ Local, unv: &mut UnvisitedStmts) {
            self.visitor.visit_local(&self.state,local);
            for b in $($ref)+ local.bindings {
                self.drive_binding(b);
            }
            for e in $($ref)+ local.init {
                self.drive_expr(e, unv);
            }
        }

        fn drive_assign(&mut self, assign: $($ref)+ Assign, unv: &mut UnvisitedStmts) {
            self.visitor.visit_assign(&self.state,assign);
            for b in $($ref)+ assign.vars {
                self.drive_var(b, unv);
            }
            for e in $($ref)+ assign.exprs {
                self.drive_expr(e, unv);
            }
        }

        fn drive_return(&mut self, return_: $($ref)+ Return, unv: &mut UnvisitedStmts) {
            self.visitor.visit_return(&self.state,return_);
            for e in $($ref)+ return_.exprs {
                self.drive_expr(e, unv);
            }
        }

        fn drive_break(&mut self, break_: $($ref)+ Break) {
            self.visitor.visit_break(&self.state,break_);
        }

        fn drive_continue(&mut self, continue_: $($ref)+ Continue) {
            self.visitor.visit_continue(&self.state,continue_);
        }

        fn drive_var(&mut self, var: $($ref)+ Var, unv: &mut UnvisitedStmts) {
            self.visitor.visit_var(&self.state,var);
            match var {
                Var::Name(_) => {}
                Var::TableAccess(ta) => self.drive_table_access(ta, unv),
                Var::FieldAccess(fa) => self.drive_field_access(fa, unv),
            }
        }

        fn drive_table_access(&mut self, table_access: $($ref)+ TableAccess, unv: &mut UnvisitedStmts) {
            self.visitor.visit_table_access(&self.state,table_access);
            self.drive_expr($($ref)+ table_access.expr, unv);
            self.drive_expr($($ref)+ table_access.index, unv);
        }

        fn drive_field_access(&mut self, field_access: $($ref)+ FieldAccess, unv: &mut UnvisitedStmts) {
            self.visitor.visit_field_access(&self.state,field_access);
            self.drive_expr($($ref)+ field_access.expr, unv);
        }

        fn drive_binding(&mut self, binding: $($ref)+ Binding) {
            self.visitor.visit_binding(&self.state,binding);
            // TODO: type traversal
        }

        fn drive_table_constructor(
            &mut self,
            table_constructor: $($ref)+ TableConstructor,
            unv: &mut UnvisitedStmts,
        ) {
            self.visitor.visit_table_constructor(&self.state,table_constructor);
            for f in $($ref)+ table_constructor.fields {
                self.drive_table_field(f, unv);
            }
        }

        fn drive_table_field(&mut self, table_field: $($ref)+ TableField, unv: &mut UnvisitedStmts) {
            self.visitor.visit_table_field(&self.state,table_field);
            match table_field {
                TableField::ExplicitKey { key: _, value } => self.drive_expr(value, unv),
                TableField::ImplicitKey(value) => self.drive_expr(value, unv),
                TableField::ArrayKey {  key, value } => {
                    self.drive_expr(key, unv);
                    self.drive_expr(value, unv);
                }
            }
        }

        fn drive_function_expr(&mut self, function_expr: $($ref)+ FunctionBody, unv: &mut UnvisitedStmts) {
            self.visitor.visit_function_expr(&self.state,function_expr);
            self.drive_function_body(function_expr, unv);
        }

        fn drive_if_else_expr(&mut self, if_else_expr: $($ref)+ IfElseExpr, unv: &mut UnvisitedStmts) {
            self.visitor.visit_if_else_expr(if_else_expr);
            self.drive_expr($($ref)+ if_else_expr.cond, unv);
            self.drive_expr($($ref)+ if_else_expr.if_expr, unv);
            self.drive_expr($($ref)+ if_else_expr.else_expr, unv);
            for (cond, expr) in $($ref)+ if_else_expr.else_if_exprs {
                self.drive_expr(cond, unv);
                self.drive_expr(expr, unv);
            }
        }

        fn drive_bin_op(&mut self, bin_op: $($ref)+ BinOp, unv: &mut UnvisitedStmts) {
            self.visitor.visit_bin_op(&self.state,bin_op);
            self.drive_expr($($ref)+ bin_op.lhs, unv);
            self.drive_expr($($ref)+ bin_op.rhs, unv);
        }

        fn drive_un_op(&mut self, un_op: $($ref)+ UnOp, unv: &mut UnvisitedStmts) {
            self.visitor.visit_un_op(&self.state,un_op);
            self.drive_expr($($ref)+ un_op.expr, unv);
        }

        fn drive_string_interp(&mut self, string_interp: $($ref)+ StringInterp, unv: &mut UnvisitedStmts) {
            self.visitor.visit_string_interp(&self.state,string_interp);
            for p in $($ref)+ string_interp.parts {
                match p {
                    StringInterpPart::String(_) => {}
                    StringInterpPart::Expr(e) => self.drive_expr(e, unv),
                }
            }
        }

        fn drive_type_assertion(&mut self, type_assertion: $($ref)+ TypeAssertion, unv: &mut UnvisitedStmts) {
            self.visitor.visit_type_assertion(&self.state,type_assertion);
            self.drive_expr($($ref)+ type_assertion.expr, unv);
        }

        fn drive_type_def(&mut self, type_def: $($ref)+ TypeDef, _unv: &mut UnvisitedStmts) {
            self.visitor.visit_type_def(&self.state,type_def);
        }
    };
}

impl VisitorDriverState {
    /// Retrieves the pointer to the current statement that is being visited.
    pub fn curr_stmt(&self) -> usize {
        self.curr_stmt
    }

    /// Retrieves the parent index of the given statement pointer. If the given index is the
    /// root statement, then `None` is returned.
    pub fn parent_stmt(&self, stmt_ptr: usize) -> Option<usize> {
        self.from_map.get(&stmt_ptr).copied()
    }
}

pub trait Visitor<C: ChunkRef> {
    trait_visitor!(C, &);
}

pub trait VisitorMut {
    trait_visitor!(Chunk, &mut);
}

#[derive(Default)]
pub struct VisitorDriverState {
    curr_stmt: usize,
    /// map that maps stmt index to their parent stmt index. the root stmt isn't stored as
    /// a key in this map.
    from_map: HashMap<usize, usize>,
}

pub struct VisitorDriver<'a, C: ChunkRef, V: Visitor<C>> {
    visitor: &'a mut V,
    state: VisitorDriverState,
    _phantom: PhantomData<C>,
}

pub struct VisitorMutDriver<'a, V: VisitorMut> {
    visitor: &'a mut V,
    state: VisitorDriverState,
}

impl<'a, C: ChunkRef, V: Visitor<C>> VisitorDriver<'a, C, V> {
    /// Creates a new visitor driver.
    pub fn new(visitor: &'a mut V) -> Self {
        Self {
            visitor,
            state: VisitorDriverState::default(),
            _phantom: PhantomData,
        }
    }

    /// Runs the visitor over the chunk, visiting all statements in the chunk. The visitor
    /// will visit all statements and expressions in DFS order.
    ///
    /// # Panics
    /// - If the statement arena in the chunk is malformed and some statement pointer is out of
    /// bounds.
    /// - If a statement is marked as `StmtStatus::PreAllocated`.
    pub fn drive(&mut self, chunk: C) {
        self.visitor.visit_chunk(&self.state, &chunk);

        let mut unvisited_stmts = VecDeque::new();
        self.drive_block(chunk.block(), &mut unvisited_stmts);

        while let Some((parent_stmt_ptr, stmt_ptr)) = unvisited_stmts.pop_back() {
            let stmt_status = chunk.get_stmt(stmt_ptr);
            self.state.from_map.insert(stmt_ptr, parent_stmt_ptr);
            self.state.curr_stmt = stmt_ptr;
            match stmt_status {
                StmtStatus::Some(s, c) => self.drive_stmt(s, c, &mut unvisited_stmts),
                StmtStatus::None => {} // was removed, nothing to do
                StmtStatus::PreAllocated => panic!("PreAllocated stmts should not be visited"),
                StmtStatus::Error(p) => self.visitor.visit_parse_error(&self.state, p),
            }
        }
    }

    impl_visitor_driver!(&);
}

impl<'a, V: VisitorMut> VisitorMutDriver<'a, V> {
    /// Creates a new visitor driver.
    pub fn new(visitor: &'a mut V) -> Self {
        Self {
            visitor,
            state: VisitorDriverState::default(),
        }
    }

    /// Runs the visitor over the chunk, visiting all statements in the chunk. The visitor
    /// will visit all statements and expressions in DFS order.
    ///
    /// # Panics
    /// - If the statement arena in the chunk is malformed and some statement pointer is out of
    /// bounds.
    /// - If a statement is marked as `StmtStatus::PreAllocated`.
    pub fn drive(&mut self, chunk: &mut Chunk) {
        self.visitor.visit_chunk(&self.state, chunk);

        let mut unvisited_stmts = VecDeque::new();
        self.drive_block(&mut chunk.block, &mut unvisited_stmts);

        while let Some((parent_stmt_ptr, stmt_ptr)) = unvisited_stmts.pop_back() {
            let stmt_status = &mut chunk.stmts[stmt_ptr];
            self.state.from_map.insert(stmt_ptr, parent_stmt_ptr);
            self.state.curr_stmt = stmt_ptr;
            match stmt_status {
                StmtStatus::Some(s, c) => self.drive_stmt(s, c, &mut unvisited_stmts),
                StmtStatus::None => {} // was removed, nothing to do
                StmtStatus::PreAllocated => panic!("PreAllocated stmts should not be visited"),
                StmtStatus::Error(p) => self.visitor.visit_parse_error(&self.state, p),
            }
        }
    }

    impl_visitor_driver!(&mut);
}
