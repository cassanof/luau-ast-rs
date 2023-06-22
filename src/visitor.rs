use crate::{ast::*, errors::ParseError};

type UnvisitedStmts = Vec<usize>; // where usize is the ptr in the chunk's stmts vec

macro_rules! trait_visitor {
    // ref is a tt if it's &, but it's two tts if it's &mut
    ($($ref:tt)+) => {
        fn visit_chunk(&mut self, _chunk: $($ref)+ Chunk) {}
        fn visit_block(&mut self, _block: $($ref)+ Block) {}
        fn visit_parse_error(&mut self, _error: $($ref)+ ParseError) {}
        fn visit_stmt(&mut self, _stmt: $($ref)+ Stmt, _coms: $($ref)+ [Comment]) {}
        fn visit_comp_op(&mut self, _comp_op: $($ref)+ CompOp) {}
        fn visit_bin_op(&mut self, _bin_op: $($ref)+ BinOp) {}
        fn visit_un_op(&mut self, _un_op: $($ref)+ UnOp) {}
        fn visit_call(&mut self, _call: $($ref)+ Call) {}
        fn visit_callargs(&mut self, _callargs: $($ref)+ CallArgs) {}
        fn visit_do(&mut self, _do_: $($ref)+ Do) {}
        fn visit_while(&mut self, _while_: $($ref)+ While) {}
        fn visit_repeat(&mut self, _repeat: $($ref)+ Repeat) {}
        fn visit_if(&mut self, _if_: $($ref)+ If) {}
        fn visit_if_else_expr(&mut self, _if_else_else: $($ref)+ IfElseExpr) {}
        fn visit_for(&mut self, _for_: $($ref)+ For) {}
        fn visit_for_in(&mut self, _for_in: $($ref)+ ForIn) {}
        fn visit_function_body(&mut self, _function_body: $($ref)+ FunctionBody) {}
        fn visit_function_def(&mut self, _function_def: $($ref)+ FunctionDef) {}
        fn visit_local_function_def(&mut self, _local_function_def: $($ref)+ LocalFunctionDef) {}
        fn visit_function_expr(&mut self, _function_expr: $($ref)+ FunctionBody) {}
        fn visit_local(&mut self, _local: $($ref)+ Local) {}
        fn visit_assign(&mut self, _assign: $($ref)+ Assign) {}
        fn visit_return(&mut self, _return: $($ref)+ Return) {}
        fn visit_break(&mut self, _break: $($ref)+ Break) {}
        fn visit_continue(&mut self, _continue: $($ref)+ Continue) {}
        fn visit_expr(&mut self, _expr: $($ref)+ Expr) {}
        fn visit_wrap(&mut self, _wrap: $($ref)+ Expr) {}
        fn visit_number(&mut self, _number: $($ref)+ f64) {}
        fn visit_string(&mut self, _string: $($ref)+ str) {}
        fn visit_nil(&mut self) {}
        fn visit_bool(&mut self, _bool: $($ref)+ bool) {}
        fn visit_vararg(&mut self) {}
        fn visit_var(&mut self, _var: $($ref)+ Var) {}
        fn visit_table_access(&mut self, _table_access: $($ref)+ TableAccess) {}
        fn visit_field_access(&mut self, _field_access: $($ref)+ FieldAccess) {}
        fn visit_binding(&mut self, _binding: $($ref)+ Binding) {}
        fn visit_table_constructor(&mut self, _table_constructor: $($ref)+ TableConstructor) {}
        fn visit_table_field(&mut self, _table_field: $($ref)+ TableField) {}
        fn visit_string_interp(&mut self, _string_interp: $($ref)+ StringInterp) {}
        fn visit_type_assertion(&mut self, _type_assertion: $($ref)+ TypeAssertion) {}
    };
}

macro_rules! impl_visitor_driver {
    // ref is a tt if it's &, but it's two tts if it's &mut
    ($($ref:tt)+) => {
        /// Creates a new visitor driver.
        pub fn new(visitor: &'a mut V) -> Self {
            Self { visitor }
        }

        /// Runs the visitor over the chunk, visiting all statements in the chunk.
        ///
        /// # Panics
        /// - If the statement arena in the chunk is malformed and some statement pointer is out of
        /// bounds.
        /// - If a statement is marked as `StmtStatus::PreAllocated`.
        pub fn drive(&mut self, chunk: $($ref)+ Chunk) {
            self.visitor.visit_chunk(chunk);

            let mut unvisited_stmts = Vec::new();
            self.drive_block($($ref)+ chunk.block, &mut unvisited_stmts);

            while let Some(stmt_ptr) = unvisited_stmts.pop() {
                let stmt_status = $($ref)+ chunk.stmts[stmt_ptr];
                match stmt_status {
                    StmtStatus::Some(s, c) => self.drive_stmt(s, c, &mut unvisited_stmts),
                    StmtStatus::None => {} // was removed, nothing to do
                    StmtStatus::PreAllocated => panic!("PreAllocated stmts should not be visited"),
                    StmtStatus::Error(p) => self.visitor.visit_parse_error(p),
                }
            }
        }

        fn drive_block(&mut self, block: $($ref)+ Block, unv: &mut UnvisitedStmts) {
            self.visitor.visit_block(block);
            for stmt_ptr in block.stmt_ptrs.iter() {
                unv.push(*stmt_ptr);
            }
        }

        fn drive_stmt(&mut self, stmt: $($ref)+ Stmt, coms: $($ref)+ [Comment], unv: &mut UnvisitedStmts) {
            self.visitor.visit_stmt(stmt, coms);
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
            }
        }

        fn drive_expr(&mut self, expr: $($ref)+ Expr, unv: &mut UnvisitedStmts) {
            self.visitor.visit_expr(expr);
            match expr {
                Expr::Number(n) => self.visitor.visit_number(n),
                Expr::String(s) => self.visitor.visit_string(s),
                Expr::Nil => self.visitor.visit_nil(),
                Expr::Bool(b) => self.visitor.visit_bool(b),
                Expr::VarArg => self.visitor.visit_vararg(),
                Expr::Var(v) => self.drive_var(v, unv),
                Expr::Wrap(e) => self.drive_wrap(e, unv),
                Expr::Call(c) => self.drive_call(c, unv),
                Expr::TableConstructor(t) => self.drive_table_constructor(t, unv),
                Expr::Function(f) => self.drive_function_expr(f, unv),
                Expr::IfElseExpr(e) => self.drive_if_else_expr(e, unv),
                Expr::BinOp(b) => self.drive_bin_op(b, unv),
                Expr::UnOp(u) => self.drive_un_op(u, unv),
                Expr::StringInterp(s) => self.drive_string_interp(s, unv),
                Expr::TypeAssertion(t) => self.drive_type_assertion(t, unv),
            }
        }

        fn drive_comp_op(&mut self, comp_op: $($ref)+ CompOp, unv: &mut UnvisitedStmts) {
            self.visitor.visit_comp_op(comp_op);
            self.drive_expr($($ref)+ comp_op.rhs, unv);
            self.drive_var($($ref)+ comp_op.lhs, unv);
        }

        fn drive_call(&mut self, call: $($ref)+ Call, unv: &mut UnvisitedStmts) {
            self.visitor.visit_call(call);
            self.drive_expr($($ref)+ call.func, unv);
            self.drive_callargs($($ref)+ call.args, unv);
        }

        fn drive_callargs(&mut self, callargs: $($ref)+ CallArgs, unv: &mut UnvisitedStmts) {
            self.visitor.visit_callargs(callargs);
            match callargs {
                CallArgs::String(s) => self.visitor.visit_string(s),
                CallArgs::Table(t) => self.drive_table_constructor(t, unv),
                CallArgs::Exprs(es) => {
                    for e in es {
                        self.drive_expr(e, unv);
                    }
                }
            }
        }

        fn drive_do(&mut self, do_: $($ref)+ Do, unv: &mut UnvisitedStmts) {
            self.visitor.visit_do(do_);
            self.drive_block($($ref)+ do_.block, unv);
        }

        fn drive_while(&mut self, while_: $($ref)+ While, unv: &mut UnvisitedStmts) {
            self.visitor.visit_while(while_);
            self.drive_expr($($ref)+ while_.cond, unv);
            self.drive_block($($ref)+ while_.block, unv);
        }

        fn drive_repeat(&mut self, repeat: $($ref)+ Repeat, unv: &mut UnvisitedStmts) {
            self.visitor.visit_repeat(repeat);
            self.drive_block($($ref)+ repeat.block, unv);
            self.drive_expr($($ref)+ repeat.cond, unv);
        }

        fn drive_if(&mut self, if_: $($ref)+ If, unv: &mut UnvisitedStmts) {
            self.visitor.visit_if(if_);
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
            self.visitor.visit_for(for_);
            self.drive_binding($($ref)+ for_.var);
            self.drive_expr($($ref)+ for_.start, unv);
            self.drive_expr($($ref)+ for_.end, unv);
            if let Some(step) = $($ref)+ for_.step {
                self.drive_expr(step, unv);
            }
            self.drive_block($($ref)+ for_.block, unv);
        }

        fn drive_for_in(&mut self, for_in: $($ref)+ ForIn, unv: &mut UnvisitedStmts) {
            self.visitor.visit_for_in(for_in);
            for v in $($ref)+ for_in.vars {
                self.drive_binding(v);
            }
            for e in $($ref)+ for_in.exprs {
                self.drive_expr(e, unv);
            }
            self.drive_block($($ref)+ for_in.block, unv);
        }

        fn drive_function_body(&mut self, function_body: $($ref)+ FunctionBody, unv: &mut UnvisitedStmts) {
            self.visitor.visit_function_body(function_body);
            // TODO: traverse generics
            for p in $($ref)+ function_body.params {
                self.drive_binding(p);
            }
            // TODO: traverse ret type
            self.drive_block($($ref)+ function_body.block, unv);
        }

        fn drive_function_def(&mut self, function_def: $($ref)+ FunctionDef, unv: &mut UnvisitedStmts) {
            self.visitor.visit_function_def(function_def);
            self.drive_function_body($($ref)+ function_def.body, unv);
        }

        fn drive_local_function_def(
            &mut self,
            local_function_def: $($ref)+ LocalFunctionDef,
            unv: &mut UnvisitedStmts,
        ) {
            self.visitor.visit_local_function_def(local_function_def);
            self.drive_function_body($($ref)+ local_function_def.body, unv);
        }

        fn drive_local(&mut self, local: $($ref)+ Local, unv: &mut UnvisitedStmts) {
            self.visitor.visit_local(local);
            for b in $($ref)+ local.bindings {
                self.drive_binding(b);
            }
            for e in $($ref)+ local.init {
                self.drive_expr(e, unv);
            }
        }

        fn drive_assign(&mut self, assign: $($ref)+ Assign, unv: &mut UnvisitedStmts) {
            self.visitor.visit_assign(assign);
            for b in $($ref)+ assign.vars {
                self.drive_var(b, unv);
            }
            for e in $($ref)+ assign.exprs {
                self.drive_expr(e, unv);
            }
        }

        fn drive_return(&mut self, return_: $($ref)+ Return, unv: &mut UnvisitedStmts) {
            self.visitor.visit_return(return_);
            for e in $($ref)+ return_.exprs {
                self.drive_expr(e, unv);
            }
        }

        fn drive_break(&mut self, break_: $($ref)+ Break) {
            self.visitor.visit_break(break_);
        }

        fn drive_continue(&mut self, continue_: $($ref)+ Continue) {
            self.visitor.visit_continue(continue_);
        }

        fn drive_var(&mut self, var: $($ref)+ Var, unv: &mut UnvisitedStmts) {
            self.visitor.visit_var(var);
            match var {
                Var::Name(_) => {}
                Var::TableAccess(ta) => self.drive_table_access(ta, unv),
                Var::FieldAccess(fa) => self.drive_field_access(fa, unv),
            }
        }

        fn drive_table_access(&mut self, table_access: $($ref)+ TableAccess, unv: &mut UnvisitedStmts) {
            self.visitor.visit_table_access(table_access);
            self.drive_expr($($ref)+ table_access.expr, unv);
            self.drive_expr($($ref)+ table_access.index, unv);
        }

        fn drive_field_access(&mut self, field_access: $($ref)+ FieldAccess, unv: &mut UnvisitedStmts) {
            self.visitor.visit_field_access(field_access);
            self.drive_expr($($ref)+ field_access.expr, unv);
        }

        fn drive_binding(&mut self, binding: $($ref)+ Binding) {
            self.visitor.visit_binding(binding);
            // TODO: type traversal
        }

        fn drive_wrap(&mut self, wrap: $($ref)+ Expr, unv: &mut UnvisitedStmts) {
            self.visitor.visit_wrap(wrap);
            self.drive_expr(wrap, unv);
        }

        fn drive_table_constructor(
            &mut self,
            table_constructor: $($ref)+ TableConstructor,
            unv: &mut UnvisitedStmts,
        ) {
            self.visitor.visit_table_constructor(table_constructor);
            for f in $($ref)+ table_constructor.fields {
                self.drive_table_field(f, unv);
            }
        }

        fn drive_table_field(&mut self, table_field: $($ref)+ TableField, unv: &mut UnvisitedStmts) {
            self.visitor.visit_table_field(table_field);
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
            self.visitor.visit_function_expr(function_expr);
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
            self.visitor.visit_bin_op(bin_op);
            self.drive_expr($($ref)+ bin_op.lhs, unv);
            self.drive_expr($($ref)+ bin_op.rhs, unv);
        }

        fn drive_un_op(&mut self, un_op: $($ref)+ UnOp, unv: &mut UnvisitedStmts) {
            self.visitor.visit_un_op(un_op);
            self.drive_expr($($ref)+ un_op.expr, unv);
        }

        fn drive_string_interp(&mut self, string_interp: $($ref)+ StringInterp, unv: &mut UnvisitedStmts) {
            self.visitor.visit_string_interp(string_interp);
            for p in $($ref)+ string_interp.parts {
                match p {
                    StringInterpPart::String(_) => {}
                    StringInterpPart::Expr(e) => self.drive_expr(e, unv),
                }
            }
        }

        fn drive_type_assertion(&mut self, type_assertion: $($ref)+ TypeAssertion, unv: &mut UnvisitedStmts) {
            self.visitor.visit_type_assertion(type_assertion);
            self.drive_expr($($ref)+ type_assertion.expr, unv);
        }
    };
}

pub trait Visitor {
    trait_visitor!(&);
}

pub trait VisitorMut {
    trait_visitor!(&mut);
}

pub struct VisitorDriver<'a, V: Visitor> {
    visitor: &'a mut V,
}

pub struct VisitorMutDriver<'a, V: VisitorMut> {
    visitor: &'a mut V,
}

impl<'a, V: Visitor> VisitorDriver<'a, V> {
    impl_visitor_driver!(&);
}

impl<'a, V: VisitorMut> VisitorMutDriver<'a, V> {
    impl_visitor_driver!(&mut);
}
