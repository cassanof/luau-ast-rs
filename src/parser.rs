use std::str::FromStr;

use crate::ast::*;
use crate::errors::*;
use crate::LANG;

pub(crate) fn ts_parser() -> tree_sitter::Parser {
    let mut p = tree_sitter::Parser::new();
    p.set_language(*LANG).unwrap();
    p
}

pub struct Parser<'s> {
    ts: tree_sitter::Parser,
    text: &'s str,
    chunk: Chunk,
}

struct StmtToBeParsed<'ts> {
    ptr: usize,
    node: tree_sitter::Node<'ts>,
}

type UnparsedStmts<'ts> = Vec<StmtToBeParsed<'ts>>;

struct NodeSiblingIter<'a> {
    node: Option<tree_sitter::Node<'a>>,
}

impl<'a> Iterator for NodeSiblingIter<'a> {
    type Item = tree_sitter::Node<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.node;
        self.node = current.and_then(|n| n.next_sibling());
        current
    }
}

// TODO: write failure tests and remove ok_or_else, use unwrap_unchecked and benchmark
// or better, write states such that they carry the data

impl<'s, 'ts> Parser<'s> {
    pub fn new(text: &'s str) -> Self {
        let ts = ts_parser();
        Self {
            ts,
            text,
            chunk: Chunk::default(),
        }
    }

    pub fn parse(mut self) -> Result<Chunk> {
        let tree = self.ts.parse(self.text, None).ok_or(ParseError::TSError)?;
        let root = tree.root_node();
        let mut to_be_parsed = Vec::new();
        let block = self.parse_block(root, &mut to_be_parsed)?;
        self.chunk.block = block;

        while let Some(StmtToBeParsed { ptr, node }) = to_be_parsed.pop() {
            match self.parse_stmt(node, &mut to_be_parsed) {
                Ok(stmt) => {
                    self.chunk.set_stmt(ptr, StmtStatus::Some(stmt));
                }
                Err(err) => {
                    self.chunk.set_stmt(ptr, StmtStatus::Error(err));
                }
            }
        }

        Ok(self.chunk)
    }

    fn extract_text(&self, node: tree_sitter::Node<'ts>) -> &'s str {
        let start = node.start_byte();
        let end = node.end_byte();
        &self.text[start..end]
    }

    fn error(&self, node: tree_sitter::Node<'ts>) -> ParseError {
        let start = node.start_position();
        let end = node.end_position();

        let text = self.text;

        let clip_start = std::cmp::max(0, (start.row as i32) - 3) as usize;
        let clip_end = end.row + 3;

        let mut clipped = String::new();

        for (i, line) in text.lines().skip(clip_start).enumerate() {
            let i = i + clip_start;
            clipped.push_str(line);
            clipped.push('\n');
            // if i is in range of start.row and end.row, add a marker
            if i >= start.row && i <= end.row {
                let start_col = if i == start.row { start.column } else { 0 };
                let end_col = if i == end.row { end.column } else { line.len() };
                let marker = " ".repeat(start_col) + &"^".repeat(end_col - start_col);
                clipped.push_str(&marker);
                clipped.push('\n');
            }

            if i == clip_end {
                break;
            }
        }

        ParseError::SyntaxError {
            start_row: start.row,
            start_col: start.column,
            end_row: end.row,
            end_col: end.column,
            snippet: clipped,
        }
    }

    fn parse_block(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<Block> {
        let mut stmts = Vec::new();
        let cursor = &mut node.walk();
        for child in node.children(cursor) {
            let stmt_ptr = self.chunk.alloc();
            stmts.push(stmt_ptr);
            unp.push(StmtToBeParsed {
                ptr: stmt_ptr,
                node: child,
            });
        }
        Ok(Block { stmt_ptrs: stmts })
    }

    fn parse_stmt(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<Stmt> {
        let kind = node.kind();

        // macro for tidyness
        macro_rules! ez_parse {
            ($parse_fn:ident, $stmt:expr) => {{
                let stmt = self.$parse_fn(node, unp)?;
                $stmt(stmt)
            }};
        }

        let stmt = match kind {
            // \ all these statements create additional scopes /
            "local_var_stmt" => ez_parse!(parse_local, Stmt::Local),
            "fn_stmt" => ez_parse!(parse_function_def, Stmt::FunctionDef),
            "local_fn_stmt" => ez_parse!(parse_local_function_def, Stmt::LocalFunctionDef),
            "do_stmt" => ez_parse!(parse_do, Stmt::Do),
            "while_stmt" => ez_parse!(parse_while, Stmt::While),
            "repeat_stmt" => ez_parse!(parse_repeat, Stmt::Repeat),
            "if_stmt" => ez_parse!(parse_if, Stmt::If),
            "for_range_stmt" => ez_parse!(parse_for, Stmt::For),
            "for_in_stmt" => ez_parse!(parse_for_in, Stmt::ForIn),
            // \ statements that don't contain other statements /
            "call_stmt" => ez_parse!(parse_call, Stmt::Call),
            "var_stmt" => ez_parse!(parse_compop, Stmt::CompOp),
            "ret_stmt" => ez_parse!(parse_return, Stmt::Return),
            "assign_stmt" => ez_parse!(parse_assign, Stmt::Assign),
            // \ these don't need to be parsed separately /
            "break_stmt" => Stmt::Break(Break {}),
            "continue_stmt" => Stmt::Continue(Continue {}),
            _ => todo!("parse_stmt: {}", kind),
        };
        Ok(stmt)
    }

    fn parse_local(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<Local> {
        let cursor = &mut node.walk();
        let mut bindings = Vec::new();
        let mut init = Vec::new();

        let mut parsing_init = false;
        for child in node.children(cursor) {
            let kind = child.kind();
            match kind {
                "binding" => bindings.push(self.parse_binding(child)?),
                "local" | "," => {}

                // start of init exprs
                "=" => {
                    parsing_init = true;
                }
                // delegate to expr parser
                _ if parsing_init => init.push(self.parse_expr(child, unp)?),
                _ => todo!("parse_local: {}", kind),
            }
        }
        Ok(Local { bindings, init })
    }

    fn parse_fn_body(
        &mut self,
        // NOTE: the node has to be the "(" or "<" of the function def
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<FunctionBody> {
        let mut params = Vec::new();
        let mut block = None;
        let mut generics = Vec::new();
        let mut ret_ty = None;

        enum State {
            Generics,
            Params,
            Block,
        }

        let mut state = State::Generics;

        for node in (NodeSiblingIter { node: Some(node) }) {
            let kind = node.kind();
            match (kind, &state) {
                ("<", State::Generics) => {}
                ("generic", State::Generics) => {
                    let txt = self.extract_text(node);
                    generics.push(GenericParam::Name(txt.to_string()));
                }
                ("genpack", State::Generics) => {
                    let txt = self.extract_text(node.child(0).ok_or_else(|| self.error(node))?);
                    generics.push(GenericParam::Pack(txt.to_string()));
                }
                (">", State::Generics) => state = State::Params,
                (",", State::Generics | State::Params) => {}
                ("(", State::Generics | State::Params) => state = State::Params,
                ("param", State::Params) => params.push(self.parse_binding(node)?),
                (")", State::Params) => state = State::Block,
                ("block", State::Block) => {
                    let mut unparsed_stmts = Vec::new();
                    let b = self.parse_block(node, &mut unparsed_stmts)?;
                    unp.extend(unparsed_stmts);
                    block = Some(b);
                }
                ("end", State::Block) => break,
                _ => todo!("parse_fn_body: {}", kind),
            }
        }

        Ok(FunctionBody {
            params,
            generics,
            ret_ty,
            // there may be no block if the function is empty
            block: block.unwrap_or_default(),
        })
    }

    fn parse_function_def(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<FunctionDef> {
        let cursor = &mut node.walk();

        // vars to fill
        let mut table = Vec::new();
        let mut is_method = false;
        let mut name = String::new();
        let mut body = None;

        #[derive(Debug)] // TODO: delete
        enum State {
            // we assume that a function def starts with just it's name
            Name,
            // the name was actually a table, so now we need to parse the actual name
            NameWasTable,
        }

        let mut state = State::Name;

        println!("parse_function_def: {}", node.to_sexp());

        for child in node.children(cursor) {
            let kind = child.kind();
            match kind {
                "name" => match state {
                    State::Name => {
                        name.push_str(self.extract_text(child));
                    }
                    State::NameWasTable => {
                        let table_name = self.extract_text(child);
                        let swapparoo = std::mem::replace(&mut name, table_name.to_string());
                        // NOTE: we insert 0 because in the case of multiple fields, this is going
                        // to be the last state, and we want to insert the name at the start
                        table.insert(0, swapparoo);
                    }
                },
                "field" => {
                    let table_name = self.extract_text(child);
                    // NOTE: [1..] skips the . at the start
                    table.push(table_name[1..].to_string());
                }
                "<" | "(" => {
                    let parsed_body = self.parse_fn_body(child, unp)?;
                    body = Some(parsed_body);
                    break;
                }
                "." => {
                    state = State::NameWasTable;
                }
                ":" => {
                    state = State::NameWasTable;
                    is_method = true;
                }
                "function" => {}
                _ => todo!("parse_function_def: {}", kind),
            }
        }
        Ok(FunctionDef {
            table,
            is_method,
            name,
            body: body.ok_or_else(|| self.error(node))?,
        })
    }

    fn parse_local_function_def(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<LocalFunctionDef> {
        let cursor = &mut node.walk();

        // vars to fill
        let mut name = String::new();
        let mut body = None;

        for child in node.children(cursor) {
            let kind = child.kind();
            match kind {
                "name" => name.push_str(self.extract_text(child)),
                "<" | "(" => {
                    let parsed_body = self.parse_fn_body(child, unp)?;
                    body = Some(parsed_body);
                    break;
                }
                "local" | "function" => {}
                _ => todo!("parse_function_def: {}", kind),
            }
        }
        Ok(LocalFunctionDef {
            name,
            body: body.ok_or_else(|| self.error(node))?,
        })
    }

    fn parse_do(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<Do> {
        let cursor = &mut node.walk();
        let mut block = None;

        for child in node.children(cursor) {
            let kind = child.kind();
            match kind {
                "do" | "end" => {}
                "block" => {
                    let b = self.parse_block(child, unp)?;
                    block = Some(b);
                }
                _ => return Err(self.error(child)),
            }
        }

        Ok(Do {
            block: block.unwrap_or_default(),
        })
    }

    fn parse_while(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<While> {
        let cursor = &mut node.walk();
        let mut cond = None;
        let mut block = None;

        enum State {
            Condition,
            Block,
        }

        let mut state = State::Condition;

        for child in node.children(cursor) {
            let kind = child.kind();
            match kind {
                "while" | "do" | "end" => {}
                _ if matches!(state, State::Condition) => {
                    cond = Some(self.parse_expr(child, unp)?);
                    state = State::Block;
                }
                "block" => {
                    let b = self.parse_block(child, unp)?;
                    block = Some(b);
                }
                _ => return Err(self.error(child)),
            }
        }

        Ok(While {
            cond: cond.ok_or_else(|| self.error(node))?,
            block: block.unwrap_or_default(),
        })
    }

    fn parse_repeat(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<Repeat> {
        let cursor = &mut node.walk();
        let mut cond = None;
        let mut block = None;

        enum State {
            Block,
            Condition,
        }

        let mut state = State::Block;

        for child in node.children(cursor) {
            let kind = child.kind();
            match kind {
                "repeat" | "until" => {}
                "block" => {
                    let b = self.parse_block(child, unp)?;
                    block = Some(b);
                    state = State::Condition;
                }
                _ if matches!(state, State::Condition) => {
                    cond = Some(self.parse_expr(child, unp)?);
                }
                _ => return Err(self.error(child)),
            }
        }

        Ok(Repeat {
            cond: cond.ok_or_else(|| self.error(node))?,
            block: block.unwrap_or_default(),
        })
    }

    fn parse_if(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<If> {
        let cursor = &mut node.walk();
        let mut cond = None;
        let mut if_block = None;
        let mut else_if_blocks = Vec::new();
        let mut else_block = None;

        enum State {
            Condition,
            Else,
            Block,
        }

        let mut state = State::Condition;

        for child in node.children(cursor) {
            let kind = child.kind();
            match (kind, &state) {
                ("end", _) => {}
                ("if", State::Condition) => {}
                ("then", State::Block | State::Else) => {}
                (_, State::Condition) => {
                    cond = Some(self.parse_expr(child, unp)?);
                    state = State::Block;
                }
                (_, State::Block) => {
                    if_block = Some(self.parse_block(child, unp)?);
                    state = State::Else;
                }
                ("else_clause", State::Else) => {
                    else_block = Some(
                        child
                            .child(1)
                            .map(|n| self.parse_block(n, unp))
                            .unwrap_or_else(|| Ok(Block::default()))?,
                    );
                }
                ("elseif_clause", State::Else) => {
                    let cond_node = child.child(1).ok_or_else(|| self.error(child))?;
                    let cond = self.parse_expr(cond_node, unp)?;
                    let b = child
                        .child(3)
                        .map(|n| self.parse_block(n, unp))
                        .unwrap_or_else(|| Ok(Block::default()))?;
                    else_if_blocks.push((cond, b));
                }
                _ => return Err(self.error(child)),
            }
        }

        Ok(If {
            cond: cond.ok_or_else(|| self.error(node))?,
            block: if_block.unwrap_or_default(),
            else_if_blocks,
            else_block,
        })
    }

    fn parse_for(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<For> {
        let cursor = &mut node.walk();
        let mut var = None;
        let mut start = None;
        let mut end = None;
        let mut step = None;
        let mut block = None;

        enum State {
            Var,
            Start,
            End,
            MaybeStep,
            Step,
            Block,
        }

        let mut state = State::Var;

        for child in node.children(cursor) {
            let kind = child.kind();
            match (kind, &state) {
                ("for" | "do" | "end", _) => {}
                ("binding", State::Var) => {
                    var = Some(self.parse_binding(child)?);
                    state = State::Start;
                }
                ("=", State::Start) => {}
                (_, State::Start) => {
                    start = Some(self.parse_expr(child, unp)?);
                    state = State::End;
                }
                (",", State::End) => {}
                (_, State::End) => {
                    end = Some(self.parse_expr(child, unp)?);
                    state = State::MaybeStep;
                }
                (",", State::MaybeStep) => state = State::Step,
                (_, State::Step) => {
                    step = Some(self.parse_expr(child, unp)?);
                    state = State::Block;
                }
                (_, State::MaybeStep | State::Block) => {
                    block = Some(self.parse_block(child, unp)?);
                }
                _ => return Err(self.error(child)),
            }
        }

        Ok(For {
            var: var.ok_or_else(|| self.error(node))?,
            start: start.ok_or_else(|| self.error(node))?,
            end: end.ok_or_else(|| self.error(node))?,
            step,
            block: block.unwrap_or_default(),
        })
    }

    fn parse_for_in(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<ForIn> {
        let cursor = &mut node.walk();
        let mut vars = Vec::new();
        let mut exprs = Vec::new();
        let mut block = None;

        enum State {
            Var,
            Expr,
            Block,
        }

        let mut state = State::Var;

        for child in node.children(cursor) {
            let kind = child.kind();
            match (kind, &state) {
                ("for" | "end", _) => {}
                ("in", State::Var) => state = State::Expr,
                ("do", State::Expr) => state = State::Block,
                ("binding", State::Var) => {
                    vars.push(self.parse_binding(child)?);
                }
                (",", State::Expr | State::Var) => {}
                (_, State::Expr) => {
                    exprs.push(self.parse_expr(child, unp)?);
                }
                (_, State::Block) => {
                    block = Some(self.parse_block(child, unp)?);
                }
                _ => return Err(self.error(child)),
            }
        }

        Ok(ForIn {
            vars,
            exprs,
            block: block.unwrap_or_default(),
        })
    }

    fn parse_call(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<Call> {
        let cursor = &mut node.walk();
        let mut expr = None;
        let mut args = None;

        let mut parsing_expr = true; // first child is always expr

        for child in node.children(cursor) {
            let kind = child.kind();
            match kind {
                _ if parsing_expr => {
                    expr = Some(self.parse_expr(child, unp)?);
                    parsing_expr = false
                }
                "arglist" => {
                    let cursor = &mut child.walk();
                    for arg in child.children(cursor) {
                        let kind = arg.kind();
                        match (kind, &mut args) {
                            ("(", None) => args = Some(CallArgs::Exprs(Vec::new())),
                            (")" | ",", Some(CallArgs::Exprs(_))) => {}
                            (_, Some(CallArgs::Exprs(exp_args))) => {
                                exp_args.push(self.parse_expr(arg, unp)?)
                            }
                            ("table", None) => {
                                args = Some(CallArgs::Table(self.parse_tableconstructor(arg, unp)?))
                            }
                            ("string", None) => {
                                let txt = self.extract_text(arg);
                                args = Some(CallArgs::String(txt.to_string()))
                            }
                            _ => return Err(self.error(arg)),
                        }
                    }
                }
                _ => return Err(self.error(child)),
            }
        }

        Ok(Call {
            func: expr.ok_or_else(|| self.error(node))?,
            args: args.ok_or_else(|| self.error(node))?,
        })
    }

    fn parse_compop(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<CompOp> {
        let cursor = &mut node.walk();
        let mut var = None;
        let mut op = None;
        let mut expr = None;

        enum State {
            Var,
            Op,
            Expr,
        }

        let mut state = State::Var;

        for child in node.children(cursor) {
            let kind = child.kind();
            match (kind, &state) {
                ("var", State::Var) => {
                    var = Some(self.parse_var(child, unp)?);
                    state = State::Op;
                }
                (_, State::Op) => {
                    let txt = self.extract_text(child);
                    op = Some(CompOpKind::from_str(txt).map_err(|_| self.error(child))?);
                    state = State::Expr;
                }
                (_, State::Expr) => expr = Some(self.parse_expr(child, unp)?),
                _ => todo!("parse_compop: {}", kind),
            }
        }

        Ok(CompOp {
            lhs: var.ok_or_else(|| self.error(node))?,
            op: op.ok_or_else(|| self.error(node))?,
            rhs: Box::new(expr.ok_or_else(|| self.error(node))?),
        })
    }

    fn parse_return(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<Return> {
        let cursor = &mut node.walk();
        let mut exprs = Vec::new();
        for child in node.children(cursor) {
            let kind = child.kind();
            match kind {
                "return" => {}
                _ => exprs.push(self.parse_expr(child, unp)?),
            }
        }
        Ok(Return { exprs })
    }

    fn parse_assign(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<Assign> {
        let cursor = &mut node.walk();
        let mut vars = Vec::new();
        let mut exprs = Vec::new();

        enum State {
            Vars,
            Exprs,
        }

        let mut state = State::Vars;

        for child in node.children(cursor) {
            let kind = child.kind();
            match (kind, &state) {
                ("var", State::Vars) => vars.push(self.parse_var(child, unp)?),
                ("=", State::Vars) => state = State::Exprs,
                (",", _) => {}
                (_, State::Exprs) => exprs.push(self.parse_expr(child, unp)?),
                _ => return Err(self.error(child)),
            }
        }

        Ok(Assign { vars, exprs })
    }

    fn parse_binding(&self, node: tree_sitter::Node<'ts>) -> Result<Binding> {
        let cursor = &mut node.walk();
        let mut name = String::new();
        let mut parsing_type = false;
        for child in node.children(cursor) {
            let kind = child.kind();
            match kind {
                "name" | "vararg" => name.push_str(self.extract_text(child)),
                ":" => {
                    parsing_type = true;
                }
                _ if parsing_type => {
                    todo!("parse_binding (type delegation): {}", kind);
                }
                _ => todo!("parse_binding: {}", kind),
            }
        }
        Ok(Binding { name, ty: None })
    }

    fn parse_expr(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<Expr> {
        let kind = node.kind();
        match kind {
            "nil" => Ok(Expr::Nil),
            "number" => {
                let text = self.extract_text(node);
                let num = text.parse().map_err(|_| self.error(node))?;
                Ok(Expr::Number(num))
            }
            "string" => {
                let text = self.extract_text(node);
                Ok(Expr::String(text.to_string()))
            }
            "boolean" => {
                // true is 4 bytes, false is 5 bytes. hack! no need for text :^)
                let start = node.start_byte();
                let end = node.end_byte();

                // lets just make sure lol (only in debug mode)
                debug_assert!(matches!(&self.text[start..end], "true" | "false"));

                Ok(Expr::Bool(end - start == 4))
            }
            "anon_fn" => Ok(Expr::Function(Box::new(
                self.parse_fn_body(node.child(1).ok_or_else(|| self.error(node))?, unp)?,
            ))),
            "vararg" => Ok(Expr::VarArg),
            "exp_wrap" => Ok(Expr::Wrap(Box::new(
                self.parse_expr(node.child(1).ok_or_else(|| self.error(node))?, unp)?,
            ))),
            // delegate to parse_var
            "var" => Ok(Expr::Var(self.parse_var(node, unp)?)),
            // delegate to parse_binop
            "binexp" => Ok(Expr::BinOp(Box::new(self.parse_binop(node, unp)?))),
            // delegate to parse_unop
            "unexp" => Ok(Expr::UnOp(Box::new(self.parse_unop(node, unp)?))),
            // delegate to parse_call
            "call_stmt" => Ok(Expr::Call(Box::new(self.parse_call(node, unp)?))),
            // delegate to parse_tableconstructor
            "table" => Ok(Expr::TableConstructor(
                self.parse_tableconstructor(node, unp)?,
            )),
            // delegate to parse_ifexpr
            "ifexp" => Ok(Expr::IfElse(Box::new(self.parse_ifelseexp(node, unp)?))),
            _ => todo!("parse_expr: {}", kind),
        }
    }

    fn parse_var(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<Var> {
        let cursor = &mut node.walk();

        #[derive(Debug)] // TODO: remove me
        enum State<'ts> {
            Init,
            TableExpr(tree_sitter::Node<'ts>),
            FieldExpr(tree_sitter::Node<'ts>),
            // NOTE: we are doing this because in the grammar "var" is "name" due to a cycle
            TableExprWasVar,
            FieldExprWasVar,
        }

        let mut var = None;

        let mut state = State::Init;

        for child in node.children(cursor) {
            let kind = child.kind();
            match (kind, &state) {
                ("name", State::Init) => {
                    var = Some(Var::Name(self.extract_text(child).to_string()))
                }
                ("[", State::Init) if var.is_some() => state = State::TableExprWasVar,
                (".", State::Init) if var.is_some() => state = State::FieldExprWasVar,
                ("[", State::Init | State::TableExpr(_)) => {}
                (".", State::TableExpr(node)) => state = State::FieldExpr(*node),
                ("]", _) => {}
                (_, State::TableExprWasVar) => {
                    var = Some(Var::TableAccess(Box::new(TableAccess {
                        expr: Expr::Var(var.take().unwrap()),
                        index: self.parse_expr(child, unp)?,
                    })));
                }
                (_, State::FieldExprWasVar) => {
                    var = Some(Var::FieldAccess(Box::new(FieldAccess {
                        expr: Expr::Var(var.take().unwrap()),
                        field: self.extract_text(child).to_string(),
                    })));
                }
                (_, State::TableExpr(expr_node)) => {
                    var = Some(Var::TableAccess(Box::new(TableAccess {
                        expr: self.parse_expr(*expr_node, unp)?,
                        index: self.parse_expr(child, unp)?,
                    })));
                }
                (_, State::FieldExpr(expr_node)) => {
                    var = Some(Var::FieldAccess(Box::new(FieldAccess {
                        expr: self.parse_expr(*expr_node, unp)?,
                        field: self.extract_text(child).to_string(),
                    })));
                }
                (_, State::Init) => state = State::TableExpr(child),
            }
        }

        var.ok_or_else(|| self.error(node))
    }

    fn parse_binop(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<BinOp> {
        let cursor = &mut node.walk();
        let mut op = None;
        let mut lhs = None;
        let mut rhs = None;

        enum State {
            Lhs,
            Op,
            Rhs,
        }

        let mut state = State::Lhs;
        for child in node.children(cursor) {
            match state {
                State::Lhs => {
                    lhs = Some(self.parse_expr(child, unp)?);
                    state = State::Op;
                }
                State::Op => {
                    let txt = self.extract_text(child);
                    let parsed_op = BinOpKind::from_str(txt).map_err(|_| self.error(node))?;
                    op = Some(parsed_op);
                    state = State::Rhs;
                }
                State::Rhs => {
                    rhs = Some(self.parse_expr(child, unp)?);
                }
            }
        }
        Ok(BinOp {
            lhs: lhs.ok_or_else(|| self.error(node))?,
            op: op.ok_or(self.error(node))?,
            rhs: rhs.ok_or_else(|| self.error(node))?,
        })
    }

    fn parse_unop(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<UnOp> {
        let cursor = &mut node.walk();
        let mut op = None;
        let mut expr = None;

        enum State {
            Op,
            Expr,
        }

        let mut state = State::Op;
        for child in node.children(cursor) {
            match state {
                State::Op => {
                    let txt = self.extract_text(child);
                    let parsed_op = UnOpKind::from_str(txt).map_err(|_| self.error(node))?;
                    op = Some(parsed_op);
                    state = State::Expr;
                }
                State::Expr => {
                    expr = Some(self.parse_expr(child, unp)?);
                }
            }
        }
        Ok(UnOp {
            op: op.ok_or_else(|| self.error(node))?,
            expr: expr.ok_or_else(|| self.error(node))?,
        })
    }

    fn parse_tableconstructor(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<TableConstructor> {
        let cursor = &mut node.walk();
        let mut fields = Vec::new();

        enum FieldState<'s, 'ts> {
            Init,
            ParsingExplicit(&'s str),
            ParsingArrayNext,
            ParsingArray(tree_sitter::Node<'ts>),
        }

        let mut parse_field = |field: tree_sitter::Node<'ts>| -> Result<TableField> {
            let cursor = &mut field.walk();

            let mut state = FieldState::Init;

            for child in field.children(cursor) {
                let kind = child.kind();
                match (kind, &state) {
                    ("name", FieldState::Init) => {
                        let name = self.extract_text(child);
                        state = FieldState::ParsingExplicit(name);
                    }
                    ("=", FieldState::ParsingExplicit(_) | FieldState::ParsingArray(_)) => {}
                    (_, FieldState::ParsingExplicit(name)) => {
                        let expr = self.parse_expr(child, unp)?;
                        return Ok(TableField::ExplicitKey {
                            key: name.to_string(),
                            value: expr,
                        });
                    }
                    ("[", FieldState::Init) => {
                        state = FieldState::ParsingArrayNext;
                    }
                    (_, FieldState::ParsingArrayNext) => {
                        state = FieldState::ParsingArray(child);
                    }
                    ("]", FieldState::ParsingArray(_)) => {}
                    (_, FieldState::ParsingArray(key_node)) => {
                        let key = self.parse_expr(*key_node, unp)?;
                        let value = self.parse_expr(child, unp)?;
                        return Ok(TableField::ArrayKey { key, value });
                    }
                    (_, FieldState::Init) => {
                        let expr = self.parse_expr(child, unp)?;
                        return Ok(TableField::ImplicitKey(expr));
                    }
                }
            }
            Err(self.error(node))
        };

        for child in node.children(cursor) {
            let kind = child.kind();
            match kind {
                "}" | "{" => {}
                "fieldlist" => {
                    let cursor = &mut child.walk();
                    for field in child.children(cursor) {
                        let kind = field.kind();
                        match kind {
                            "field" => fields.push(parse_field(field)?),
                            "," | ";" => {}
                            _ => return Err(self.error(node)),
                        }
                    }
                }
                _ => return Err(self.error(node)),
            }
        }
        Ok(TableConstructor { fields })
    }

    fn parse_ifelseexp(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<IfElseExp> {
        let cursor = &mut node.walk();
        let mut cond = None;
        let mut if_exp = None;
        let mut else_if_exprs = Vec::new();
        let mut else_expr = None;

        enum State<'ts> {
            Condition,
            Body,
            ElseBody,
            ElseIfConditionNext,
            ElseIfCondition(tree_sitter::Node<'ts>),
        }

        let mut state = State::Condition;

        for child in node.children(cursor) {
            let kind = child.kind();
            match (kind, &state) {
                ("if", State::Condition) => {}
                ("then", State::Body | State::ElseBody | State::ElseIfCondition(_)) => {}
                (_, State::Condition) => {
                    cond = Some(self.parse_expr(child, unp)?);
                    state = State::Body;
                }
                (_, State::Body) => {
                    if_exp = Some(self.parse_expr(child, unp)?);
                    state = State::ElseBody;
                }
                ("else", State::ElseBody) => {}
                ("elseif", State::ElseBody) => {
                    state = State::ElseIfConditionNext;
                }
                (_, State::ElseBody) => {
                    else_expr = Some(self.parse_expr(child, unp)?);
                }
                (_, State::ElseIfConditionNext) => {
                    state = State::ElseIfCondition(child);
                }
                (_, State::ElseIfCondition(cond_node)) => {
                    let cond = self.parse_expr(*cond_node, unp)?;
                    let body = self.parse_expr(child, unp)?;
                    else_if_exprs.push((cond, body));
                    state = State::ElseBody;
                }
            }
        }
        Ok(IfElseExp {
            cond: cond.ok_or_else(|| self.error(node))?,
            if_expr: if_exp.ok_or_else(|| self.error(node))?,
            else_expr: else_expr.ok_or_else(|| self.error(node))?,
            else_if_exprs,
        })
    }
}

impl FromStr for BinOpKind {
    type Err = ();

    /// binop = '+' | '-' | '*' | '/' | '^' | '%' | '..' | '<' | '<=' | '>' | '>=' | '==' | '~=' | 'and' | 'or'
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "+" => Ok(BinOpKind::Add),
            "-" => Ok(BinOpKind::Sub),
            "*" => Ok(BinOpKind::Mul),
            "/" => Ok(BinOpKind::Div),
            "^" => Ok(BinOpKind::Pow),
            "%" => Ok(BinOpKind::Mod),
            ".." => Ok(BinOpKind::Concat),
            "<" => Ok(BinOpKind::Lt),
            "<=" => Ok(BinOpKind::Le),
            ">" => Ok(BinOpKind::Gt),
            ">=" => Ok(BinOpKind::Ge),
            "==" => Ok(BinOpKind::Eq),
            "~=" => Ok(BinOpKind::Ne),
            "and" => Ok(BinOpKind::And),
            "or" => Ok(BinOpKind::Or),
            _ => Err(()),
        }
    }
}

impl FromStr for UnOpKind {
    type Err = ();

    /// unop = '-' | 'not' | '#'
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "-" => Ok(UnOpKind::Neg),
            "#" => Ok(UnOpKind::Len),
            "not" => Ok(UnOpKind::Not),
            _ => Err(()),
        }
    }
}

impl FromStr for CompOpKind {
    type Err = ();

    /// compoundop :: '+=' | '-=' | '*=' | '/=' | '%=' | '^=' | '..='
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "+=" => Ok(CompOpKind::Add),
            "-=" => Ok(CompOpKind::Sub),
            "*=" => Ok(CompOpKind::Mul),
            "/=" => Ok(CompOpKind::Div),
            "%=" => Ok(CompOpKind::Mod),
            "^=" => Ok(CompOpKind::Pow),
            "..=" => Ok(CompOpKind::Concat),
            _ => Err(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use crate::parser::*;

    macro_rules! assert_parse {
        ($source:expr, $expected:expr) => {
            let ast = Parser::new($source).parse();
            assert_eq!(ast.unwrap(), $expected);
        };
    }

    #[test]
    fn parse_locals() {
        assert_parse!(
            "local a = 1\nlocal b, c = 2, 3",
            Chunk {
                block: Block {
                    stmt_ptrs: vec![0, 1]
                },
                stmts: vec![
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "a".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::Number(1.0)]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![
                            Binding {
                                name: "b".to_string(),
                                ty: None
                            },
                            Binding {
                                name: "c".to_string(),
                                ty: None
                            }
                        ],
                        init: vec![Expr::Number(2.0), Expr::Number(3.0)]
                    }))
                ]
            }
        );
    }

    #[test]
    fn parse_local_empty() {
        assert_parse!(
            "local a",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(Stmt::Local(Local {
                    bindings: vec![Binding {
                        name: "a".to_string(),
                        ty: None
                    }],
                    init: vec![]
                }))]
            }
        );
    }

    #[test]
    fn parse_string_double() {
        assert_parse!(
            "local a = \"hello\"",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(Stmt::Local(Local {
                    bindings: vec![Binding {
                        name: "a".to_string(),
                        ty: None
                    }],
                    init: vec![Expr::String("\"hello\"".to_string())]
                }))],
            }
        );
    }

    #[test]
    fn parse_string_single() {
        assert_parse!(
            "local a = 'hello'",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(Stmt::Local(Local {
                    bindings: vec![Binding {
                        name: "a".to_string(),
                        ty: None
                    }],
                    init: vec![Expr::String("'hello'".to_string())]
                }))],
            }
        );
    }

    #[test]
    fn parse_string_single_double() {
        assert_parse!(
            "local a = '\"hello\"'",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(Stmt::Local(Local {
                    bindings: vec![Binding {
                        name: "a".to_string(),
                        ty: None
                    }],
                    init: vec![Expr::String("'\"hello\"'".to_string())]
                }))],
            }
        );
    }

    #[test]
    fn parse_string_double_single() {
        assert_parse!(
            "local a = \"'hello'\"",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(Stmt::Local(Local {
                    bindings: vec![Binding {
                        name: "a".to_string(),
                        ty: None
                    }],
                    init: vec![Expr::String("\"'hello'\"".to_string())]
                }))],
            }
        );
    }

    #[test]
    fn parse_string_square() {
        assert_parse!(
            "local a = [[hello]]",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(Stmt::Local(Local {
                    bindings: vec![Binding {
                        name: "a".to_string(),
                        ty: None
                    }],
                    init: vec![Expr::String("[[hello]]".to_string())]
                }))],
            }
        );
    }

    #[test]
    fn parse_string_square_double() {
        assert_parse!(
            "local a = [[\"hello\"]]",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(Stmt::Local(Local {
                    bindings: vec![Binding {
                        name: "a".to_string(),
                        ty: None
                    }],
                    init: vec![Expr::String("[[\"hello\"]]".to_string())]
                }))],
            }
        );
    }

    #[test]
    fn parse_string_square_single() {
        assert_parse!(
            "local a = [['hello']]",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(Stmt::Local(Local {
                    bindings: vec![Binding {
                        name: "a".to_string(),
                        ty: None
                    }],
                    init: vec![Expr::String("[['hello']]".to_string())]
                }))],
            }
        );
    }

    #[test]
    fn parse_nil() {
        assert_parse!(
            "local a = nil",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(Stmt::Local(Local {
                    bindings: vec![Binding {
                        name: "a".to_string(),
                        ty: None
                    }],
                    init: vec![Expr::Nil]
                }))],
            }
        );
    }

    #[test]
    fn parse_bools() {
        assert_parse!(
            "local a = true\nlocal b = false",
            Chunk {
                block: Block {
                    stmt_ptrs: vec![0, 1]
                },
                stmts: vec![
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "a".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::Bool(true)]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "b".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::Bool(false)]
                    }))
                ]
            }
        );
    }

    #[test]
    fn parse_var() {
        assert_parse!(
            "local a = 1\nlocal b = a",
            Chunk {
                block: Block {
                    stmt_ptrs: vec![0, 1]
                },
                stmts: vec![
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "a".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::Number(1.0)]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "b".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::Var(Var::Name("a".to_string()))]
                    }))
                ]
            }
        );
    }

    #[test]
    fn parse_fn_and_call() {
        assert_parse!(
            "function f(n)\nprint(n)\nend\nf(3)",
            Chunk {
                block: Block {
                    stmt_ptrs: vec![
                        0, // function f(n) print(n) end
                        1, // f(3)
                    ],
                },
                stmts: vec![
                    StmtStatus::Some(Stmt::FunctionDef(FunctionDef {
                        name: "f".to_string(),
                        body: FunctionBody {
                            ret_ty: None,
                            block: Block { stmt_ptrs: vec![2] },
                            generics: vec![],
                            params: vec![Binding {
                                name: "n".to_string(),
                                ty: None
                            }],
                        },
                        table: vec![],
                        is_method: false
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("f".to_string())),
                        args: CallArgs::Exprs(vec![Expr::Number(3.0)])
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: CallArgs::Exprs(vec![Expr::Var(Var::Name("n".to_string()))])
                    }))
                ]
            }
        );
    }

    #[test]
    fn parse_fn_and_call_in_local() {
        assert_parse!(
            "function f(n)\nprint(n)\nend\nlocal res = f(3)",
            Chunk {
                block: Block {
                    stmt_ptrs: vec![
                        0, // function f(n) print(n) end
                        1, // local res = f(3)
                    ],
                },
                stmts: vec![
                    StmtStatus::Some(Stmt::FunctionDef(FunctionDef {
                        name: "f".to_string(),
                        body: FunctionBody {
                            ret_ty: None,
                            block: Block { stmt_ptrs: vec![2] },
                            generics: vec![],
                            params: vec![Binding {
                                name: "n".to_string(),
                                ty: None
                            }],
                        },
                        table: vec![],
                        is_method: false
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "res".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::Call(Box::new(Call {
                            func: Expr::Var(Var::Name("f".to_string())),
                            args: CallArgs::Exprs(vec![Expr::Number(3.0)])
                        }))]
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: CallArgs::Exprs(vec![Expr::Var(Var::Name("n".to_string()))])
                    }))
                ]
            }
        );
    }

    #[test]
    fn parse_fn_and_call_in_local_multi() {
        assert_parse!(
            "function f(n)\nprint(n)\nend\nlocal res1, res2 = f(3), f(4)",
            Chunk {
                block: Block {
                    stmt_ptrs: vec![
                        0, // function f(n) print(n) end
                        1, // local res = f(3)
                    ],
                },
                stmts: vec![
                    StmtStatus::Some(Stmt::FunctionDef(FunctionDef {
                        name: "f".to_string(),
                        body: FunctionBody {
                            ret_ty: None,
                            block: Block { stmt_ptrs: vec![2] },
                            generics: vec![],
                            params: vec![Binding {
                                name: "n".to_string(),
                                ty: None
                            }],
                        },
                        table: vec![],
                        is_method: false
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![
                            Binding {
                                name: "res1".to_string(),
                                ty: None
                            },
                            Binding {
                                name: "res2".to_string(),
                                ty: None
                            }
                        ],
                        init: vec![
                            Expr::Call(Box::new(Call {
                                func: Expr::Var(Var::Name("f".to_string())),
                                args: CallArgs::Exprs(vec![Expr::Number(3.0)])
                            })),
                            Expr::Call(Box::new(Call {
                                func: Expr::Var(Var::Name("f".to_string())),
                                args: CallArgs::Exprs(vec![Expr::Number(4.0)])
                            }))
                        ]
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: CallArgs::Exprs(vec![Expr::Var(Var::Name("n".to_string()))])
                    }))
                ]
            }
        );
    }

    #[test]
    fn parse_fn_multiarg() {
        assert_parse!(
            "function f(a, b)\nprint(a)\nprint(b)\nend\nf(3, 4)",
            Chunk {
                block: Block {
                    stmt_ptrs: vec![
                        0, // function f(a, b) print(a) print(b) end
                        1, // f(3, 4)
                    ],
                },
                stmts: vec![
                    StmtStatus::Some(Stmt::FunctionDef(FunctionDef {
                        name: "f".to_string(),
                        body: FunctionBody {
                            ret_ty: None,
                            block: Block {
                                stmt_ptrs: vec![2, 3]
                            },
                            generics: vec![],
                            params: vec![
                                Binding {
                                    name: "a".to_string(),
                                    ty: None
                                },
                                Binding {
                                    name: "b".to_string(),
                                    ty: None
                                }
                            ],
                        },
                        table: vec![],
                        is_method: false
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("f".to_string())),
                        args: CallArgs::Exprs(vec![Expr::Number(3.0), Expr::Number(4.0)])
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: CallArgs::Exprs(vec![Expr::Var(Var::Name("a".to_string()))])
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: CallArgs::Exprs(vec![Expr::Var(Var::Name("b".to_string()))])
                    }))
                ]
            }
        );
    }

    #[test]
    fn parse_table_fn() {
        assert_parse!(
            "function t.f(n)\nprint(n)\nend",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![
                    StmtStatus::Some(Stmt::FunctionDef(FunctionDef {
                        name: "f".to_string(),
                        body: FunctionBody {
                            generics: vec![],
                            ret_ty: None,
                            block: Block { stmt_ptrs: vec![1] },
                            params: vec![Binding {
                                name: "n".to_string(),
                                ty: None
                            }],
                        },
                        table: vec!["t".to_string()],
                        is_method: false
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: CallArgs::Exprs(vec![Expr::Var(Var::Name("n".to_string()))])
                    }))
                ]
            }
        );
    }

    #[test]
    fn parse_table_fn_nested() {
        assert_parse!(
            "function t.a.f(n)\nprint(n)\nend",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![
                    StmtStatus::Some(Stmt::FunctionDef(FunctionDef {
                        name: "f".to_string(),
                        body: FunctionBody {
                            generics: vec![],
                            ret_ty: None,
                            block: Block { stmt_ptrs: vec![1] },
                            params: vec![Binding {
                                name: "n".to_string(),
                                ty: None
                            }],
                        },
                        table: vec!["t", "a"].iter().map(|s| s.to_string()).collect(),
                        is_method: false
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: CallArgs::Exprs(vec![Expr::Var(Var::Name("n".to_string()))])
                    }))
                ]
            }
        );
    }

    #[test]
    fn parse_table_fn_nested2() {
        assert_parse!(
            "function t.a.b.c.f(n)\nprint(n)\nend",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![
                    StmtStatus::Some(Stmt::FunctionDef(FunctionDef {
                        name: "f".to_string(),
                        body: FunctionBody {
                            generics: vec![],
                            ret_ty: None,
                            block: Block { stmt_ptrs: vec![1] },
                            params: vec![Binding {
                                name: "n".to_string(),
                                ty: None
                            }],
                        },
                        table: vec!["t", "a", "b", "c"]
                            .iter()
                            .map(|s| s.to_string())
                            .collect(),
                        is_method: false
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: CallArgs::Exprs(vec![Expr::Var(Var::Name("n".to_string()))])
                    }))
                ]
            }
        );
    }

    #[test]
    fn parse_method_fn() {
        assert_parse!(
            "function t:f(n)\nprint(self)\nend",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![
                    StmtStatus::Some(Stmt::FunctionDef(FunctionDef {
                        name: "f".to_string(),
                        body: FunctionBody {
                            ret_ty: None,
                            generics: vec![],
                            block: Block { stmt_ptrs: vec![1] },
                            params: vec![Binding {
                                name: "n".to_string(),
                                ty: None
                            }],
                        },
                        table: vec!["t".to_string()],
                        is_method: true
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: CallArgs::Exprs(vec![Expr::Var(Var::Name("self".to_string()))])
                    }))
                ]
            }
        );
    }

    #[test]
    fn parse_local_fn_and_call() {
        assert_parse!(
            "local function f(n)\nprint(n)\nend\nf(3)",
            Chunk {
                block: Block {
                    stmt_ptrs: vec![
                        0, // function f(n) print(n) end
                        1, // f(3)
                    ],
                },
                stmts: vec![
                    StmtStatus::Some(Stmt::LocalFunctionDef(LocalFunctionDef {
                        name: "f".to_string(),
                        body: FunctionBody {
                            ret_ty: None,
                            generics: vec![],
                            block: Block {
                                stmt_ptrs: vec![2], // print(n)
                            },
                            params: vec![Binding {
                                name: "n".to_string(),
                                ty: None
                            }],
                        }
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("f".to_string())),
                        args: CallArgs::Exprs(vec![Expr::Number(3.0)])
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: CallArgs::Exprs(vec![Expr::Var(Var::Name("n".to_string()))])
                    }))
                ]
            }
        );
    }

    #[test]
    fn parse_fn_generics() {
        assert_parse!(
            "function f<T, U...>(n) end",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(Stmt::FunctionDef(FunctionDef {
                    name: "f".to_string(),
                    body: FunctionBody {
                        ret_ty: None,
                        generics: vec![
                            GenericParam::Name("T".to_string()),
                            GenericParam::Pack("U".to_string())
                        ],
                        block: Block { stmt_ptrs: vec![] },
                        params: vec![Binding {
                            name: "n".to_string(),
                            ty: None
                        }],
                    },
                    table: vec![],
                    is_method: false
                })),]
            }
        );
    }

    #[test]
    fn parse_fn_empty_body() {
        assert_parse!(
            "function f() end",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(Stmt::FunctionDef(FunctionDef {
                    name: "f".to_string(),
                    body: FunctionBody {
                        ret_ty: None,
                        generics: vec![],
                        block: Block { stmt_ptrs: vec![] },
                        params: vec![]
                    },
                    table: vec![],
                    is_method: false
                }))]
            }
        );
    }

    #[test]
    fn all_binops() {
        assert_parse!(
            r#"
            local plus = 1 + 2
            local minus = 1 - 2
            local mul = 1 * 2
            local div = 1 / 2
            local mod = 1 % 2
            local pow = 1 ^ 2
            local concat = "a" .. "b"
            local eq = 1 == 2
            local neq = 1 ~= 2
            local lt = 1 < 2
            local gt = 1 > 2
            local leq = 1 <= 2
            local geq = 1 >= 2
            local and = true and false
            local or = true or false
            "#,
            Chunk {
                block: Block {
                    stmt_ptrs: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
                },
                stmts: vec![
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "plus".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::BinOp(Box::new(BinOp {
                            lhs: Expr::Number(1.0),
                            op: BinOpKind::Add,
                            rhs: Expr::Number(2.0)
                        })),]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "minus".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::BinOp(Box::new(BinOp {
                            lhs: Expr::Number(1.0),
                            op: BinOpKind::Sub,
                            rhs: Expr::Number(2.0)
                        })),]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "mul".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::BinOp(Box::new(BinOp {
                            lhs: Expr::Number(1.0),
                            op: BinOpKind::Mul,
                            rhs: Expr::Number(2.0)
                        })),]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "div".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::BinOp(Box::new(BinOp {
                            lhs: Expr::Number(1.0),
                            op: BinOpKind::Div,
                            rhs: Expr::Number(2.0)
                        })),]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "mod".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::BinOp(Box::new(BinOp {
                            lhs: Expr::Number(1.0),
                            op: BinOpKind::Mod,
                            rhs: Expr::Number(2.0)
                        })),]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "pow".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::BinOp(Box::new(BinOp {
                            lhs: Expr::Number(1.0),
                            op: BinOpKind::Pow,
                            rhs: Expr::Number(2.0)
                        })),]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "concat".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::BinOp(Box::new(BinOp {
                            lhs: Expr::String("\"a\"".to_string()),
                            op: BinOpKind::Concat,
                            rhs: Expr::String("\"b\"".to_string())
                        })),]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "eq".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::BinOp(Box::new(BinOp {
                            lhs: Expr::Number(1.0),
                            op: BinOpKind::Eq,
                            rhs: Expr::Number(2.0)
                        })),]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "neq".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::BinOp(Box::new(BinOp {
                            lhs: Expr::Number(1.0),
                            op: BinOpKind::Ne,
                            rhs: Expr::Number(2.0)
                        })),]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "lt".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::BinOp(Box::new(BinOp {
                            lhs: Expr::Number(1.0),
                            op: BinOpKind::Lt,
                            rhs: Expr::Number(2.0)
                        })),]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "gt".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::BinOp(Box::new(BinOp {
                            lhs: Expr::Number(1.0),
                            op: BinOpKind::Gt,
                            rhs: Expr::Number(2.0)
                        })),]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "leq".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::BinOp(Box::new(BinOp {
                            lhs: Expr::Number(1.0),
                            op: BinOpKind::Le,
                            rhs: Expr::Number(2.0)
                        })),]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "geq".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::BinOp(Box::new(BinOp {
                            lhs: Expr::Number(1.0),
                            op: BinOpKind::Ge,
                            rhs: Expr::Number(2.0)
                        })),]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "and".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::BinOp(Box::new(BinOp {
                            lhs: Expr::Bool(true),
                            op: BinOpKind::And,
                            rhs: Expr::Bool(false)
                        })),]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "or".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::BinOp(Box::new(BinOp {
                            lhs: Expr::Bool(true),
                            op: BinOpKind::Or,
                            rhs: Expr::Bool(false)
                        })),]
                    })),
                ]
            }
        );
    }

    #[test]
    fn all_unops() {
        assert_parse!(
            r#"
            local not = not true
            local neg = - 1
            local neg_wrap = -(1)
            local len = #"aaaaa"
            "#,
            Chunk {
                block: Block {
                    stmt_ptrs: vec![0, 1, 2, 3],
                },
                stmts: vec![
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "not".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::UnOp(Box::new(UnOp {
                            op: UnOpKind::Not,
                            expr: Expr::Bool(true)
                        })),]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "neg".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::UnOp(Box::new(UnOp {
                            op: UnOpKind::Neg,
                            expr: Expr::Number(1.0)
                        })),]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "neg_wrap".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::UnOp(Box::new(UnOp {
                            op: UnOpKind::Neg,
                            expr: Expr::Wrap(Box::new(Expr::Number(1.0)))
                        })),]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "len".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::UnOp(Box::new(UnOp {
                            op: UnOpKind::Len,
                            expr: Expr::String("\"aaaaa\"".to_string())
                        })),]
                    })),
                ]
            }
        );
    }

    #[test]
    fn all_compops() {
        assert_parse!(
            r#"
            local guinea_pig = 1
            guinea_pig += 1
            guinea_pig -= 1
            guinea_pig *= 1
            guinea_pig /= 1
            guinea_pig %= 1
            guinea_pig ^= 1
            guinea_pig ..= 1
            "#,
            Chunk {
                block: Block {
                    stmt_ptrs: vec![0, 1, 2, 3, 4, 5, 6, 7],
                },
                stmts: vec![
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "guinea_pig".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::Number(1.0)]
                    })),
                    StmtStatus::Some(Stmt::CompOp(CompOp {
                        lhs: Var::Name("guinea_pig".to_string()),
                        op: CompOpKind::Add,
                        rhs: Box::new(Expr::Number(1.0))
                    })),
                    StmtStatus::Some(Stmt::CompOp(CompOp {
                        lhs: Var::Name("guinea_pig".to_string()),
                        op: CompOpKind::Sub,
                        rhs: Box::new(Expr::Number(1.0))
                    })),
                    StmtStatus::Some(Stmt::CompOp(CompOp {
                        lhs: Var::Name("guinea_pig".to_string()),
                        op: CompOpKind::Mul,
                        rhs: Box::new(Expr::Number(1.0))
                    })),
                    StmtStatus::Some(Stmt::CompOp(CompOp {
                        lhs: Var::Name("guinea_pig".to_string()),
                        op: CompOpKind::Div,
                        rhs: Box::new(Expr::Number(1.0))
                    })),
                    StmtStatus::Some(Stmt::CompOp(CompOp {
                        lhs: Var::Name("guinea_pig".to_string()),
                        op: CompOpKind::Mod,
                        rhs: Box::new(Expr::Number(1.0))
                    })),
                    StmtStatus::Some(Stmt::CompOp(CompOp {
                        lhs: Var::Name("guinea_pig".to_string()),
                        op: CompOpKind::Pow,
                        rhs: Box::new(Expr::Number(1.0))
                    })),
                    StmtStatus::Some(Stmt::CompOp(CompOp {
                        lhs: Var::Name("guinea_pig".to_string()),
                        op: CompOpKind::Concat,
                        rhs: Box::new(Expr::Number(1.0))
                    })),
                ]
            }
        );
    }

    #[test]
    fn test_wrap() {
        assert_parse!(
            r#"
        local cray_cray = (1 + (2 * 3) - (1)) / 4
        "#,
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(Stmt::Local(Local {
                    bindings: vec![Binding {
                        name: "cray_cray".to_string(),
                        ty: None
                    }],
                    init: vec![Expr::BinOp(Box::new(BinOp {
                        lhs: Expr::Wrap(Box::new(Expr::BinOp(Box::new(BinOp {
                            lhs: Expr::BinOp(Box::new(BinOp {
                                lhs: Expr::Number(1.0),
                                op: BinOpKind::Add,
                                rhs: Expr::Wrap(Box::new(Expr::BinOp(Box::new(BinOp {
                                    lhs: Expr::Number(2.0),
                                    op: BinOpKind::Mul,
                                    rhs: Expr::Number(3.0)
                                })))),
                            })),
                            op: BinOpKind::Sub,
                            rhs: Expr::Wrap(Box::new(Expr::Number(1.0)))
                        })))),
                        op: BinOpKind::Div,
                        rhs: Expr::Number(4.0)
                    })),]
                })),]
            }
        );
    }

    #[test]
    fn test_return_break_continue() {
        assert_parse!(
            "function a(n)\ncontinue\nbreak\nreturn\nend",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![
                    StmtStatus::Some(Stmt::FunctionDef(FunctionDef {
                        name: "a".to_string(),
                        table: vec![],
                        is_method: false,
                        body: FunctionBody {
                            ret_ty: None,
                            generics: vec![],
                            block: Block {
                                stmt_ptrs: vec![1, 2, 3]
                            },
                            params: vec![Binding {
                                name: "n".to_string(),
                                ty: None
                            }],
                        },
                    })),
                    StmtStatus::Some(Stmt::Continue(Continue {})),
                    StmtStatus::Some(Stmt::Break(Break {})),
                    StmtStatus::Some(Stmt::Return(Return { exprs: vec![] })),
                ]
            }
        );
    }

    #[test]
    fn test_varargs() {
        assert_parse!(
            "function a(...)\nend\nlocal e = ...\n",
            Chunk {
                block: Block {
                    stmt_ptrs: vec![0, 1]
                },
                stmts: vec![
                    StmtStatus::Some(Stmt::FunctionDef(FunctionDef {
                        name: "a".to_string(),
                        table: vec![],
                        is_method: false,
                        body: FunctionBody {
                            ret_ty: None,
                            generics: vec![],
                            block: Block { stmt_ptrs: vec![] },
                            params: vec![Binding {
                                name: "...".to_string(),
                                ty: None
                            }],
                        },
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "e".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::VarArg]
                    }))
                ],
            }
        );
    }

    #[test]
    fn test_assignment() {
        assert_parse!(
            "local a = 1\na = false\na, b = 1, 2",
            Chunk {
                block: Block {
                    stmt_ptrs: vec![0, 1, 2],
                },
                stmts: vec![
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "a".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::Number(1.0)]
                    })),
                    StmtStatus::Some(Stmt::Assign(Assign {
                        vars: vec![Var::Name("a".to_string())],
                        exprs: vec![Expr::Bool(false)]
                    })),
                    StmtStatus::Some(Stmt::Assign(Assign {
                        vars: vec![Var::Name("a".to_string()), Var::Name("b".to_string())],
                        exprs: vec![Expr::Number(1.0), Expr::Number(2.0)]
                    })),
                ]
            }
        );
    }

    #[test]
    fn test_do_end() {
        assert_parse!(
            "do\nprint(1)\nend",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![
                    StmtStatus::Some(Stmt::Do(Do {
                        block: Block { stmt_ptrs: vec![1] }
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: CallArgs::Exprs(vec![Expr::Number(1.0)])
                    }))
                ]
            }
        );
    }

    #[test]
    fn test_while() {
        assert_parse!(
            "while true do\nprint(1)\nend",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![
                    StmtStatus::Some(Stmt::While(While {
                        cond: Expr::Bool(true),
                        block: Block { stmt_ptrs: vec![1] }
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: CallArgs::Exprs(vec![Expr::Number(1.0)])
                    }))
                ]
            }
        );
    }

    #[test]
    fn test_repeat() {
        assert_parse!(
            "repeat\nprint(1)\nuntil true",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![
                    StmtStatus::Some(Stmt::Repeat(Repeat {
                        block: Block { stmt_ptrs: vec![1] },
                        cond: Expr::Bool(true)
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: CallArgs::Exprs(vec![Expr::Number(1.0)])
                    }))
                ]
            }
        );
    }

    // if tests:
    // 1. if x then
    // 2. if x then else
    // 3. if x then elseif y then
    // 4. if x then elseif y then else
    // 5. if x then elseif y then elseif z then

    #[test]
    fn test_if() {
        assert_parse!(
            "if true then\nprint(1)\nend",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![
                    StmtStatus::Some(Stmt::If(If {
                        cond: Expr::Bool(true),
                        block: Block { stmt_ptrs: vec![1] },
                        else_if_blocks: vec![],
                        else_block: None
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: CallArgs::Exprs(vec![Expr::Number(1.0)])
                    }))
                ]
            }
        );
    }

    #[test]
    fn test_if_else() {
        assert_parse!(
            "if true then\nprint(1)\nelse\nprint(2)\nend",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![
                    StmtStatus::Some(Stmt::If(If {
                        cond: Expr::Bool(true),
                        block: Block { stmt_ptrs: vec![1] },
                        else_if_blocks: vec![],
                        else_block: Some(Block { stmt_ptrs: vec![2] })
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: CallArgs::Exprs(vec![Expr::Number(1.0)])
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: CallArgs::Exprs(vec![Expr::Number(2.0)])
                    }))
                ]
            }
        );
    }

    #[test]
    fn test_if_elseif() {
        assert_parse!(
            "if true then\nprint(1)\nelseif false then\nprint(2)\nend",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![
                    StmtStatus::Some(Stmt::If(If {
                        cond: Expr::Bool(true),
                        block: Block { stmt_ptrs: vec![1] },
                        else_if_blocks: vec![(Expr::Bool(false), Block { stmt_ptrs: vec![2] },)],
                        else_block: None
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: CallArgs::Exprs(vec![Expr::Number(1.0)])
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: CallArgs::Exprs(vec![Expr::Number(2.0)])
                    }))
                ]
            }
        );
    }

    #[test]
    fn test_if_elseif_else() {
        assert_parse!(
            "if true then\nprint(1)\nelseif false then\nprint(2)\nelse\nprint(3)\nend",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![
                    StmtStatus::Some(Stmt::If(If {
                        cond: Expr::Bool(true),
                        block: Block { stmt_ptrs: vec![1] },
                        else_if_blocks: vec![(Expr::Bool(false), Block { stmt_ptrs: vec![2] },)],
                        else_block: Some(Block { stmt_ptrs: vec![3] })
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: CallArgs::Exprs(vec![Expr::Number(1.0)])
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: CallArgs::Exprs(vec![Expr::Number(2.0)])
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: CallArgs::Exprs(vec![Expr::Number(3.0)])
                    }))
                ]
            }
        );
    }

    // for tests:
    // 1. for i = 1, 10 do print(i) end
    // 2. for i = 1, 10, 2 do print(i) end
    // 3. for i in a do print(i) end
    // 4. for i, v in a do print(i, v) end
    // 5. for i, v in a, b, c do print(i, v) end

    #[test]
    fn test_for() {
        assert_parse!(
            "for i = 1, 10 do\nprint(i)\nend",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![
                    StmtStatus::Some(Stmt::For(For {
                        start: Expr::Number(1.0),
                        end: Expr::Number(10.0),
                        step: None,
                        block: Block { stmt_ptrs: vec![1] },
                        var: Binding {
                            name: "i".to_string(),
                            ty: None
                        }
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: CallArgs::Exprs(vec![Expr::Var(Var::Name("i".to_string()))])
                    }))
                ]
            }
        );
    }

    #[test]
    fn test_for_step() {
        assert_parse!(
            "for i = 1, 10, 2 do\nprint(i)\nend",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![
                    StmtStatus::Some(Stmt::For(For {
                        start: Expr::Number(1.0),
                        end: Expr::Number(10.0),
                        step: Some(Expr::Number(2.0)),
                        block: Block { stmt_ptrs: vec![1] },
                        var: Binding {
                            name: "i".to_string(),
                            ty: None
                        }
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: CallArgs::Exprs(vec![Expr::Var(Var::Name("i".to_string()))])
                    }))
                ]
            }
        );
    }

    #[test]
    fn test_for_in() {
        assert_parse!(
            "for i in a do\nprint(i)\nend",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![
                    StmtStatus::Some(Stmt::ForIn(ForIn {
                        block: Block { stmt_ptrs: vec![1] },
                        vars: vec![Binding {
                            name: "i".to_string(),
                            ty: None
                        }],
                        exprs: vec![Expr::Var(Var::Name("a".to_string()))]
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: CallArgs::Exprs(vec![Expr::Var(Var::Name("i".to_string()))])
                    }))
                ]
            }
        );
    }

    #[test]
    fn test_for_in_multivar_single_expr() {
        assert_parse!(
            "for i, v in a do\nprint(i, v)\nend",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![
                    StmtStatus::Some(Stmt::ForIn(ForIn {
                        block: Block { stmt_ptrs: vec![1] },
                        vars: vec![
                            Binding {
                                name: "i".to_string(),
                                ty: None
                            },
                            Binding {
                                name: "v".to_string(),
                                ty: None
                            }
                        ],
                        exprs: vec![Expr::Var(Var::Name("a".to_string()))]
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: CallArgs::Exprs(vec![
                            Expr::Var(Var::Name("i".to_string())),
                            Expr::Var(Var::Name("v".to_string()))
                        ])
                    }))
                ]
            }
        );
    }

    #[test]
    fn test_for_in_multivar_mutli_expr() {
        assert_parse!(
            "for i, v in a, b, c do\nprint(i, v)\nend",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![
                    StmtStatus::Some(Stmt::ForIn(ForIn {
                        block: Block { stmt_ptrs: vec![1] },
                        vars: vec![
                            Binding {
                                name: "i".to_string(),
                                ty: None
                            },
                            Binding {
                                name: "v".to_string(),
                                ty: None
                            }
                        ],
                        exprs: vec![
                            Expr::Var(Var::Name("a".to_string())),
                            Expr::Var(Var::Name("b".to_string())),
                            Expr::Var(Var::Name("c".to_string()))
                        ]
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: CallArgs::Exprs(vec![
                            Expr::Var(Var::Name("i".to_string())),
                            Expr::Var(Var::Name("v".to_string()))
                        ])
                    }))
                ]
            }
        );
    }

    #[test]
    fn test_anonymous_function() {
        assert_parse!(
            "local f = function(x, y) return x + y end\nf(1, 2)",
            Chunk {
                block: Block {
                    stmt_ptrs: vec![0, 1]
                },
                stmts: vec![
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "f".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::Function(Box::new(FunctionBody {
                            generics: vec![],
                            ret_ty: None,
                            params: vec![
                                Binding {
                                    name: "x".to_string(),
                                    ty: None
                                },
                                Binding {
                                    name: "y".to_string(),
                                    ty: None
                                }
                            ],
                            block: Block { stmt_ptrs: vec![2] }
                        }))]
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("f".to_string())),
                        args: CallArgs::Exprs(vec![Expr::Number(1.0), Expr::Number(2.0)])
                    })),
                    StmtStatus::Some(Stmt::Return(Return {
                        exprs: vec![Expr::BinOp(Box::new(BinOp {
                            op: BinOpKind::Add,
                            lhs: Expr::Var(Var::Name("x".to_string())),
                            rhs: Expr::Var(Var::Name("y".to_string()))
                        }))]
                    }))
                ]
            }
        );
    }

    // tests for tableconstructor:
    // 1. local t = {}
    // 2. local t = {1, 2, 3}
    // 3. local t = {a = 1, b = 2, c = 3}
    // 4. local t = {["a"] = 1, ["b"] = 2, ["c"] = 3}
    // 5. local t = {a = 1, 2, ["c"] = 3}

    #[test]
    fn test_tableconstructor_empty() {
        assert_parse!(
            "local t = {}",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(Stmt::Local(Local {
                    bindings: vec![Binding {
                        name: "t".to_string(),
                        ty: None
                    }],
                    init: vec![Expr::TableConstructor(TableConstructor { fields: vec![] })]
                }))]
            }
        );
    }

    #[test]
    fn test_tableconstructor_array() {
        assert_parse!(
            "local t = {1, 2, 3}",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(Stmt::Local(Local {
                    bindings: vec![Binding {
                        name: "t".to_string(),
                        ty: None
                    }],
                    init: vec![Expr::TableConstructor(TableConstructor {
                        fields: vec![
                            TableField::ImplicitKey(Expr::Number(1.0)),
                            TableField::ImplicitKey(Expr::Number(2.0)),
                            TableField::ImplicitKey(Expr::Number(3.0))
                        ]
                    })]
                }))]
            }
        );
    }

    #[test]
    fn test_tableconstructor_map() {
        assert_parse!(
            "local t = {a = 1, b = 2, c = 3}",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(Stmt::Local(Local {
                    bindings: vec![Binding {
                        name: "t".to_string(),
                        ty: None
                    }],
                    init: vec![Expr::TableConstructor(TableConstructor {
                        fields: vec![
                            TableField::ExplicitKey {
                                key: "a".to_string(),
                                value: Expr::Number(1.0)
                            },
                            TableField::ExplicitKey {
                                key: "b".to_string(),
                                value: Expr::Number(2.0)
                            },
                            TableField::ExplicitKey {
                                key: "c".to_string(),
                                value: Expr::Number(3.0)
                            }
                        ]
                    })]
                }))]
            }
        );
    }

    #[test]
    fn test_tableconstructor_map_with_string_keys() {
        assert_parse!(
            "local t = {['a'] = 1, ['b'] = 2, ['c'] = 3}",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(Stmt::Local(Local {
                    bindings: vec![Binding {
                        name: "t".to_string(),
                        ty: None
                    }],
                    init: vec![Expr::TableConstructor(TableConstructor {
                        fields: vec![
                            TableField::ArrayKey {
                                key: Expr::String("'a'".to_string()),
                                value: Expr::Number(1.0)
                            },
                            TableField::ArrayKey {
                                key: Expr::String("'b'".to_string()),
                                value: Expr::Number(2.0)
                            },
                            TableField::ArrayKey {
                                key: Expr::String("'c'".to_string()),
                                value: Expr::Number(3.0)
                            }
                        ]
                    })]
                }))]
            }
        );
    }

    #[test]
    fn test_tableconstructor_map_with_mixed_keys() {
        assert_parse!(
            "local t = {a = 1, 2, ['c'] = 3}",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(Stmt::Local(Local {
                    bindings: vec![Binding {
                        name: "t".to_string(),
                        ty: None
                    }],
                    init: vec![Expr::TableConstructor(TableConstructor {
                        fields: vec![
                            TableField::ExplicitKey {
                                key: "a".to_string(),
                                value: Expr::Number(1.0)
                            },
                            TableField::ImplicitKey(Expr::Number(2.0)),
                            TableField::ArrayKey {
                                key: Expr::String("'c'".to_string()),
                                value: Expr::Number(3.0)
                            }
                        ]
                    })]
                }))]
            }
        );
    }

    #[test]
    fn test_table_call() {
        assert_parse!(
            "local t = f{name = 1, 2, 3}",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(Stmt::Local(Local {
                    bindings: vec![Binding {
                        name: "t".to_string(),
                        ty: None
                    }],
                    init: vec![Expr::Call(Box::new(Call {
                        func: Expr::Var(Var::Name("f".to_string())),
                        args: CallArgs::Table(TableConstructor {
                            fields: vec![
                                TableField::ExplicitKey {
                                    key: "name".to_string(),
                                    value: Expr::Number(1.0)
                                },
                                TableField::ImplicitKey(Expr::Number(2.0)),
                                TableField::ImplicitKey(Expr::Number(3.0))
                            ]
                        })
                    }))]
                }))]
            }
        );
    }

    #[test]
    fn test_string_call() {
        assert_parse!(
            "local t = f'hello'",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(Stmt::Local(Local {
                    bindings: vec![Binding {
                        name: "t".to_string(),
                        ty: None
                    }],
                    init: vec![Expr::Call(Box::new(Call {
                        func: Expr::Var(Var::Name("f".to_string())),
                        args: CallArgs::String("'hello'".to_string())
                    }))]
                }))]
            }
        );
    }

    #[test]
    fn test_nested_calls() {
        assert_parse!(
            "local t = (f())()()",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(Stmt::Local(Local {
                    bindings: vec![Binding {
                        name: "t".to_string(),
                        ty: None
                    }],
                    init: vec![Expr::Call(Box::new(Call {
                        func: Expr::Call(Box::new(Call {
                            func: Expr::Wrap(Box::new(Expr::Call(Box::new(Call {
                                func: Expr::Var(Var::Name("f".to_string())),
                                args: CallArgs::Exprs(vec![])
                            })))),
                            args: CallArgs::Exprs(vec![])
                        })),
                        args: CallArgs::Exprs(vec![])
                    }))]
                }))]
            }
        );
    }

    // ifelseexp:
    // 1. local x = if true then 1 else 2
    // 2. local x = if true then 1 elseif false then 2 else 3
    // 3. local x = if true then 1 elseif false then 2 elseif false then 2 else 4

    #[test]
    fn test_ifelseexp() {
        assert_parse!(
            "local x = if true then 1 else 2",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(Stmt::Local(Local {
                    bindings: vec![Binding {
                        name: "x".to_string(),
                        ty: None
                    }],
                    init: vec![Expr::IfElse(Box::new(IfElseExp {
                        cond: Expr::Bool(true),
                        if_expr: Expr::Number(1.0),
                        else_expr: Expr::Number(2.0),
                        else_if_exprs: vec![]
                    }))]
                }))]
            }
        );
    }

    #[test]
    fn test_ifelseexp_with_else_if() {
        assert_parse!(
            "local x = if true then 1 elseif false then 2 else 3",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(Stmt::Local(Local {
                    bindings: vec![Binding {
                        name: "x".to_string(),
                        ty: None
                    }],
                    init: vec![Expr::IfElse(Box::new(IfElseExp {
                        cond: Expr::Bool(true),
                        if_expr: Expr::Number(1.0),
                        else_expr: Expr::Number(3.0),
                        else_if_exprs: vec![(Expr::Bool(false), Expr::Number(2.0))]
                    }))]
                }))]
            }
        );
    }

    #[test]
    fn test_ifelseexp_with_multiple_else_if() {
        assert_parse!(
            "local x = if true then 1 elseif false then 2 elseif false then 2 else 4",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(Stmt::Local(Local {
                    bindings: vec![Binding {
                        name: "x".to_string(),
                        ty: None
                    }],
                    init: vec![Expr::IfElse(Box::new(IfElseExp {
                        cond: Expr::Bool(true),
                        if_expr: Expr::Number(1.0),
                        else_expr: Expr::Number(4.0),
                        else_if_exprs: vec![
                            (Expr::Bool(false), Expr::Number(2.0)),
                            (Expr::Bool(false), Expr::Number(2.0))
                        ]
                    }))]
                }))]
            }
        );
    }

    #[test]
    fn parse_table_access() {
        assert_parse!(
            "local x = t[1]",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(Stmt::Local(Local {
                    bindings: vec![Binding {
                        name: "x".to_string(),
                        ty: None
                    }],
                    init: vec![Expr::Var(Var::TableAccess(Box::new(TableAccess {
                        expr: Expr::Var(Var::Name("t".to_string())),
                        index: Expr::Number(1.0)
                    })))]
                }))]
            }
        );
    }

    #[test]
    fn parse_table_access2() {
        assert_parse!(
            "local x = (1)[1]",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(Stmt::Local(Local {
                    bindings: vec![Binding {
                        name: "x".to_string(),
                        ty: None
                    }],
                    init: vec![Expr::Var(Var::TableAccess(Box::new(TableAccess {
                        expr: Expr::Wrap(Box::new(Expr::Number(1.0))),
                        index: Expr::Number(1.0)
                    })))]
                }))]
            }
        );
    }

    #[test]
    fn parse_field_access1() {
        assert_parse!(
            "local x = t.x",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(Stmt::Local(Local {
                    bindings: vec![Binding {
                        name: "x".to_string(),
                        ty: None
                    }],
                    init: vec![Expr::Var(Var::FieldAccess(Box::new(FieldAccess {
                        expr: Expr::Var(Var::Name("t".to_string())),
                        field: "x".to_string()
                    })))]
                }))]
            }
        );
    }

    #[test]
    fn parse_field_access2() {
        assert_parse!(
            "local x = (1).x",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(Stmt::Local(Local {
                    bindings: vec![Binding {
                        name: "x".to_string(),
                        ty: None
                    }],
                    init: vec![Expr::Var(Var::FieldAccess(Box::new(FieldAccess {
                        expr: Expr::Wrap(Box::new(Expr::Number(1.0))),
                        field: "x".to_string()
                    })))]
                }))]
            }
        );
    }
}
