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
        let (block, mut to_be_parsed) = self.parse_block(root)?;
        self.chunk.block = block;

        while let Some(StmtToBeParsed { ptr, node }) = to_be_parsed.pop() {
            match self.parse_stmt(node) {
                Ok((stmt, more_to_parse)) => {
                    self.chunk.set_stmt(ptr, StmtStatus::Some(stmt));
                    to_be_parsed.extend(more_to_parse);
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

        ParseError::SyntaxError(start, end, clipped)
    }

    fn parse_block(&mut self, node: tree_sitter::Node<'ts>) -> Result<(Block, UnparsedStmts<'ts>)> {
        let mut stmts = Vec::new();
        let mut to_be_parsed = Vec::new();
        let cursor = &mut node.walk();
        for child in node.children(cursor) {
            let stmt_ptr = self.chunk.alloc();
            stmts.push(stmt_ptr);
            to_be_parsed.push(StmtToBeParsed {
                ptr: stmt_ptr,
                node: child,
            });
        }
        Ok((Block { stmt_ptrs: stmts }, to_be_parsed))
    }

    fn parse_stmt(&mut self, node: tree_sitter::Node<'ts>) -> Result<(Stmt, UnparsedStmts<'ts>)> {
        let kind = node.kind();
        let mut to_be_parsed = Vec::new();

        /// parses and adds unparsed statements to `to_be_parsed`
        macro_rules! ez_parse {
            ($parse_fn:ident, $stmt:expr) => {{
                let (stmt, unparsed_stmts) = self.$parse_fn(node)?;
                to_be_parsed.extend(unparsed_stmts);
                $stmt(stmt)
            }};
        }

        let stmt = match kind {
            "local_var_stmt" => Stmt::Local(self.parse_local(node)?),
            "fn_stmt" => ez_parse!(parse_function_def, Stmt::FunctionDef),
            "local_fn_stmt" => ez_parse!(parse_local_function_def, Stmt::LocalFunctionDef),
            "do_stmt" => ez_parse!(parse_do, Stmt::Do),
            "while_stmt" => ez_parse!(parse_while, Stmt::While),
            "repeat_stmt" => ez_parse!(parse_repeat, Stmt::Repeat),
            "if_stmt" => ez_parse!(parse_if, Stmt::If),
            "call_stmt" => Stmt::Call(self.parse_call(node)?),
            "var_stmt" => Stmt::CompOp(self.parse_compop(node)?),
            "ret_stmt" => Stmt::Return(self.parse_return(node)?),
            "assign_stmt" => Stmt::Assign(self.parse_assign(node)?),
            // these don't need to be parsed separately
            "break_stmt" => Stmt::Break(Break {}),
            "continue_stmt" => Stmt::Continue(Continue {}),
            _ => todo!("parse_stmt: {}", kind),
        };
        Ok((stmt, to_be_parsed))
    }

    fn parse_local(&mut self, node: tree_sitter::Node<'ts>) -> Result<Local> {
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
                _ if parsing_init => init.push(self.parse_expr(child)?),
                _ => todo!("parse_local: {}", kind),
            }
        }
        Ok(Local { bindings, init })
    }

    fn parse_fn_body(
        &mut self,
        // NOTE: the node has to be the "(" or "<" of the function def
        node: tree_sitter::Node<'ts>,
    ) -> Result<(FunctionBody, UnparsedStmts<'ts>)> {
        let mut params = Vec::new();
        let mut to_be_parsed = Vec::new();
        let mut block = None;
        let mut generics = Vec::new();
        let mut ret_ty = None;

        enum Step {
            Generics,
            Params,
            Block,
        }

        let mut step = Step::Generics;

        for node in (NodeSiblingIter { node: Some(node) }) {
            let kind = node.kind();
            match (kind, &step) {
                ("<", Step::Generics) => {}
                ("generic", Step::Generics) => {
                    let txt = self.extract_text(node);
                    generics.push(GenericParam::Name(txt.to_string()));
                }
                ("genpack", Step::Generics) => {
                    let txt = self.extract_text(node.child(0).ok_or_else(|| self.error(node))?);
                    generics.push(GenericParam::Pack(txt.to_string()));
                }
                (">", Step::Generics) => step = Step::Params,
                (",", Step::Generics | Step::Params) => {}
                ("(", Step::Generics | Step::Params) => step = Step::Params,
                ("param", Step::Params) => params.push(self.parse_binding(node)?),
                (")", Step::Params) => step = Step::Block,
                ("block", Step::Block) => {
                    let (b, unparsed_stmts) = self.parse_block(node)?;
                    to_be_parsed.extend(unparsed_stmts);
                    block = Some(b);
                }
                ("end", Step::Block) => break,
                _ => todo!("parse_fn_body: {}", kind),
            }
        }

        Ok((
            FunctionBody {
                params,
                generics,
                ret_ty,
                // there may be no block if the function is empty
                block: block.unwrap_or_default(),
            },
            to_be_parsed,
        ))
    }

    fn parse_function_def(
        &mut self,
        node: tree_sitter::Node<'ts>,
    ) -> Result<(FunctionDef, UnparsedStmts<'ts>)> {
        let cursor = &mut node.walk();

        // vars to fill
        let mut table = None;
        let mut is_method = false;
        let mut name = String::new();
        let mut body = None;
        let mut to_be_parsed = Vec::new();

        enum Step {
            // we assume that a function def starts with just it's name
            Name,
            // the name was actually a table, so now we need to parse the actual name
            NameWasTable,
        }

        let mut step = Step::Name;

        for child in node.children(cursor) {
            let kind = child.kind();
            match kind {
                "name" => match step {
                    Step::Name => {
                        name.push_str(self.extract_text(child));
                    }
                    Step::NameWasTable => {
                        let table_name = self.extract_text(child);
                        table = Some(std::mem::replace(&mut name, table_name.to_string()));
                    }
                },
                "<" | "(" => {
                    let (parsed_body, unparsed_stmts) = self.parse_fn_body(child)?;
                    to_be_parsed.extend(unparsed_stmts);
                    body = Some(parsed_body);
                    break;
                }
                "." => {
                    step = Step::NameWasTable;
                }
                ":" => {
                    step = Step::NameWasTable;
                    is_method = true;
                }
                "function" => {}
                _ => todo!("parse_function_def: {}", kind),
            }
        }
        Ok((
            FunctionDef {
                table,
                is_method,
                name,
                body: body.ok_or_else(|| self.error(node))?,
            },
            to_be_parsed,
        ))
    }

    fn parse_local_function_def(
        &mut self,
        node: tree_sitter::Node<'ts>,
    ) -> Result<(LocalFunctionDef, UnparsedStmts<'ts>)> {
        let cursor = &mut node.walk();

        // vars to fill
        let mut name = String::new();
        let mut body = None;
        let mut to_be_parsed = Vec::new();

        for child in node.children(cursor) {
            let kind = child.kind();
            match kind {
                "name" => name.push_str(self.extract_text(child)),
                "<" | "(" => {
                    let (parsed_body, unparsed_stmts) = self.parse_fn_body(child)?;
                    to_be_parsed.extend(unparsed_stmts);
                    body = Some(parsed_body);
                    break;
                }
                "local" | "function" => {}
                _ => todo!("parse_function_def: {}", kind),
            }
        }
        Ok((
            LocalFunctionDef {
                name,
                body: body.ok_or_else(|| self.error(node))?,
            },
            to_be_parsed,
        ))
    }

    fn parse_do(&mut self, node: tree_sitter::Node<'ts>) -> Result<(Do, UnparsedStmts<'ts>)> {
        let cursor = &mut node.walk();
        let mut block = None;
        let mut to_be_parsed = Vec::new();

        for child in node.children(cursor) {
            let kind = child.kind();
            match kind {
                "do" | "end" => {}
                "block" => {
                    let (b, unparsed_stmts) = self.parse_block(child)?;
                    to_be_parsed.extend(unparsed_stmts);
                    block = Some(b);
                }
                _ => return Err(self.error(child)),
            }
        }

        Ok((
            Do {
                block: block.ok_or_else(|| self.error(node))?,
            },
            to_be_parsed,
        ))
    }

    fn parse_while(&mut self, node: tree_sitter::Node<'ts>) -> Result<(While, UnparsedStmts<'ts>)> {
        let cursor = &mut node.walk();
        let mut cond = None;
        let mut block = None;
        let mut to_be_parsed = Vec::new();

        enum Step {
            Condition,
            Block,
        }

        let mut step = Step::Condition;

        for child in node.children(cursor) {
            let kind = child.kind();
            match kind {
                "while" | "do" | "end" => {}
                _ if matches!(step, Step::Condition) => {
                    cond = Some(self.parse_expr(child)?);
                    step = Step::Block;
                }
                "block" => {
                    let (b, unparsed_stmts) = self.parse_block(child)?;
                    to_be_parsed.extend(unparsed_stmts);
                    block = Some(b);
                }
                _ => return Err(self.error(child)),
            }
        }

        Ok((
            While {
                cond: cond.ok_or_else(|| self.error(node))?,
                block: block.ok_or_else(|| self.error(node))?,
            },
            to_be_parsed,
        ))
    }

    fn parse_repeat(
        &mut self,
        node: tree_sitter::Node<'ts>,
    ) -> Result<(Repeat, UnparsedStmts<'ts>)> {
        let cursor = &mut node.walk();
        let mut cond = None;
        let mut block = None;
        let mut to_be_parsed = Vec::new();

        enum Step {
            Block,
            Condition,
        }

        let mut step = Step::Block;

        for child in node.children(cursor) {
            let kind = child.kind();
            match kind {
                "repeat" | "until" => {}
                "block" => {
                    let (b, unparsed_stmts) = self.parse_block(child)?;
                    to_be_parsed.extend(unparsed_stmts);
                    block = Some(b);
                    step = Step::Condition;
                }
                _ if matches!(step, Step::Condition) => {
                    cond = Some(self.parse_expr(child)?);
                }
                _ => return Err(self.error(child)),
            }
        }

        Ok((
            Repeat {
                cond: cond.ok_or_else(|| self.error(node))?,
                block: block.ok_or_else(|| self.error(node))?,
            },
            to_be_parsed,
        ))
    }

    fn parse_if(&mut self, node: tree_sitter::Node<'ts>) -> Result<(If, UnparsedStmts<'ts>)> {
        let cursor = &mut node.walk();
        let mut cond = None;
        let mut if_block = None;
        let mut else_if_blocks = Vec::new();
        let mut else_block = None;
        let mut to_be_parsed = Vec::new();

        #[derive(Debug)]
        enum Step {
            Condition,
            Else,
            Block,
        }

        let mut step = Step::Condition;

        for child in node.children(cursor) {
            let kind = child.kind();
            match (kind, &step) {
                ("end", _) => {}
                ("if", Step::Condition) => {}
                ("then", Step::Block | Step::Else) => {}
                (_, Step::Condition) => {
                    cond = Some(self.parse_expr(child)?);
                    step = Step::Block;
                }
                (_, Step::Block) => {
                    let (b, unparsed_stmts) = self.parse_block(child)?;
                    to_be_parsed.extend(unparsed_stmts);
                    if_block = Some(b);
                    step = Step::Else;
                }
                ("else_clause", Step::Else) => {
                    let (b, unparsed_stmts) = child
                        .child(1)
                        .map(|n| self.parse_block(n))
                        .unwrap_or_else(|| Ok((Block::default(), Vec::new())))?;
                    to_be_parsed.extend(unparsed_stmts);
                    else_block = Some(b);
                }
                ("elseif_clause", Step::Else) => {
                    let cond_node = child.child(1).ok_or_else(|| self.error(child))?;
                    let cond = self.parse_expr(cond_node)?;
                    let (b, unparsed_stmts) = child
                        .child(3)
                        .map(|n| self.parse_block(n))
                        .unwrap_or_else(|| Ok((Block::default(), Vec::new())))?;
                    to_be_parsed.extend(unparsed_stmts);
                    else_if_blocks.push((cond, b));
                }
                _ => todo!("parse_if ({:?}): {}", step, kind),
            }
        }

        Ok((
            If {
                cond: cond.ok_or_else(|| self.error(node))?,
                block: if_block.ok_or_else(|| self.error(node))?,
                else_if_blocks,
                else_block,
            },
            to_be_parsed,
        ))
    }

    fn parse_call(&mut self, node: tree_sitter::Node<'ts>) -> Result<Call> {
        let cursor = &mut node.walk();
        let mut expr = None;
        let mut args = Vec::new();

        let mut parsing_expr = true; // first child is always expr
        let mut parsing_args = false;

        for child in node.children(cursor) {
            let kind = child.kind();
            match kind {
                _ if parsing_expr => {
                    expr = Some(self.parse_expr(child)?);
                    parsing_expr = false
                }
                "arglist" => {
                    let cursor = &mut child.walk();
                    for arg in child.children(cursor) {
                        let kind = arg.kind();
                        match kind {
                            "(" => parsing_args = true,
                            ")" => parsing_args = false,
                            "," => {}
                            _ if parsing_args => args.push(self.parse_expr(arg)?),
                            _ => todo!("parse_call (arglist): {}", kind),
                        }
                    }
                }
                _ => todo!("parse_call: {}", kind),
            }
        }
        Ok(Call {
            func: expr.ok_or_else(|| self.error(node))?,
            args,
        })
    }

    fn parse_compop(&mut self, node: tree_sitter::Node<'ts>) -> Result<CompOp> {
        let cursor = &mut node.walk();
        let mut var = None;
        let mut op = None;
        let mut expr = None;

        enum Step {
            Var,
            Op,
            Expr,
        }

        let mut step = Step::Var;

        for child in node.children(cursor) {
            let kind = child.kind();
            match (kind, &step) {
                ("var", Step::Var) => {
                    var = Some(self.parse_var(child)?);
                    step = Step::Op;
                }
                (_, Step::Op) => {
                    let txt = self.extract_text(child);
                    op = Some(CompOpKind::from_str(txt).map_err(|_| self.error(child))?);
                    step = Step::Expr;
                }
                (_, Step::Expr) => expr = Some(self.parse_expr(child)?),
                _ => todo!("parse_compop: {}", kind),
            }
        }

        Ok(CompOp {
            lhs: var.ok_or_else(|| self.error(node))?,
            op: op.ok_or_else(|| self.error(node))?,
            rhs: Box::new(expr.ok_or_else(|| self.error(node))?),
        })
    }

    fn parse_return(&mut self, node: tree_sitter::Node<'ts>) -> Result<Return> {
        let cursor = &mut node.walk();
        let mut exprs = Vec::new();
        for child in node.children(cursor) {
            let kind = child.kind();
            match kind {
                "return" => {}
                _ => exprs.push(self.parse_expr(child)?),
            }
        }
        Ok(Return { exprs })
    }

    fn parse_assign(&mut self, node: tree_sitter::Node<'ts>) -> Result<Assign> {
        let cursor = &mut node.walk();
        let mut vars = Vec::new();
        let mut exprs = Vec::new();

        enum Step {
            Vars,
            Exprs,
        }

        let mut step = Step::Vars;

        for child in node.children(cursor) {
            let kind = child.kind();
            match (kind, &step) {
                ("var", Step::Vars) => vars.push(self.parse_var(child)?),
                ("=", Step::Vars) => step = Step::Exprs,
                (",", _) => {}
                (_, Step::Exprs) => exprs.push(self.parse_expr(child)?),
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

    fn parse_expr(&mut self, node: tree_sitter::Node<'ts>) -> Result<Expr> {
        let kind = node.kind();
        match kind {
            "number" => {
                let text = self.extract_text(node);
                let num = text.parse().map_err(|_| self.error(node))?;
                Ok(Expr::Number(num))
            }
            "string" => {
                let text = self.extract_text(node);
                let text = text.trim_matches('"');
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
            "vararg" => Ok(Expr::VarArg),
            "exp_wrap" => Ok(Expr::Wrap(Box::new(
                self.parse_expr(node.child(1).ok_or_else(|| self.error(node))?)?,
            ))),
            // delegate to parse_var
            "var" => Ok(Expr::Var(self.parse_var(node)?)),
            // delegate to parse_binop
            "binexp" => Ok(Expr::BinOp(self.parse_binop(node)?)),
            // delegate to parse_unop
            "unexp" => Ok(Expr::UnOp(self.parse_unop(node)?)),
            _ => todo!("parse_expr: {}", kind),
        }
    }

    fn parse_var(&mut self, node: tree_sitter::Node<'ts>) -> Result<Var> {
        let mut name = String::new();
        let cursor = &mut node.walk();
        for child in node.children(cursor) {
            let kind = child.kind();
            match kind {
                "name" => name.push_str(self.extract_text(child)),
                _ => todo!("parse_var: {}", kind),
            }
        }
        Ok(Var::Name(name))
    }

    fn parse_binop(&mut self, node: tree_sitter::Node<'ts>) -> Result<BinOp> {
        let cursor = &mut node.walk();
        let mut op = None;
        let mut lhs = None;
        let mut rhs = None;

        enum Step {
            Lhs,
            Op,
            Rhs,
        }

        let mut step = Step::Lhs;
        for child in node.children(cursor) {
            match step {
                Step::Lhs => {
                    lhs = Some(self.parse_expr(child)?);
                    step = Step::Op;
                }
                Step::Op => {
                    let txt = self.extract_text(child);
                    let parsed_op = BinOpKind::from_str(txt).map_err(|_| self.error(node))?;
                    op = Some(parsed_op);
                    step = Step::Rhs;
                }
                Step::Rhs => {
                    rhs = Some(self.parse_expr(child)?);
                }
            }
        }
        Ok(BinOp {
            lhs: Box::new(lhs.ok_or_else(|| self.error(node))?),
            op: op.ok_or(self.error(node))?,
            rhs: Box::new(rhs.ok_or_else(|| self.error(node))?),
        })
    }

    fn parse_unop(&mut self, node: tree_sitter::Node<'ts>) -> Result<UnOp> {
        let cursor = &mut node.walk();
        let mut op = None;
        let mut expr = None;

        enum Step {
            Op,
            Expr,
        }

        let mut step = Step::Op;
        for child in node.children(cursor) {
            match step {
                Step::Op => {
                    let txt = self.extract_text(child);
                    let parsed_op = UnOpKind::from_str(txt).map_err(|_| self.error(node))?;
                    op = Some(parsed_op);
                    step = Step::Expr;
                }
                Step::Expr => {
                    expr = Some(self.parse_expr(child)?);
                }
            }
        }
        Ok(UnOp {
            op: op.ok_or_else(|| self.error(node))?,
            expr: Box::new(expr.ok_or_else(|| self.error(node))?),
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
    fn parse_string() {
        assert_parse!(
            "local a = \"hello\"",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(Stmt::Local(Local {
                    bindings: vec![Binding {
                        name: "a".to_string(),
                        ty: None
                    }],
                    init: vec![Expr::String("hello".to_string())]
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
                        table: None,
                        is_method: false
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("f".to_string())),
                        args: vec![Expr::Number(3.0)]
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: vec![Expr::Var(Var::Name("n".to_string()))]
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
                        table: None,
                        is_method: false
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("f".to_string())),
                        args: vec![Expr::Number(3.0), Expr::Number(4.0)]
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: vec![Expr::Var(Var::Name("a".to_string()))]
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: vec![Expr::Var(Var::Name("b".to_string()))]
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
                        table: Some("t".to_string()),
                        is_method: false
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: vec![Expr::Var(Var::Name("n".to_string()))]
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
                        table: Some("t".to_string()),
                        is_method: true
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: vec![Expr::Var(Var::Name("self".to_string()))]
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
                        args: vec![Expr::Number(3.0)]
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: vec![Expr::Var(Var::Name("n".to_string()))]
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
                    table: None,
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
                    table: None,
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
                        init: vec![Expr::BinOp(BinOp {
                            lhs: Box::new(Expr::Number(1.0)),
                            op: BinOpKind::Add,
                            rhs: Box::new(Expr::Number(2.0))
                        }),]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "minus".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::BinOp(BinOp {
                            lhs: Box::new(Expr::Number(1.0)),
                            op: BinOpKind::Sub,
                            rhs: Box::new(Expr::Number(2.0))
                        }),]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "mul".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::BinOp(BinOp {
                            lhs: Box::new(Expr::Number(1.0)),
                            op: BinOpKind::Mul,
                            rhs: Box::new(Expr::Number(2.0))
                        }),]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "div".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::BinOp(BinOp {
                            lhs: Box::new(Expr::Number(1.0)),
                            op: BinOpKind::Div,
                            rhs: Box::new(Expr::Number(2.0))
                        }),]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "mod".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::BinOp(BinOp {
                            lhs: Box::new(Expr::Number(1.0)),
                            op: BinOpKind::Mod,
                            rhs: Box::new(Expr::Number(2.0))
                        }),]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "pow".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::BinOp(BinOp {
                            lhs: Box::new(Expr::Number(1.0)),
                            op: BinOpKind::Pow,
                            rhs: Box::new(Expr::Number(2.0))
                        }),]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "concat".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::BinOp(BinOp {
                            lhs: Box::new(Expr::String("a".to_string())),
                            op: BinOpKind::Concat,
                            rhs: Box::new(Expr::String("b".to_string()))
                        }),]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "eq".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::BinOp(BinOp {
                            lhs: Box::new(Expr::Number(1.0)),
                            op: BinOpKind::Eq,
                            rhs: Box::new(Expr::Number(2.0))
                        }),]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "neq".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::BinOp(BinOp {
                            lhs: Box::new(Expr::Number(1.0)),
                            op: BinOpKind::Ne,
                            rhs: Box::new(Expr::Number(2.0))
                        }),]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "lt".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::BinOp(BinOp {
                            lhs: Box::new(Expr::Number(1.0)),
                            op: BinOpKind::Lt,
                            rhs: Box::new(Expr::Number(2.0))
                        }),]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "gt".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::BinOp(BinOp {
                            lhs: Box::new(Expr::Number(1.0)),
                            op: BinOpKind::Gt,
                            rhs: Box::new(Expr::Number(2.0))
                        }),]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "leq".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::BinOp(BinOp {
                            lhs: Box::new(Expr::Number(1.0)),
                            op: BinOpKind::Le,
                            rhs: Box::new(Expr::Number(2.0))
                        }),]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "geq".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::BinOp(BinOp {
                            lhs: Box::new(Expr::Number(1.0)),
                            op: BinOpKind::Ge,
                            rhs: Box::new(Expr::Number(2.0))
                        }),]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "and".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::BinOp(BinOp {
                            lhs: Box::new(Expr::Bool(true)),
                            op: BinOpKind::And,
                            rhs: Box::new(Expr::Bool(false))
                        }),]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "or".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::BinOp(BinOp {
                            lhs: Box::new(Expr::Bool(true)),
                            op: BinOpKind::Or,
                            rhs: Box::new(Expr::Bool(false))
                        }),]
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
                        init: vec![Expr::UnOp(UnOp {
                            op: UnOpKind::Not,
                            expr: Box::new(Expr::Bool(true))
                        }),]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "neg".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::UnOp(UnOp {
                            op: UnOpKind::Neg,
                            expr: Box::new(Expr::Number(1.0))
                        }),]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "neg_wrap".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::UnOp(UnOp {
                            op: UnOpKind::Neg,
                            expr: Box::new(Expr::Wrap(Box::new(Expr::Number(1.0))))
                        }),]
                    })),
                    StmtStatus::Some(Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "len".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::UnOp(UnOp {
                            op: UnOpKind::Len,
                            expr: Box::new(Expr::String("aaaaa".to_string()))
                        }),]
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
                    init: vec![Expr::BinOp(BinOp {
                        lhs: Box::new(Expr::Wrap(Box::new(Expr::BinOp(BinOp {
                            lhs: Box::new(Expr::BinOp(BinOp {
                                lhs: Box::new(Expr::Number(1.0)),
                                op: BinOpKind::Add,
                                rhs: Box::new(Expr::Wrap(Box::new(Expr::BinOp(BinOp {
                                    lhs: Box::new(Expr::Number(2.0)),
                                    op: BinOpKind::Mul,
                                    rhs: Box::new(Expr::Number(3.0))
                                }))))
                            })),
                            op: BinOpKind::Sub,
                            rhs: Box::new(Expr::Wrap(Box::new(Expr::Number(1.0))))
                        })))),
                        op: BinOpKind::Div,
                        rhs: Box::new(Expr::Number(4.0))
                    }),]
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
                        table: None,
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
                        table: None,
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
                        args: vec![Expr::Number(1.0)]
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
                        args: vec![Expr::Number(1.0)]
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
                        args: vec![Expr::Number(1.0)]
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
                        args: vec![Expr::Number(1.0)]
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
                        args: vec![Expr::Number(1.0)]
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: vec![Expr::Number(2.0)]
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
                        args: vec![Expr::Number(1.0)]
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: vec![Expr::Number(2.0)]
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
                        args: vec![Expr::Number(1.0)]
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: vec![Expr::Number(2.0)]
                    })),
                    StmtStatus::Some(Stmt::Call(Call {
                        func: Expr::Var(Var::Name("print".to_string())),
                        args: vec![Expr::Number(3.0)]
                    }))
                ]
            }
        );
    }
}
