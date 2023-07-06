use std::collections::VecDeque;
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
    acc_trailing_comments: Vec<Comment>,
    pretty_errors: bool,
}

struct StmtToBeParsed<'ts> {
    ptr: usize,
    prev_stmt: Option<tree_sitter::Node<'ts>>,
    prev_ptr: usize,
    node: tree_sitter::Node<'ts>,
    comment_nodes: Vec<tree_sitter::Node<'ts>>,
}

type UnparsedStmts<'ts> = VecDeque<StmtToBeParsed<'ts>>;

struct NodeSiblingIter<'a> {
    node: Option<tree_sitter::Node<'a>>,
}

impl<'a> NodeSiblingIter<'a> {
    fn new(node: tree_sitter::Node<'a>) -> Self {
        Self { node: Some(node) }
    }
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
            acc_trailing_comments: Vec::new(),
            pretty_errors: false,
        }
    }

    pub fn set_pretty_errors(&mut self, pretty_errors: bool) {
        self.pretty_errors = pretty_errors;
    }

    pub fn parse(mut self) -> Result<Chunk> {
        let tree = self.ts.parse(self.text, None).ok_or(ParseError::TSError)?;
        let root = tree.root_node();
        let mut to_be_parsed = VecDeque::new();
        let block = self.parse_block(root, &mut to_be_parsed)?;
        self.chunk.block = block;

        while let Some(StmtToBeParsed {
            ptr,
            prev_stmt,
            prev_ptr,
            node,
            comment_nodes,
        }) = to_be_parsed.pop_front()
        {
            match self.parse_stmt(node, &mut to_be_parsed) {
                Ok(stmt) => {
                    // resolve comments
                    let mut comms = Vec::new();
                    for node in comment_nodes {
                        let txt = self.extract_text(node).to_string();

                        match prev_stmt {
                            Some(prev_stmt) => {
                                let comm_start_row = node.start_position().row;
                                let comm_end_row = node.end_position().row;
                                let prev_start_row = prev_stmt.start_position().row;
                                let prev_end_row = prev_stmt.end_position().row;

                                if comm_start_row >= prev_start_row && comm_end_row <= prev_end_row
                                {
                                    // this is a trailing comment of the previous statement
                                    let comm = Comment::Trailing(txt);
                                    self.chunk.add_comment(prev_ptr, comm);
                                } else {
                                    // just a leading comment of this statement
                                    comms.push(Comment::Leading(txt));
                                }
                            }
                            None => comms.push(Comment::Leading(txt)),
                        };
                    }

                    // add all the accumulated trailing comments
                    comms.append(&mut self.acc_trailing_comments);
                    self.chunk.set_stmt(ptr, StmtStatus::Some(stmt, comms));
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
        let mut clipped = String::new();
        let start = node.start_position();
        let end = node.end_position();

        if self.pretty_errors {
            let clip_start = std::cmp::max(0, (start.row as i32) - 3) as usize;
            let clip_end = end.row + 3;

            for (i, line) in self.text.lines().skip(clip_start).enumerate() {
                let i = i + clip_start;
                clipped.push_str(line);
                clipped.push('\n');
                // if i is in range of start.row and end.row, add a marker
                if i >= start.row && i <= end.row {
                    // TODO: this logic does not work for multi-line errors
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
        } else {
            clipped = self.extract_text(node).to_string();
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
        let cursor = &mut node.walk();

        let mut stmts = Vec::new();
        let mut comments = Vec::new();

        let mut prev_stmt = None;
        let mut prev_ptr = 0;

        for child in node.children(cursor) {
            // some edge cases
            let kind = child.kind();
            match kind {
                "comment" => {
                    comments.push(child);
                    continue;
                }
                ";" => continue,
                _ => {}
            }

            let stmt_ptr = self.chunk.alloc();
            stmts.push(stmt_ptr);

            unp.push_back(StmtToBeParsed {
                ptr: stmt_ptr,
                prev_stmt,
                prev_ptr,
                node: child,
                comment_nodes: comments.drain(..).collect(),
            });

            prev_stmt = Some(child);
            prev_ptr = stmt_ptr;
        }

        // parse the remaining comments
        for comment in comments {
            self.parse_comment_tr(comment);
        }

        Ok(Block { stmt_ptrs: stmts })
    }

    fn parse_comment_tr(&mut self, node: tree_sitter::Node<'ts>) {
        let text = self.extract_text(node);
        let comment = Comment::Trailing(text.to_string());
        self.acc_trailing_comments.push(comment);
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
            "type_stmt" => ez_parse!(parse_type_def, Stmt::TypeDef),
            // \ these don't need to be parsed separately /
            "break_stmt" => Stmt::Break(Break {}),
            "continue_stmt" => Stmt::Continue(Continue {}),
            _ => return Err(self.error(node)),
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
                "binding" => bindings.push(self.parse_binding(child, unp)?),
                "local" | "," => {}

                // start of init exprs
                "=" => {
                    parsing_init = true;
                }
                "comment" => self.parse_comment_tr(child),
                // delegate to expr parser
                _ if parsing_init => init.push(self.parse_expr(child, unp)?),
                _ => return Err(self.error(child)),
            }
        }
        Ok(Local { bindings, init })
    }

    fn parse_function_body(
        &mut self,
        // NOTE: the node has to be the "(" or "<" of the function def
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<FunctionBody> {
        let mut params = Vec::new();
        let mut block = None;
        let mut generics = Vec::new();
        let mut vararg = None;
        let mut ret_ty = None;

        enum State {
            Generics,
            Params,
            RetTy,
            Block,
        }

        let mut state = State::Generics;

        for node in NodeSiblingIter::new(node) {
            let kind = node.kind();
            match (kind, &state) {
                ("<", State::Generics) => {}
                ("generic" | "genpack", State::Generics) => {
                    generics.push(self.parse_generic_param(node)?);
                }
                (">", State::Generics) => state = State::Params,
                (",", State::Generics | State::Params) => {}
                ("(", State::Generics | State::Params) => state = State::Params,
                ("param", State::Params) => match node.child(0) {
                    // yuck...
                    Some(n) if n.kind() == "vararg" => {
                        vararg = Some(self.parse_vararg(node, unp)?);
                    }
                    Some(n) if n.kind() == "comment" => self.parse_comment_tr(n),
                    _ => params.push(self.parse_binding(node, unp)?),
                },
                (")", State::Params) => state = State::Block,
                (":", State::Block) => state = State::RetTy,
                ("comment", _) => self.parse_comment_tr(node),
                (_, State::RetTy) => {
                    ret_ty = Some(self.parse_type_or_pack(node, unp)?);
                    state = State::Block;
                }
                ("block", State::Block) => {
                    let b = self.parse_block(node, unp)?;
                    block = Some(b);
                }
                ("end", State::Block) => break,
                _ => return Err(self.error(node)),
            }
        }

        Ok(FunctionBody {
            params,
            generics,
            vararg,
            ret_ty,
            // there may be no block if the function is empty
            block: block.unwrap_or_default(),
        })
    }

    fn parse_generic_param(&mut self, node: tree_sitter::Node<'ts>) -> Result<GenericParam> {
        let kind = node.kind();
        match kind {
            "generic" => {
                let txt = self.extract_text(node);
                Ok(GenericParam::Name(txt.to_string()))
            }
            "genpack" => {
                let txt = self.extract_text(node.child(0).ok_or_else(|| self.error(node))?);
                Ok(GenericParam::Pack(txt.to_string()))
            }
            _ => Err(self.error(node)),
        }
    }

    fn parse_generic_def(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<GenericDef> {
        let mut cursor = node.walk();
        let mut param = None;
        let mut default = None;
        let mut default_next = false;
        for child in node.children(&mut cursor) {
            let kind = child.kind();
            match kind {
                "generic" | "genpack" => {
                    param = Some(self.parse_generic_param(child)?);
                }
                "=" => default_next = true,
                _ if default_next => {
                    default = Some(self.parse_type_or_pack(child, unp)?);
                    break;
                }
                _ => return Err(self.error(child)),
            }
        }
        Ok(GenericDef {
            param: param.ok_or_else(|| self.error(node))?,
            default,
        })
    }

    fn parse_vararg(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<Option<Type>> {
        let cursor = &mut node.walk();
        let mut res = None;
        let mut next_is_type = false;
        for child in node.children(cursor) {
            let kind = child.kind();
            match kind {
                "vararg" => {}
                ":" => next_is_type = true,
                _ if next_is_type => {
                    res = Some(self.parse_type(child, unp)?);
                    break;
                }
                "comment" => self.parse_comment_tr(child),
                _ => return Err(self.error(child)),
            }
        }
        Ok(res)
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

        enum State {
            // we assume that a function def starts with just it's name
            Name,
            // the name was actually a table, so now we need to parse the actual name
            NameWasTable,
        }

        let mut state = State::Name;

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
                    let parsed_body = self.parse_function_body(child, unp)?;
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
                "comment" => self.parse_comment_tr(child),
                _ => return Err(self.error(child)),
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
                    let parsed_body = self.parse_function_body(child, unp)?;
                    body = Some(parsed_body);
                    break;
                }
                "local" | "function" => {}
                "comment" => self.parse_comment_tr(child),
                _ => return Err(self.error(child)),
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
                "comment" => self.parse_comment_tr(child),
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
                "comment" => self.parse_comment_tr(child),
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
                "comment" => self.parse_comment_tr(child),
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
                ("comment", _) => self.parse_comment_tr(child),
                (_, State::Condition) => {
                    cond = Some(self.parse_expr(child, unp)?);
                    state = State::Block;
                }
                ("else_clause", State::Else | State::Block) => {
                    else_block = Some(
                        child
                            .child(1)
                            .map(|n| self.parse_block(n, unp))
                            .unwrap_or_else(|| Ok(Block::default()))?,
                    );
                }
                ("elseif_clause", State::Else | State::Block) => {
                    let cond_node = child.child(1).ok_or_else(|| self.error(child))?;
                    let cond = self.parse_expr(cond_node, unp)?;
                    let b = child
                        .child(3)
                        .map(|n| self.parse_block(n, unp))
                        .unwrap_or_else(|| Ok(Block::default()))?;
                    else_if_blocks.push((cond, b));
                }
                (_, State::Block) => {
                    if_block = Some(self.parse_block(child, unp)?);
                    state = State::Else;
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
                    var = Some(self.parse_binding(child, unp)?);
                    state = State::Start;
                }
                ("=", State::Start) => {}
                ("comment", _) => self.parse_comment_tr(child),
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
                    vars.push(self.parse_binding(child, unp)?);
                }
                (",", State::Expr | State::Var) => {}
                ("comment", _) => self.parse_comment_tr(child),
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
        let mut method = None;

        let mut parsing_expr = true; // first child is always expr
        let mut has_method = false;

        for child in node.children(cursor) {
            let kind = child.kind();
            match kind {
                "comment" => self.parse_comment_tr(child),
                _ if parsing_expr => {
                    expr = Some(self.parse_expr(child, unp)?);
                    parsing_expr = false
                }
                ":" => has_method = true,
                "name" if has_method => {
                    let method_name = self.extract_text(child);
                    method = Some(method_name.to_string());
                }
                "arglist" => {
                    let cursor = &mut child.walk();
                    for arg in child.children(cursor) {
                        let kind = arg.kind();
                        match (kind, &mut args) {
                            ("(", None) => args = Some(CallArgs::Exprs(Vec::new())),
                            (")" | ",", Some(CallArgs::Exprs(_))) => {}
                            ("comment", _) => self.parse_comment_tr(arg),
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
            method,
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
                ("comment", _) => self.parse_comment_tr(child),
                (_, State::Op) => {
                    let txt = self.extract_text(child);
                    op = Some(CompOpKind::from_str(txt).map_err(|_| self.error(child))?);
                    state = State::Expr;
                }
                (_, State::Expr) => expr = Some(self.parse_expr(child, unp)?),
                _ => return Err(self.error(child)),
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
                "return" | "," | ";" => {}
                "comment" => self.parse_comment_tr(child),
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
                ("comment", _) => self.parse_comment_tr(child),
                (_, State::Exprs) => exprs.push(self.parse_expr(child, unp)?),
                _ => return Err(self.error(child)),
            }
        }

        Ok(Assign { vars, exprs })
    }

    fn parse_type_def(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<TypeDef> {
        let cursor = &mut node.walk();
        let mut is_exported = false;
        let mut name = String::new();
        let mut generics = Vec::new();
        let mut ty = None;

        enum State {
            Init,
            Generics,
            Type,
        }

        let mut state = State::Init;

        for child in node.children(cursor) {
            let kind = child.kind();
            match (kind, &state) {
                ("type" | " ", State::Init) => {}
                ("name", State::Init) => name.push_str(self.extract_text(child)),
                ("export", State::Init) => is_exported = true,
                ("=", State::Init) => state = State::Type,
                ("<", State::Init) => state = State::Generics,
                (",", State::Generics) => {}
                ("genericdef" | "genpackdef", State::Generics) => {
                    generics.push(self.parse_generic_def(child, unp)?)
                }
                (">", State::Generics) => state = State::Init,
                ("comment", _) => self.parse_comment_tr(child),
                (_, State::Type) => ty = Some(self.parse_type(child, unp)?),
                _ => {
                    eprintln!("parse_type_def, kind: {}", kind);
                    return Err(self.error(child));
                }
            }
        }

        Ok(TypeDef {
            name,
            ty: ty.ok_or_else(|| self.error(node))?,
            generics,
            is_exported,
        })
    }

    fn parse_binding(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<Binding> {
        let cursor = &mut node.walk();
        let mut name = String::new();
        let mut ty = None;
        let mut parsing_type = false;
        for child in node.children(cursor) {
            let kind = child.kind();
            match kind {
                "name" | "vararg" => name.push_str(self.extract_text(child)),
                ":" => {
                    parsing_type = true;
                }
                "comment" => self.parse_comment_tr(child),
                _ if parsing_type => {
                    ty = Some(self.parse_type(child, unp)?);
                    parsing_type = false;
                }
                _ => return Err(self.error(child)),
            }
        }
        Ok(Binding { name, ty })
    }

    fn parse_type(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<Type> {
        let kind = node.kind();
        match kind {
            "singleton" => {
                let singleton = node.child(0).ok_or_else(|| self.error(node))?;
                let kind = singleton.kind();
                match kind {
                    "nil" => Ok(Type::Nil),
                    "boolean" => Ok(Type::Bool(self.parse_bool(singleton)?)),
                    "string" => Ok(Type::String(self.extract_text(singleton).to_string())),
                    _ => Err(self.error(singleton)),
                }
            }
            "dyntype" => Ok(Type::TypeOf(
                self.parse_expr(node.child(2).ok_or_else(|| self.error(node))?, unp)?,
            )),
            "wraptype" => Ok(self.parse_type(node.child(1).ok_or_else(|| self.error(node))?, unp)?),
            "untype" => Ok(Type::Optional(Box::new(
                self.parse_type(node.child(0).ok_or_else(|| self.error(node))?, unp)?,
            ))),
            "namedtype" | "name" | "generic" => Ok(Type::Named(self.parse_named_type(node, unp)?)),
            "bintype" => self.parse_bintype(node, unp),
            "tbtype" => Ok(Type::Table(self.parse_table_type(node, unp)?)),
            "packtype" => Ok(Type::Pack(Box::new(
                self.parse_type(node.child(0).ok_or_else(|| self.error(node))?, unp)?,
            ))),
            "fntype" => Ok(Type::Function(Box::new(
                self.parse_function_type(node, unp)?,
            ))),
            _ => Err(self.error(node)),
        }
    }

    fn parse_function_type(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<FunctionType> {
        let cursor = &mut node.walk();
        let mut params = TypeList::default();
        let mut generics = vec![];
        let mut ret_ty = None;

        enum State {
            Init,
            Generics,
            Return,
        }

        let mut state = State::Init;

        for child in node.children(cursor) {
            let kind = child.kind();
            match (kind, &state) {
                ("paramlist", State::Init) => {
                    params = self.parse_type_list(child, unp)?;
                    state = State::Return;
                }
                ("<", State::Init) => {
                    state = State::Generics;
                }
                ("generic" | "genpack", State::Generics) => {
                    generics.push(self.parse_generic_param(child)?);
                }
                (",", State::Generics) => {}
                (">", State::Generics) => {
                    state = State::Init;
                }
                (_, State::Return) => {
                    ret_ty = Some(self.parse_type_or_pack(child, unp)?);
                }
                ("comment", _) => self.parse_comment_tr(child),
                _ => todo!("parse_function_type: {}", kind),
            }
        }

        Ok(FunctionType {
            params,
            generics,
            ret_ty: ret_ty.ok_or_else(|| self.error(node))?,
        })
    }

    fn parse_table_type(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<TableType> {
        let cursor = &mut node.walk();
        let mut props = Vec::new();

        enum State<'s, 'ts> {
            Init,
            PropType(&'s str),
            IndexerTypeNext,
            IndexerType(tree_sitter::Node<'ts>),
        }

        let mut state = State::Init;

        for child in node.children(cursor) {
            let kind = child.kind();
            match (kind, &state) {
                ("{" | "}" | "," | ";", State::Init) => {}
                ("array", State::Init) => {
                    props.push(TableProp::Array(self.parse_type(
                        child.child(0).ok_or_else(|| self.error(child))?,
                        unp,
                    )?));
                    break; // can't allow more props after 1 array
                }
                ("name", State::Init) => state = State::PropType(self.extract_text(child)),
                (":", State::PropType(_) | State::IndexerType(_)) => {}
                ("[", State::Init) => state = State::IndexerTypeNext,
                ("]", State::IndexerType(_)) => {}
                ("comment", _) => self.parse_comment_tr(child),
                (_, State::PropType(key)) => {
                    props.push(TableProp::Prop {
                        key: key.to_string(),
                        value: self.parse_type(child, unp)?,
                    });
                    state = State::Init;
                }
                (_, State::IndexerTypeNext) => state = State::IndexerType(child),
                (_, State::IndexerType(indexer)) => {
                    props.push(TableProp::Indexer {
                        key: self.parse_type(*indexer, unp)?,
                        value: self.parse_type(child, unp)?,
                    });
                    state = State::Init;
                }
                _ => return Err(self.error(child)),
            }
        }
        Ok(TableType { props })
    }

    fn parse_bintype(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<Type> {
        let cursor = &mut node.walk();

        let mut lhs = None;
        let mut rhs = None;

        enum State {
            Left,
            Op,
            Right,
        }

        enum TypeKind {
            Union,
            Intersection,
        }

        let mut ty_kind = None;
        let mut state = State::Left;

        for child in node.children(cursor) {
            let kind = child.kind();
            match (kind, &state) {
                ("comment", _) => self.parse_comment_tr(child),
                (_, State::Left) => {
                    lhs = Some(self.parse_type(child, unp)?);
                    state = State::Op;
                }
                ("|", State::Op) => {
                    ty_kind = Some(TypeKind::Union);
                    state = State::Right;
                }
                ("&", State::Op) => {
                    ty_kind = Some(TypeKind::Intersection);
                    state = State::Right;
                }
                (_, State::Right) => {
                    rhs = Some(self.parse_type(child, unp)?);
                    state = State::Op;
                }
                _ => return Err(self.error(child)),
            }
        }

        let left = lhs.ok_or_else(|| self.error(node))?;
        let right = rhs.ok_or_else(|| self.error(node))?;

        match ty_kind.ok_or_else(|| self.error(node))? {
            TypeKind::Union => Ok(Type::Union(Box::new(UnionType { left, right }))),
            TypeKind::Intersection => Ok(Type::Intersection(Box::new(IntersectionType {
                left,
                right,
            }))),
        }
    }

    fn parse_named_type(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<NamedType> {
        let cursor = &mut node.walk();
        let mut table = None;
        let mut name = String::new();
        let mut params = Vec::new();

        enum State {
            Name,
            Field,
            Params,
        }

        let mut state = State::Name;

        for child in node.children(cursor) {
            let kind = child.kind();
            match (kind, &state) {
                ("name", State::Name) => {
                    name.push_str(self.extract_text(child));
                    state = State::Params
                }
                (".", State::Name | State::Params) => {
                    table = Some(name);
                    name = String::new();
                    state = State::Field;
                }
                ("name", State::Field) => {
                    name.push_str(self.extract_text(child));
                    state = State::Params;
                }
                ("<", State::Params) => {
                    params = self.parse_type_params(child, unp)?;
                    break;
                }
                ("comment", _) => self.parse_comment_tr(child),
                _ => return Err(self.error(child)),
            }
        }
        Ok(NamedType {
            table,
            name,
            params,
        })
    }

    // NOTE: the node has to be on a '<' token or first param
    fn parse_type_params(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<Vec<TypeOrPack>> {
        let mut params = Vec::new();
        for child in NodeSiblingIter::new(node) {
            let kind = child.kind();
            match kind {
                "<" | ">" | "," => {}
                "typeparam" => {
                    let ty = self.parse_type_or_pack(child.child(0).unwrap(), unp)?;
                    params.push(ty);
                }
                "comment" => self.parse_comment_tr(child),
                _ => return Err(self.error(child)),
            }
        }
        Ok(params)
    }

    fn parse_type_or_pack(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<TypeOrPack> {
        let kind = node.kind();
        match kind {
            // TODO:
            // ugly, i know, this is temporary, we will match all types, and leave typepack as
            // the only one
            "name" | "generic" | "singleton" | "packtype" | "namedtype" | "wraptype"
            | "dyntype" | "fntype" | "tbtype" | "bintype" | "untype" => {
                Ok(TypeOrPack::Type(self.parse_type(node, unp)?))
            }
            "typepack" => Ok(TypeOrPack::Pack(self.parse_type_pack(node, unp)?)),
            _ => {
                eprintln!(
                    "TODO: parse_type_or_pack {}: {}",
                    kind,
                    self.extract_text(node)
                );
                Err(self.error(node))
            }
        }
    }

    fn parse_type_pack(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<TypePack> {
        let cursor = &mut node.walk();

        enum State {
            Init,
            List,
        }

        let mut state = State::Init;

        for child in node.children(cursor) {
            let kind = child.kind();
            match (kind, &state) {
                ("variadic", State::Init) => {
                    return Ok(TypePack::Variadic(self.parse_type(
                        child.child(1).ok_or_else(|| self.error(child))?,
                        unp,
                    )?))
                }
                ("genpack", State::Init) => {
                    return Ok(TypePack::Generic(self.parse_type(
                        child.child(0).ok_or_else(|| self.error(child))?,
                        unp,
                    )?))
                }
                ("(", State::Init) => {
                    state = State::List;
                }
                (")", State::List) => {
                    // edge case: empty list
                    return Ok(TypePack::Listed(TypeList::default()));
                }
                (_, State::List) => return Ok(TypePack::Listed(self.parse_type_list(child, unp)?)),
                ("comment", _) => self.parse_comment_tr(child),
                _ => return Err(self.error(child)),
            }
        }
        Err(self.error(node))
    }

    fn parse_type_list(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<TypeList> {
        let cursor = &mut node.walk();
        let mut typelist = TypeList::default();
        for child in node.children(cursor) {
            let kind = child.kind();
            match kind {
                "variadic" => {
                    typelist.vararg = Some(
                        self.parse_type(child.child(1).ok_or_else(|| self.error(child))?, unp)?,
                    )
                }
                "(" | ")" | "->" | "," => {}
                "comment" => self.parse_comment_tr(child),
                _ => typelist.types.push(self.parse_type(child, unp)?),
            }
        }
        Ok(typelist)
    }

    fn parse_expr(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<Expr> {
        // grows the call stack if needed, such that large expressions can be parsed
        // without stack overflow
        stacker::maybe_grow(32 * 1024, 1024 * 1024, || {
            let kind = node.kind();
            match kind {
                "nil" => Ok(Expr::Nil),
                "number" => Ok(Expr::Number(self.parse_number(node)?)),
                "string" => Ok(Expr::String(self.extract_text(node).to_string())),
                "boolean" => Ok(Expr::Bool(self.parse_bool(node)?)),
                "anon_fn" => Ok(Expr::Function(Box::new(self.parse_function_body(
                    node.child(1).ok_or_else(|| self.error(node))?,
                    unp,
                )?))),
                "vararg" => Ok(Expr::VarArg),
                "exp_wrap" => {
                    Ok(self.parse_expr(node.child(1).ok_or_else(|| self.error(node))?, unp)?)
                }
                "var" => Ok(Expr::Var(self.parse_var(node, unp)?)),
                "binexp" => Ok(Expr::BinOp(Box::new(self.parse_binop(node, unp)?))),
                "unexp" => Ok(Expr::UnOp(Box::new(self.parse_unop(node, unp)?))),
                "call_stmt" => Ok(Expr::Call(Box::new(self.parse_call(node, unp)?))),
                "table" => Ok(Expr::TableConstructor(
                    self.parse_tableconstructor(node, unp)?,
                )),
                "ifexp" => Ok(Expr::IfElseExpr(Box::new(self.parse_ifelseexp(node, unp)?))),
                "string_interp" => Ok(Expr::StringInterp(self.parse_string_interp(node, unp)?)),
                "cast" => Ok(Expr::TypeAssertion(Box::new(
                    self.parse_type_assertion(node, unp)?,
                ))),
                _ => Err(self.error(node)),
            }
        })
    }

    fn parse_number(&mut self, node: tree_sitter::Node<'ts>) -> Result<f64> {
        let text = self.extract_text(node).replace('_', "");
        let num = if let Some(strip) = text.strip_prefix("0x") {
            (i64::from_str_radix(strip, 16).map_err(|_| self.error(node))?) as f64
        } else if let Some(strip) = text.strip_prefix("0b") {
            (i64::from_str_radix(strip, 2).map_err(|_| self.error(node))?) as f64
        } else {
            text.parse().map_err(|_| self.error(node))?
        };
        Ok(num)
    }

    fn parse_bool(&mut self, node: tree_sitter::Node<'ts>) -> Result<bool> {
        // true is 4 bytes, false is 5 bytes. hack! no need for text :^)
        let start = node.start_byte();
        let end = node.end_byte();

        // lets just make sure lol (only in debug mode)
        debug_assert!(matches!(&self.text[start..end], "true" | "false"));
        Ok(end - start == 4)
    }

    fn parse_var(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<Var> {
        let cursor = &mut node.walk();

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
                (".", State::FieldExprWasVar) => {}
                (".", State::TableExprWasVar) => state = State::FieldExprWasVar,
                ("[", State::TableExprWasVar) => {}
                ("[", State::FieldExprWasVar) => state = State::TableExprWasVar,
                ("comment", _) => self.parse_comment_tr(child),
                (_, State::TableExprWasVar) => {
                    var = Some(Var::TableAccess(Box::new(TableAccess {
                        expr: Expr::Var(var.take().unwrap()),
                        index: self.parse_expr(child, unp)?,
                    })));
                    state = State::Init;
                }
                (_, State::FieldExprWasVar) => {
                    var = Some(Var::FieldAccess(Box::new(FieldAccess {
                        expr: Expr::Var(var.take().unwrap()),
                        field: self.extract_text(child).to_string(),
                    })));
                    state = State::Init;
                }
                (_, State::TableExpr(expr_node)) => {
                    var = Some(Var::TableAccess(Box::new(TableAccess {
                        expr: self.parse_expr(*expr_node, unp)?,
                        index: self.parse_expr(child, unp)?,
                    })));
                    state = State::Init;
                }
                (_, State::FieldExpr(expr_node)) => {
                    var = Some(Var::FieldAccess(Box::new(FieldAccess {
                        expr: self.parse_expr(*expr_node, unp)?,
                        field: self.extract_text(child).to_string(),
                    })));
                    state = State::Init;
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
                _ if child.kind() == "comment" => self.parse_comment_tr(child),
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
                _ if child.kind() == "comment" => self.parse_comment_tr(child),
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

    fn parse_field(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<TableField> {
        let cursor = &mut node.walk();

        enum FieldState<'s, 'ts> {
            Init,
            ParsingExplicit(&'s str),
            ParsingArrayNext,
            ParsingArray(tree_sitter::Node<'ts>),
        }

        let mut state = FieldState::Init;

        for child in node.children(cursor) {
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
    }

    fn parse_tableconstructor(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<TableConstructor> {
        let cursor = &mut node.walk();
        let mut fields = Vec::new();

        for child in node.children(cursor) {
            let kind = child.kind();
            match kind {
                "}" | "{" => {}
                "fieldlist" => {
                    let cursor = &mut child.walk();
                    for field in child.children(cursor) {
                        let kind = field.kind();
                        match kind {
                            "field" => fields.push(self.parse_field(field, unp)?),
                            "," | ";" => {}
                            "comment" => self.parse_comment_tr(field),
                            _ => return Err(self.error(node)),
                        }
                    }
                }
                "comment" => self.parse_comment_tr(child),
                _ => return Err(self.error(node)),
            }
        }

        Ok(TableConstructor { fields })
    }

    fn parse_ifelseexp(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<IfElseExpr> {
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
                ("comment", _) => self.parse_comment_tr(child),
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
        Ok(IfElseExpr {
            cond: cond.ok_or_else(|| self.error(node))?,
            if_expr: if_exp.ok_or_else(|| self.error(node))?,
            else_expr: else_expr.ok_or_else(|| self.error(node))?,
            else_if_exprs,
        })
    }

    fn parse_string_interp(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<StringInterp> {
        let cursor = &mut node.walk();
        let mut parts = Vec::new();

        for child in node.children(cursor) {
            let kind = child.kind();
            match kind {
                "interp_start" | "interp_end" => {}
                "interp_content" => {
                    let txt = self.extract_text(child);
                    parts.push(StringInterpPart::String(txt.to_string()));
                }
                "interp_exp" => {
                    let expr =
                        self.parse_expr(child.child(1).ok_or_else(|| self.error(child))?, unp)?;
                    parts.push(StringInterpPart::Expr(expr));
                }
                _ => return Err(self.error(node)),
            }
        }
        Ok(StringInterp { parts })
    }

    fn parse_type_assertion(
        &mut self,
        node: tree_sitter::Node<'ts>,
        unp: &mut UnparsedStmts<'ts>,
    ) -> Result<TypeAssertion> {
        let cursor = &mut node.walk();
        enum State<'ts> {
            Expr,
            TypeNext(tree_sitter::Node<'ts>),
        }

        let mut state = State::Expr;

        for child in node.children(cursor) {
            let kind = child.kind();
            match (kind, &state) {
                ("comment", _) => self.parse_comment_tr(child),
                (_, State::Expr) => {
                    state = State::TypeNext(child);
                }
                ("::", State::TypeNext(_)) => {}
                (_, State::TypeNext(expr_node)) => {
                    let expr = self.parse_expr(*expr_node, unp)?;
                    let ty = self.parse_type(child, unp)?;
                    return Ok(TypeAssertion { expr, ty });
                }
            }
        }
        Err(self.error(node))
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
                    StmtStatus::Some(
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "a".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::Number(1.0)]
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Local(Local {
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
                        }),
                        vec![]
                    )
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
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "a".to_string(),
                            ty: None
                        }],
                        init: vec![]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn parse_string_double() {
        assert_parse!(
            "local a = \"hello\"",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "a".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::String("\"hello\"".to_string())]
                    }),
                    vec![]
                )],
            }
        );
    }

    #[test]
    fn parse_string_single() {
        assert_parse!(
            "local a = 'hello'",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "a".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::String("'hello'".to_string())]
                    }),
                    vec![]
                )],
            }
        );
    }

    #[test]
    fn parse_string_single_double() {
        assert_parse!(
            "local a = '\"hello\"'",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "a".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::String("'\"hello\"'".to_string())]
                    }),
                    vec![]
                )],
            }
        );
    }

    #[test]
    fn parse_string_double_single() {
        assert_parse!(
            "local a = \"'hello'\"",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "a".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::String("\"'hello'\"".to_string())]
                    }),
                    vec![]
                )],
            }
        );
    }

    #[test]
    fn parse_string_square() {
        assert_parse!(
            "local a = [[hello]]",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "a".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::String("[[hello]]".to_string())]
                    }),
                    vec![]
                )],
            }
        );
    }

    #[test]
    fn parse_string_square_double() {
        assert_parse!(
            "local a = [[\"hello\"]]",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "a".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::String("[[\"hello\"]]".to_string())]
                    }),
                    vec![]
                )],
            }
        );
    }

    #[test]
    fn parse_string_square_single() {
        assert_parse!(
            "local a = [['hello']]",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "a".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::String("[['hello']]".to_string())]
                    }),
                    vec![]
                )],
            }
        );
    }

    #[test]
    fn parse_nil() {
        assert_parse!(
            "local a = nil",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "a".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::Nil]
                    }),
                    vec![]
                )],
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
                    StmtStatus::Some(
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "a".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::Bool(true)]
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "b".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::Bool(false)]
                        }),
                        vec![]
                    )
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
                    StmtStatus::Some(
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "a".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::Number(1.0)]
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "b".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::Var(Var::Name("a".to_string()))]
                        }),
                        vec![]
                    )
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
                    StmtStatus::Some(
                        Stmt::FunctionDef(FunctionDef {
                            name: "f".to_string(),
                            body: FunctionBody {
                                ret_ty: None,
                                block: Block { stmt_ptrs: vec![2] },
                                generics: vec![],
                                vararg: None,
                                params: vec![Binding {
                                    name: "n".to_string(),
                                    ty: None
                                }],
                            },
                            table: vec![],
                            is_method: false
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Call(Call {
                            func: Expr::Var(Var::Name("f".to_string())),
                            args: CallArgs::Exprs(vec![Expr::Number(3.0)]),
                            method: None
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Call(Call {
                            func: Expr::Var(Var::Name("print".to_string())),
                            args: CallArgs::Exprs(vec![Expr::Var(Var::Name("n".to_string()))]),
                            method: None
                        }),
                        vec![]
                    )
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
                    StmtStatus::Some(
                        Stmt::FunctionDef(FunctionDef {
                            name: "f".to_string(),
                            body: FunctionBody {
                                ret_ty: None,
                                block: Block { stmt_ptrs: vec![2] },
                                vararg: None,
                                generics: vec![],
                                params: vec![Binding {
                                    name: "n".to_string(),
                                    ty: None
                                }],
                            },
                            table: vec![],
                            is_method: false
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "res".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::Call(Box::new(Call {
                                func: Expr::Var(Var::Name("f".to_string())),
                                args: CallArgs::Exprs(vec![Expr::Number(3.0)]),
                                method: None
                            }))]
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Call(Call {
                            func: Expr::Var(Var::Name("print".to_string())),
                            args: CallArgs::Exprs(vec![Expr::Var(Var::Name("n".to_string()))]),
                            method: None
                        }),
                        vec![]
                    )
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
                    StmtStatus::Some(
                        Stmt::FunctionDef(FunctionDef {
                            name: "f".to_string(),
                            body: FunctionBody {
                                ret_ty: None,
                                vararg: None,
                                block: Block { stmt_ptrs: vec![2] },
                                generics: vec![],
                                params: vec![Binding {
                                    name: "n".to_string(),
                                    ty: None
                                }],
                            },
                            table: vec![],
                            is_method: false
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Local(Local {
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
                                    method: None,
                                    args: CallArgs::Exprs(vec![Expr::Number(3.0)])
                                })),
                                Expr::Call(Box::new(Call {
                                    func: Expr::Var(Var::Name("f".to_string())),
                                    args: CallArgs::Exprs(vec![Expr::Number(4.0)]),
                                    method: None
                                }))
                            ]
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Call(Call {
                            func: Expr::Var(Var::Name("print".to_string())),
                            args: CallArgs::Exprs(vec![Expr::Var(Var::Name("n".to_string()))]),
                            method: None
                        }),
                        vec![]
                    )
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
                    StmtStatus::Some(
                        Stmt::FunctionDef(FunctionDef {
                            name: "f".to_string(),
                            body: FunctionBody {
                                ret_ty: None,
                                vararg: None,
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
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Call(Call {
                            func: Expr::Var(Var::Name("f".to_string())),
                            args: CallArgs::Exprs(vec![Expr::Number(3.0), Expr::Number(4.0)]),
                            method: None
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Call(Call {
                            func: Expr::Var(Var::Name("print".to_string())),
                            args: CallArgs::Exprs(vec![Expr::Var(Var::Name("a".to_string()))]),
                            method: None
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Call(Call {
                            func: Expr::Var(Var::Name("print".to_string())),
                            args: CallArgs::Exprs(vec![Expr::Var(Var::Name("b".to_string()))]),
                            method: None
                        }),
                        vec![]
                    )
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
                    StmtStatus::Some(
                        Stmt::FunctionDef(FunctionDef {
                            name: "f".to_string(),
                            body: FunctionBody {
                                generics: vec![],
                                ret_ty: None,
                                vararg: None,
                                block: Block { stmt_ptrs: vec![1] },
                                params: vec![Binding {
                                    name: "n".to_string(),
                                    ty: None
                                }],
                            },
                            table: vec!["t".to_string()],
                            is_method: false
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Call(Call {
                            func: Expr::Var(Var::Name("print".to_string())),
                            args: CallArgs::Exprs(vec![Expr::Var(Var::Name("n".to_string()))]),
                            method: None
                        }),
                        vec![]
                    )
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
                    StmtStatus::Some(
                        Stmt::FunctionDef(FunctionDef {
                            name: "f".to_string(),
                            body: FunctionBody {
                                generics: vec![],
                                ret_ty: None,
                                vararg: None,
                                block: Block { stmt_ptrs: vec![1] },
                                params: vec![Binding {
                                    name: "n".to_string(),
                                    ty: None
                                }],
                            },
                            table: vec!["t", "a"].iter().map(|s| s.to_string()).collect(),
                            is_method: false
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Call(Call {
                            func: Expr::Var(Var::Name("print".to_string())),
                            args: CallArgs::Exprs(vec![Expr::Var(Var::Name("n".to_string()))]),
                            method: None
                        }),
                        vec![]
                    )
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
                    StmtStatus::Some(
                        Stmt::FunctionDef(FunctionDef {
                            name: "f".to_string(),
                            body: FunctionBody {
                                generics: vec![],
                                vararg: None,
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
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Call(Call {
                            func: Expr::Var(Var::Name("print".to_string())),
                            args: CallArgs::Exprs(vec![Expr::Var(Var::Name("n".to_string()))]),
                            method: None
                        }),
                        vec![]
                    )
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
                    StmtStatus::Some(
                        Stmt::FunctionDef(FunctionDef {
                            name: "f".to_string(),
                            body: FunctionBody {
                                ret_ty: None,
                                vararg: None,
                                generics: vec![],
                                block: Block { stmt_ptrs: vec![1] },
                                params: vec![Binding {
                                    name: "n".to_string(),
                                    ty: None
                                }],
                            },
                            table: vec!["t".to_string()],
                            is_method: true
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Call(Call {
                            func: Expr::Var(Var::Name("print".to_string())),
                            args: CallArgs::Exprs(vec![Expr::Var(Var::Name("self".to_string()))]),
                            method: None
                        }),
                        vec![]
                    )
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
                    StmtStatus::Some(
                        Stmt::LocalFunctionDef(LocalFunctionDef {
                            name: "f".to_string(),
                            body: FunctionBody {
                                ret_ty: None,
                                vararg: None,
                                generics: vec![],
                                block: Block {
                                    stmt_ptrs: vec![2], // print(n)
                                },
                                params: vec![Binding {
                                    name: "n".to_string(),
                                    ty: None
                                }],
                            }
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Call(Call {
                            func: Expr::Var(Var::Name("f".to_string())),
                            args: CallArgs::Exprs(vec![Expr::Number(3.0)]),
                            method: None
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Call(Call {
                            func: Expr::Var(Var::Name("print".to_string())),
                            args: CallArgs::Exprs(vec![Expr::Var(Var::Name("n".to_string()))]),
                            method: None
                        }),
                        vec![]
                    )
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
                stmts: vec![StmtStatus::Some(
                    Stmt::FunctionDef(FunctionDef {
                        name: "f".to_string(),
                        body: FunctionBody {
                            ret_ty: None,
                            vararg: None,
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
                    }),
                    vec![]
                ),]
            }
        );
    }

    #[test]
    fn parse_fn_empty_body() {
        assert_parse!(
            "function f() end",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::FunctionDef(FunctionDef {
                        name: "f".to_string(),
                        body: FunctionBody {
                            ret_ty: None,
                            vararg: None,
                            generics: vec![],
                            block: Block { stmt_ptrs: vec![] },
                            params: vec![]
                        },
                        table: vec![],
                        is_method: false
                    }),
                    vec![]
                )]
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
                    StmtStatus::Some(
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "plus".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::BinOp(Box::new(BinOp {
                                lhs: Expr::Number(1.0),
                                op: BinOpKind::Add,
                                rhs: Expr::Number(2.0)
                            })),]
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "minus".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::BinOp(Box::new(BinOp {
                                lhs: Expr::Number(1.0),
                                op: BinOpKind::Sub,
                                rhs: Expr::Number(2.0)
                            })),]
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "mul".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::BinOp(Box::new(BinOp {
                                lhs: Expr::Number(1.0),
                                op: BinOpKind::Mul,
                                rhs: Expr::Number(2.0)
                            })),]
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "div".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::BinOp(Box::new(BinOp {
                                lhs: Expr::Number(1.0),
                                op: BinOpKind::Div,
                                rhs: Expr::Number(2.0)
                            })),]
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "mod".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::BinOp(Box::new(BinOp {
                                lhs: Expr::Number(1.0),
                                op: BinOpKind::Mod,
                                rhs: Expr::Number(2.0)
                            })),]
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "pow".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::BinOp(Box::new(BinOp {
                                lhs: Expr::Number(1.0),
                                op: BinOpKind::Pow,
                                rhs: Expr::Number(2.0)
                            })),]
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "concat".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::BinOp(Box::new(BinOp {
                                lhs: Expr::String("\"a\"".to_string()),
                                op: BinOpKind::Concat,
                                rhs: Expr::String("\"b\"".to_string())
                            })),]
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "eq".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::BinOp(Box::new(BinOp {
                                lhs: Expr::Number(1.0),
                                op: BinOpKind::Eq,
                                rhs: Expr::Number(2.0)
                            })),]
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "neq".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::BinOp(Box::new(BinOp {
                                lhs: Expr::Number(1.0),
                                op: BinOpKind::Ne,
                                rhs: Expr::Number(2.0)
                            })),]
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "lt".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::BinOp(Box::new(BinOp {
                                lhs: Expr::Number(1.0),
                                op: BinOpKind::Lt,
                                rhs: Expr::Number(2.0)
                            })),]
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "gt".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::BinOp(Box::new(BinOp {
                                lhs: Expr::Number(1.0),
                                op: BinOpKind::Gt,
                                rhs: Expr::Number(2.0)
                            })),]
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "leq".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::BinOp(Box::new(BinOp {
                                lhs: Expr::Number(1.0),
                                op: BinOpKind::Le,
                                rhs: Expr::Number(2.0)
                            })),]
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "geq".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::BinOp(Box::new(BinOp {
                                lhs: Expr::Number(1.0),
                                op: BinOpKind::Ge,
                                rhs: Expr::Number(2.0)
                            })),]
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "and".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::BinOp(Box::new(BinOp {
                                lhs: Expr::Bool(true),
                                op: BinOpKind::And,
                                rhs: Expr::Bool(false)
                            })),]
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "or".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::BinOp(Box::new(BinOp {
                                lhs: Expr::Bool(true),
                                op: BinOpKind::Or,
                                rhs: Expr::Bool(false)
                            })),]
                        }),
                        vec![]
                    ),
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
                    StmtStatus::Some(
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "not".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::UnOp(Box::new(UnOp {
                                op: UnOpKind::Not,
                                expr: Expr::Bool(true)
                            })),]
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "neg".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::UnOp(Box::new(UnOp {
                                op: UnOpKind::Neg,
                                expr: Expr::Number(1.0)
                            })),]
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "neg_wrap".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::UnOp(Box::new(UnOp {
                                op: UnOpKind::Neg,
                                expr: Expr::Number(1.0)
                            })),]
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "len".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::UnOp(Box::new(UnOp {
                                op: UnOpKind::Len,
                                expr: Expr::String("\"aaaaa\"".to_string())
                            })),]
                        }),
                        vec![]
                    ),
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
                    StmtStatus::Some(
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "guinea_pig".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::Number(1.0)]
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::CompOp(CompOp {
                            lhs: Var::Name("guinea_pig".to_string()),
                            op: CompOpKind::Add,
                            rhs: Box::new(Expr::Number(1.0))
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::CompOp(CompOp {
                            lhs: Var::Name("guinea_pig".to_string()),
                            op: CompOpKind::Sub,
                            rhs: Box::new(Expr::Number(1.0))
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::CompOp(CompOp {
                            lhs: Var::Name("guinea_pig".to_string()),
                            op: CompOpKind::Mul,
                            rhs: Box::new(Expr::Number(1.0))
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::CompOp(CompOp {
                            lhs: Var::Name("guinea_pig".to_string()),
                            op: CompOpKind::Div,
                            rhs: Box::new(Expr::Number(1.0))
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::CompOp(CompOp {
                            lhs: Var::Name("guinea_pig".to_string()),
                            op: CompOpKind::Mod,
                            rhs: Box::new(Expr::Number(1.0))
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::CompOp(CompOp {
                            lhs: Var::Name("guinea_pig".to_string()),
                            op: CompOpKind::Pow,
                            rhs: Box::new(Expr::Number(1.0))
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::CompOp(CompOp {
                            lhs: Var::Name("guinea_pig".to_string()),
                            op: CompOpKind::Concat,
                            rhs: Box::new(Expr::Number(1.0))
                        }),
                        vec![]
                    ),
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
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "cray_cray".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::BinOp(Box::new(BinOp {
                            lhs: Expr::BinOp(Box::new(BinOp {
                                lhs: Expr::BinOp(Box::new(BinOp {
                                    lhs: Expr::Number(1.0),
                                    op: BinOpKind::Add,
                                    rhs: Expr::BinOp(Box::new(BinOp {
                                        lhs: Expr::Number(2.0),
                                        op: BinOpKind::Mul,
                                        rhs: Expr::Number(3.0)
                                    })),
                                })),
                                op: BinOpKind::Sub,
                                rhs: Expr::Number(1.0)
                            })),
                            op: BinOpKind::Div,
                            rhs: Expr::Number(4.0)
                        })),]
                    }),
                    vec![]
                ),]
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
                    StmtStatus::Some(
                        Stmt::FunctionDef(FunctionDef {
                            name: "a".to_string(),
                            table: vec![],
                            is_method: false,
                            body: FunctionBody {
                                ret_ty: None,
                                vararg: None,
                                generics: vec![],
                                block: Block {
                                    stmt_ptrs: vec![1, 2, 3]
                                },
                                params: vec![Binding {
                                    name: "n".to_string(),
                                    ty: None
                                }],
                            },
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(Stmt::Continue(Continue {}), vec![]),
                    StmtStatus::Some(Stmt::Break(Break {}), vec![]),
                    StmtStatus::Some(Stmt::Return(Return { exprs: vec![] }), vec![]),
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
                    StmtStatus::Some(
                        Stmt::FunctionDef(FunctionDef {
                            name: "a".to_string(),
                            table: vec![],
                            is_method: false,
                            body: FunctionBody {
                                ret_ty: None,
                                generics: vec![],
                                vararg: Some(None),
                                block: Block { stmt_ptrs: vec![] },
                                params: vec![],
                            },
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "e".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::VarArg]
                        }),
                        vec![]
                    )
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
                    StmtStatus::Some(
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "a".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::Number(1.0)]
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Assign(Assign {
                            vars: vec![Var::Name("a".to_string())],
                            exprs: vec![Expr::Bool(false)]
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Assign(Assign {
                            vars: vec![Var::Name("a".to_string()), Var::Name("b".to_string())],
                            exprs: vec![Expr::Number(1.0), Expr::Number(2.0)]
                        }),
                        vec![]
                    ),
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
                    StmtStatus::Some(
                        Stmt::Do(Do {
                            block: Block { stmt_ptrs: vec![1] }
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Call(Call {
                            func: Expr::Var(Var::Name("print".to_string())),
                            args: CallArgs::Exprs(vec![Expr::Number(1.0)]),
                            method: None
                        }),
                        vec![]
                    )
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
                    StmtStatus::Some(
                        Stmt::While(While {
                            cond: Expr::Bool(true),
                            block: Block { stmt_ptrs: vec![1] }
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Call(Call {
                            func: Expr::Var(Var::Name("print".to_string())),
                            args: CallArgs::Exprs(vec![Expr::Number(1.0)]),
                            method: None
                        }),
                        vec![]
                    )
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
                    StmtStatus::Some(
                        Stmt::Repeat(Repeat {
                            block: Block { stmt_ptrs: vec![1] },
                            cond: Expr::Bool(true)
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Call(Call {
                            func: Expr::Var(Var::Name("print".to_string())),
                            args: CallArgs::Exprs(vec![Expr::Number(1.0)]),
                            method: None
                        }),
                        vec![]
                    )
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
                    StmtStatus::Some(
                        Stmt::If(If {
                            cond: Expr::Bool(true),
                            block: Block { stmt_ptrs: vec![1] },
                            else_if_blocks: vec![],
                            else_block: None
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Call(Call {
                            func: Expr::Var(Var::Name("print".to_string())),
                            args: CallArgs::Exprs(vec![Expr::Number(1.0)]),
                            method: None
                        }),
                        vec![]
                    )
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
                    StmtStatus::Some(
                        Stmt::If(If {
                            cond: Expr::Bool(true),
                            block: Block { stmt_ptrs: vec![1] },
                            else_if_blocks: vec![],
                            else_block: Some(Block { stmt_ptrs: vec![2] })
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Call(Call {
                            func: Expr::Var(Var::Name("print".to_string())),
                            args: CallArgs::Exprs(vec![Expr::Number(1.0)]),
                            method: None
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Call(Call {
                            func: Expr::Var(Var::Name("print".to_string())),
                            args: CallArgs::Exprs(vec![Expr::Number(2.0)]),
                            method: None
                        }),
                        vec![]
                    )
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
                    StmtStatus::Some(
                        Stmt::If(If {
                            cond: Expr::Bool(true),
                            block: Block { stmt_ptrs: vec![1] },
                            else_if_blocks: vec![
                                (Expr::Bool(false), Block { stmt_ptrs: vec![2] },)
                            ],
                            else_block: None
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Call(Call {
                            func: Expr::Var(Var::Name("print".to_string())),
                            args: CallArgs::Exprs(vec![Expr::Number(1.0)]),
                            method: None
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Call(Call {
                            func: Expr::Var(Var::Name("print".to_string())),
                            args: CallArgs::Exprs(vec![Expr::Number(2.0)]),
                            method: None
                        }),
                        vec![]
                    )
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
                    StmtStatus::Some(
                        Stmt::If(If {
                            cond: Expr::Bool(true),
                            block: Block { stmt_ptrs: vec![1] },
                            else_if_blocks: vec![
                                (Expr::Bool(false), Block { stmt_ptrs: vec![2] },)
                            ],
                            else_block: Some(Block { stmt_ptrs: vec![3] })
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Call(Call {
                            func: Expr::Var(Var::Name("print".to_string())),
                            args: CallArgs::Exprs(vec![Expr::Number(1.0)]),
                            method: None
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Call(Call {
                            func: Expr::Var(Var::Name("print".to_string())),
                            args: CallArgs::Exprs(vec![Expr::Number(2.0)]),
                            method: None
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Call(Call {
                            func: Expr::Var(Var::Name("print".to_string())),
                            args: CallArgs::Exprs(vec![Expr::Number(3.0)]),
                            method: None
                        }),
                        vec![]
                    )
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
                    StmtStatus::Some(
                        Stmt::For(For {
                            start: Expr::Number(1.0),
                            end: Expr::Number(10.0),
                            step: None,
                            block: Block { stmt_ptrs: vec![1] },
                            var: Binding {
                                name: "i".to_string(),
                                ty: None
                            }
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Call(Call {
                            func: Expr::Var(Var::Name("print".to_string())),
                            args: CallArgs::Exprs(vec![Expr::Var(Var::Name("i".to_string()))]),
                            method: None
                        }),
                        vec![]
                    )
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
                    StmtStatus::Some(
                        Stmt::For(For {
                            start: Expr::Number(1.0),
                            end: Expr::Number(10.0),
                            step: Some(Expr::Number(2.0)),
                            block: Block { stmt_ptrs: vec![1] },
                            var: Binding {
                                name: "i".to_string(),
                                ty: None
                            }
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Call(Call {
                            func: Expr::Var(Var::Name("print".to_string())),
                            args: CallArgs::Exprs(vec![Expr::Var(Var::Name("i".to_string()))]),
                            method: None
                        }),
                        vec![]
                    )
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
                    StmtStatus::Some(
                        Stmt::ForIn(ForIn {
                            block: Block { stmt_ptrs: vec![1] },
                            vars: vec![Binding {
                                name: "i".to_string(),
                                ty: None
                            }],
                            exprs: vec![Expr::Var(Var::Name("a".to_string()))]
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Call(Call {
                            func: Expr::Var(Var::Name("print".to_string())),
                            args: CallArgs::Exprs(vec![Expr::Var(Var::Name("i".to_string()))]),
                            method: None
                        }),
                        vec![]
                    )
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
                    StmtStatus::Some(
                        Stmt::ForIn(ForIn {
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
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Call(Call {
                            func: Expr::Var(Var::Name("print".to_string())),
                            args: CallArgs::Exprs(vec![
                                Expr::Var(Var::Name("i".to_string())),
                                Expr::Var(Var::Name("v".to_string()))
                            ]),
                            method: None
                        }),
                        vec![]
                    )
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
                    StmtStatus::Some(
                        Stmt::ForIn(ForIn {
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
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Call(Call {
                            func: Expr::Var(Var::Name("print".to_string())),
                            args: CallArgs::Exprs(vec![
                                Expr::Var(Var::Name("i".to_string())),
                                Expr::Var(Var::Name("v".to_string()))
                            ]),
                            method: None
                        }),
                        vec![]
                    )
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
                    StmtStatus::Some(
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "f".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::Function(Box::new(FunctionBody {
                                generics: vec![],
                                vararg: None,
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
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Call(Call {
                            func: Expr::Var(Var::Name("f".to_string())),
                            args: CallArgs::Exprs(vec![Expr::Number(1.0), Expr::Number(2.0)]),
                            method: None
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Return(Return {
                            exprs: vec![Expr::BinOp(Box::new(BinOp {
                                op: BinOpKind::Add,
                                lhs: Expr::Var(Var::Name("x".to_string())),
                                rhs: Expr::Var(Var::Name("y".to_string()))
                            }))]
                        }),
                        vec![]
                    )
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
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "t".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::TableConstructor(TableConstructor { fields: vec![] })]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_tableconstructor_array() {
        assert_parse!(
            "local t = {1, 2, 3}",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
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
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_tableconstructor_map() {
        assert_parse!(
            "local t = {a = 1, b = 2, c = 3}",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
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
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_tableconstructor_map_with_string_keys() {
        assert_parse!(
            "local t = {['a'] = 1, ['b'] = 2, ['c'] = 3}",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
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
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_tableconstructor_map_with_mixed_keys() {
        assert_parse!(
            "local t = {a = 1, 2, ['c'] = 3}",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
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
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_table_call() {
        assert_parse!(
            "local t = f{name = 1, 2, 3}",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
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
                            }),
                            method: None
                        }))]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_string_call() {
        assert_parse!(
            "local t = f'hello'",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "t".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::Call(Box::new(Call {
                            func: Expr::Var(Var::Name("f".to_string())),
                            args: CallArgs::String("'hello'".to_string()),
                            method: None
                        }))]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_nested_calls() {
        assert_parse!(
            "local t = (f())()()",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "t".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::Call(Box::new(Call {
                            func: Expr::Call(Box::new(Call {
                                func: Expr::Call(Box::new(Call {
                                    func: Expr::Var(Var::Name("f".to_string())),
                                    args: CallArgs::Exprs(vec![]),
                                    method: None
                                })),
                                args: CallArgs::Exprs(vec![]),
                                method: None
                            })),
                            args: CallArgs::Exprs(vec![]),
                            method: None
                        }))]
                    }),
                    vec![]
                )]
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
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::IfElseExpr(Box::new(IfElseExpr {
                            cond: Expr::Bool(true),
                            if_expr: Expr::Number(1.0),
                            else_expr: Expr::Number(2.0),
                            else_if_exprs: vec![]
                        }))]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_ifelseexp_with_else_if() {
        assert_parse!(
            "local x = if true then 1 elseif false then 2 else 3",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::IfElseExpr(Box::new(IfElseExpr {
                            cond: Expr::Bool(true),
                            if_expr: Expr::Number(1.0),
                            else_expr: Expr::Number(3.0),
                            else_if_exprs: vec![(Expr::Bool(false), Expr::Number(2.0))]
                        }))]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_ifelseexp_with_multiple_else_if() {
        assert_parse!(
            "local x = if true then 1 elseif false then 2 elseif false then 2 else 4",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::IfElseExpr(Box::new(IfElseExpr {
                            cond: Expr::Bool(true),
                            if_expr: Expr::Number(1.0),
                            else_expr: Expr::Number(4.0),
                            else_if_exprs: vec![
                                (Expr::Bool(false), Expr::Number(2.0)),
                                (Expr::Bool(false), Expr::Number(2.0))
                            ]
                        }))]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn parse_table_access() {
        assert_parse!(
            "local x = t[1]",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::Var(Var::TableAccess(Box::new(TableAccess {
                            expr: Expr::Var(Var::Name("t".to_string())),
                            index: Expr::Number(1.0)
                        })))]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn parse_table_access2() {
        assert_parse!(
            "local x = (1)[1]",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::Var(Var::TableAccess(Box::new(TableAccess {
                            expr: Expr::Number(1.0),
                            index: Expr::Number(1.0)
                        })))]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn parse_field_access1() {
        assert_parse!(
            "local x = t.x",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::Var(Var::FieldAccess(Box::new(FieldAccess {
                            expr: Expr::Var(Var::Name("t".to_string())),
                            field: "x".to_string()
                        })))]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn parse_field_access2() {
        assert_parse!(
            "local x = (1).x",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::Var(Var::FieldAccess(Box::new(FieldAccess {
                            expr: Expr::Number(1.0),
                            field: "x".to_string()
                        })))]
                    }),
                    vec![]
                )]
            }
        );
    }

    // string interpolation
    // 1. local x = `hello {name}`
    // 2. local x = `hello {name} {age}`
    // 3. local x = `hello {1 + 2}`
    // 4. local combos = {2, 7, 1, 8, 5}
    //    print(`The lock combination is {table.concat(combos)}. Again, {table.concat(combos, ", ")}.`)

    #[test]
    fn test_string_interpolation() {
        assert_parse!(
            "local x = `hello {name}`",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::StringInterp(StringInterp {
                            parts: vec![
                                StringInterpPart::String("hello ".to_string()),
                                StringInterpPart::Expr(Expr::Var(Var::Name("name".to_string())))
                            ]
                        })]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_string_interpolation2() {
        assert_parse!(
            "local x = `hello {name} {age}`",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::StringInterp(StringInterp {
                            parts: vec![
                                StringInterpPart::String("hello ".to_string()),
                                StringInterpPart::Expr(Expr::Var(Var::Name("name".to_string()))),
                                StringInterpPart::String(" ".to_string()),
                                StringInterpPart::Expr(Expr::Var(Var::Name("age".to_string())))
                            ]
                        })]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_string_interpolation3() {
        assert_parse!(
            "local x = `hello {1 + 2}`",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::StringInterp(StringInterp {
                            parts: vec![
                                StringInterpPart::String("hello ".to_string()),
                                StringInterpPart::Expr(Expr::BinOp(Box::new(BinOp {
                                    op: BinOpKind::Add,
                                    lhs: Expr::Number(1.0),
                                    rhs: Expr::Number(2.0)
                                })))
                            ]
                        })]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_string_interpolation4() {
        assert_parse!(
            r#"
            local combos = {2, 7, 1, 8, 5}
            print(`The lock combination is {table.concat(combos)}. Again, {table.concat(combos, ", ")}.`)
            "#,
            Chunk {
                block: Block {
                    stmt_ptrs: vec![0, 1]
                },
                stmts: vec![
                    StmtStatus::Some(
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "combos".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::TableConstructor(TableConstructor {
                                fields: vec![
                                    TableField::ImplicitKey(Expr::Number(2.0)),
                                    TableField::ImplicitKey(Expr::Number(7.0)),
                                    TableField::ImplicitKey(Expr::Number(1.0)),
                                    TableField::ImplicitKey(Expr::Number(8.0)),
                                    TableField::ImplicitKey(Expr::Number(5.0))
                                ]
                            })]
                        }),
                        vec![]
                    ),
                    StmtStatus::Some(
                        Stmt::Call(Call {
                            func: Expr::Var(Var::Name("print".to_string())),
                            args: CallArgs::Exprs(vec![Expr::StringInterp(StringInterp {
                                parts: vec![
                                    StringInterpPart::String(
                                        "The lock combination is ".to_string()
                                    ),
                                    StringInterpPart::Expr(Expr::Call(Box::new(Call {
                                        func: Expr::Var(Var::FieldAccess(Box::new(FieldAccess {
                                            expr: Expr::Var(Var::Name("table".to_string())),
                                            field: "concat".to_string()
                                        }))),
                                        args: CallArgs::Exprs(vec![Expr::Var(Var::Name(
                                            "combos".to_string()
                                        ))]),
                                        method: None
                                    }))),
                                    StringInterpPart::String(". Again, ".to_string()),
                                    StringInterpPart::Expr(Expr::Call(Box::new(Call {
                                        func: Expr::Var(Var::FieldAccess(Box::new(FieldAccess {
                                            expr: Expr::Var(Var::Name("table".to_string())),
                                            field: "concat".to_string()
                                        }))),
                                        args: CallArgs::Exprs(vec![
                                            Expr::Var(Var::Name("combos".to_string())),
                                            Expr::String("\", \"".to_string())
                                        ]),
                                        method: None
                                    }))),
                                    StringInterpPart::String(".".to_string())
                                ]
                            })]),
                            method: None
                        }),
                        vec![]
                    )
                ]
            }
        );
    }

    #[test]
    fn regression_1() {
        assert_parse!(
            "local u = workspace.TempStorage.HelperGUI",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "u".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::Var(Var::FieldAccess(Box::new(FieldAccess {
                            expr: Expr::Var(Var::FieldAccess(Box::new(FieldAccess {
                                expr: Expr::Var(Var::Name("workspace".to_string())),
                                field: "TempStorage".to_string()
                            }))),
                            field: "HelperGUI".to_string()
                        })))],
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn regression_2() {
        assert_parse!(
            "workspace.TempStorage.HelperGUI:Destroy()",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Call(Call {
                        func: Expr::Var(Var::FieldAccess(Box::new(FieldAccess {
                            expr: Expr::Var(Var::FieldAccess(Box::new(FieldAccess {
                                expr: Expr::Var(Var::Name("workspace".to_string())),
                                field: "TempStorage".to_string()
                            }))),
                            field: "HelperGUI".to_string()
                        }))),
                        args: CallArgs::Exprs(vec![]),
                        method: Some("Destroy".to_string())
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn regression_3() {
        assert_parse!(
            "local u = workspace.TempStorage.HelperGUI.aaa.bbb",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "u".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::Var(Var::FieldAccess(Box::new(FieldAccess {
                            expr: Expr::Var(Var::FieldAccess(Box::new(FieldAccess {
                                expr: Expr::Var(Var::FieldAccess(Box::new(FieldAccess {
                                    expr: Expr::Var(Var::FieldAccess(Box::new(FieldAccess {
                                        expr: Expr::Var(Var::Name("workspace".to_string())),
                                        field: "TempStorage".to_string()
                                    }))),
                                    field: "HelperGUI".to_string()
                                }))),
                                field: "aaa".to_string()
                            }))),
                            field: "bbb".to_string()
                        })))],
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn regression_4() {
        assert_parse!(
            "local x = t[1][2][3][4]",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::Var(Var::TableAccess(Box::new(TableAccess {
                            expr: Expr::Var(Var::TableAccess(Box::new(TableAccess {
                                expr: Expr::Var(Var::TableAccess(Box::new(TableAccess {
                                    expr: Expr::Var(Var::TableAccess(Box::new(TableAccess {
                                        expr: Expr::Var(Var::Name("t".to_string())),
                                        index: Expr::Number(1.0)
                                    }))),
                                    index: Expr::Number(2.0)
                                }))),
                                index: Expr::Number(3.0)
                            }))),
                            index: Expr::Number(4.0)
                        })))]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn regression_5() {
        assert_parse!(
            "local x = t.a[2][1].b.c[3]",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::Var(Var::TableAccess(Box::new(TableAccess {
                            expr: Expr::Var(Var::FieldAccess(Box::new(FieldAccess {
                                expr: Expr::Var(Var::FieldAccess(Box::new(FieldAccess {
                                    expr: Expr::Var(Var::TableAccess(Box::new(TableAccess {
                                        expr: Expr::Var(Var::TableAccess(Box::new(TableAccess {
                                            expr: Expr::Var(Var::FieldAccess(Box::new(
                                                FieldAccess {
                                                    expr: Expr::Var(Var::Name("t".to_string())),
                                                    field: "a".to_string()
                                                }
                                            ))),
                                            index: Expr::Number(2.0)
                                        }))),
                                        index: Expr::Number(1.0)
                                    }))),
                                    field: "b".to_string()
                                }))),
                                field: "c".to_string()
                            }))),
                            index: Expr::Number(3.0)
                        })))],
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn func_ret_type() {
        assert_parse!(
            "function f(): number end",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::FunctionDef(FunctionDef {
                        name: "f".to_string(),
                        table: vec![],
                        is_method: false,
                        body: FunctionBody {
                            params: vec![],
                            vararg: None,
                            generics: vec![],
                            ret_ty: Some(TypeOrPack::Type(Type::Named(NamedType {
                                table: None,
                                name: "number".to_string(),
                                params: vec![]
                            }))),
                            block: Block { stmt_ptrs: vec![] }
                        },
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn comment_1() {
        assert_parse!(
            "Configurations.MaxApplesAtATime = 5 -- the maximum amount of apples that can be on the ground at one time",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![
                    StmtStatus::Some(Stmt::Assign(
                            Assign { vars: vec![Var::FieldAccess(Box::new(FieldAccess { expr: Expr::Var(Var::Name("Configurations".to_string())), field: "MaxApplesAtATime".to_string() }))], exprs: vec![Expr::Number(5.0)] }
                    ), vec![
                        Comment::Trailing("-- the maximum amount of apples that can be on the ground at one time".to_string())
                    ])
                ]
            }
        );
    }

    #[test]
    fn comment_2() {
        assert_parse!(
            "-- Hola amigos\nlocal x = 3",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::Number(3.0)]
                    }),
                    vec![Comment::Leading("-- Hola amigos".to_string())]
                )]
            }
        );
    }

    #[test]
    fn comment_3() {
        assert_parse!(
            "local x = 3 -- Hola amigos",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::Number(3.0)]
                    }),
                    vec![Comment::Trailing("-- Hola amigos".to_string())]
                )]
            }
        );
    }

    #[test]
    fn comment_4() {
        assert_parse!(
            "local x = 3 -- Hola amigos\nlocal y = 4",
            Chunk {
                block: Block {
                    stmt_ptrs: vec![0, 1]
                },
                stmts: vec![
                    StmtStatus::Some(
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "x".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::Number(3.0)]
                        }),
                        vec![Comment::Trailing("-- Hola amigos".to_string())]
                    ),
                    StmtStatus::Some(
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "y".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::Number(4.0)]
                        }),
                        vec![]
                    )
                ]
            }
        );
    }

    #[test]
    fn comment_5() {
        assert_parse!(
            "local x = 3 -- Hola amigos\n-- Adios amigos",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::Number(3.0)]
                    }),
                    vec![
                        Comment::Trailing("-- Hola amigos".to_string()),
                        Comment::Trailing("-- Adios amigos".to_string())
                    ]
                ),]
            }
        );
    }

    #[test]
    fn nicely_commented_function() {
        assert_parse!(
            r#"
            --[[
                This function does something cool.
                It's really cool.
            ]]
            -- Totally cool, it works really well.
            function f()
                --[[
                    This is a comment inside the function.
                    It's really cool.
                ]]
                -- Totally cool, it works really well.
                return 3
            end
            "#,
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![
                    StmtStatus::Some(
                        Stmt::FunctionDef(FunctionDef {
                            name: "f".to_string(),
                            table: vec![],
                            is_method: false,
                            body: FunctionBody {
                                params: vec![],
                                vararg: None,
                                generics: vec![],
                                ret_ty: None,
                                block: Block { stmt_ptrs: vec![1] }
                            },
                        }),
                        vec![
                            Comment::Leading("--[[\n                This function does something cool.\n                It's really cool.\n            ]]".to_string()),
                            Comment::Leading("-- Totally cool, it works really well.".to_string()),
                            Comment::Trailing("--[[\n                    This is a comment inside the function.\n                    It's really cool.\n                ]]".to_string()),
                            Comment::Trailing("-- Totally cool, it works really well.".to_string())
                        ]
                    ),
                    StmtStatus::Some(
                        Stmt::Return(Return {
                            exprs: vec![Expr::Number(3.0)]
                        }),
                        vec![]
                    )
                ]
            }
        );
    }

    #[test]
    fn regression_6() {
        assert_parse!(
            "return 1, 2",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Return(Return {
                        exprs: vec![Expr::Number(1.0), Expr::Number(2.0),]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn regression_7() {
        assert_parse!(
            "local character = Tool.Parent;",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "character".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::Var(Var::FieldAccess(Box::new(FieldAccess {
                            expr: Expr::Var(Var::Name("Tool".to_string())),
                            field: "Parent".to_string()
                        })))]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn regression_8() {
        assert_parse!(
            "return 1, 2;",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Return(Return {
                        exprs: vec![Expr::Number(1.0), Expr::Number(2.0),]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn regression_9() {
        assert_parse!(
            "return chr == 0x5F",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Return(Return {
                        exprs: vec![Expr::BinOp(Box::new(BinOp {
                            lhs: Expr::Var(Var::Name("chr".to_string())),
                            op: BinOpKind::Eq,
                            rhs: Expr::Number(0x5F as f64)
                        }))]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn regression_10() {
        assert_parse!(
            "return chr == 0b100",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Return(Return {
                        exprs: vec![Expr::BinOp(Box::new(BinOp {
                            lhs: Expr::Var(Var::Name("chr".to_string())),
                            op: BinOpKind::Eq,
                            rhs: Expr::Number(0b100 as f64)
                        }))]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn regression_11() {
        assert_parse!(
            "return chr == 100_000_000",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Return(Return {
                        exprs: vec![Expr::BinOp(Box::new(BinOp {
                            lhs: Expr::Var(Var::Name("chr".to_string())),
                            op: BinOpKind::Eq,
                            rhs: Expr::Number(100_000_000.0)
                        }))]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn regression_12() {
        assert_parse!(
            r#"
 if not cacheSize then
elseif cacheSize < 0 or cacheSize ~= cacheSize then
  error("cache size cannot be a negative number or a NaN", 2);
end
            "#,
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![
                    StmtStatus::Some(
                        Stmt::If(If {
                            cond: Expr::UnOp(Box::new(UnOp {
                                op: UnOpKind::Not,
                                expr: Expr::Var(Var::Name("cacheSize".to_string())),
                            })),
                            block: Block { stmt_ptrs: vec![] },
                            else_if_blocks: vec![(
                                Expr::BinOp(Box::new(BinOp {
                                    lhs: Expr::BinOp(Box::new(BinOp {
                                        lhs: Expr::Var(Var::Name("cacheSize".to_string())),
                                        op: BinOpKind::Lt,
                                        rhs: Expr::Number(0.0),
                                    })),
                                    op: BinOpKind::Or,
                                    rhs: Expr::BinOp(Box::new(BinOp {
                                        lhs: Expr::Var(Var::Name("cacheSize".to_string())),
                                        op: BinOpKind::Ne,
                                        rhs: Expr::Var(Var::Name("cacheSize".to_string())),
                                    })),
                                })),
                                Block { stmt_ptrs: vec![1] },
                            )],
                            else_block: None,
                        }),
                        vec![],
                    ),
                    StmtStatus::Some(
                        Stmt::Call(Call {
                            func: Expr::Var(Var::Name("error".to_string())),
                            args: CallArgs::Exprs(vec![
                                Expr::String(
                                    "\"cache size cannot be a negative number or a NaN\""
                                        .to_string()
                                ),
                                Expr::Number(2.0),
                            ]),
                            method: None,
                        }),
                        vec![],
                    ),
                ],
            }
        );
    }

    #[test]
    fn lone_comment() {
        assert_parse!(
            "-- lonely, i'm so lonely...",
            Chunk {
                block: Block::default(),
                stmts: vec![]
            }
        );
    }

    #[test]
    fn test_type_singletons() {
        assert_parse!(
            "local x: nil = nil\nlocal y: true = true\nlocal z: false = false\nlocal str: \"string\" = \"string\"",
            Chunk {
                block: Block { stmt_ptrs: vec![0,1,2,3] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: Some(Type::Nil)
                        }],
                        init: vec![Expr::Nil]
                    }),
                    vec![]
                ),
                StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "y".to_string(),
                            ty: Some(Type::Bool(true))
                        }],
                        init: vec![Expr::Bool(true)]
                    }),
                    vec![]
                ),
                StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "z".to_string(),
                            ty: Some(Type::Bool(false))
                        }],
                        init: vec![Expr::Bool(false)]
                    }),
                    vec![]
                ),
                StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "str".to_string(),
                            ty: Some(Type::String("\"string\"".to_string()))
                        }],
                        init: vec![Expr::String("\"string\"".to_string())]
                    }),
                    vec![]
                )
                ]
            }
        );
    }

    #[test]
    fn test_type_named_simple() {
        assert_parse!(
            "local x: number = 1",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: Some(Type::Named(NamedType {
                                table: None,
                                name: "number".to_string(),
                                params: vec![]
                            }))
                        }],
                        init: vec![Expr::Number(1.0)]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_type_named_field() {
        assert_parse!(
            "local x: a.b = nil",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: Some(Type::Named(NamedType {
                                table: Some("a".to_string()),
                                name: "b".to_string(),
                                params: vec![]
                            }))
                        }],
                        init: vec![Expr::Nil]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_type_named_params() {
        assert_parse!(
            "local x: a.b<c, d> = nil",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: Some(Type::Named(NamedType {
                                table: Some("a".to_string()),
                                name: "b".to_string(),
                                params: vec![
                                    TypeOrPack::Type(Type::Named(NamedType {
                                        table: None,
                                        name: "c".to_string(),
                                        params: vec![]
                                    })),
                                    TypeOrPack::Type(Type::Named(NamedType {
                                        table: None,
                                        name: "d".to_string(),
                                        params: vec![]
                                    }))
                                ]
                            }))
                        }],
                        init: vec![Expr::Nil]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_type_typeof() {
        assert_parse!(
            "local x: typeof(y) = nil",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: Some(Type::TypeOf(Expr::Var(Var::Name("y".to_string()))))
                        }],
                        init: vec![Expr::Nil]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_type_wrap() {
        assert_parse!(
            "local x: (number) = nil",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: Some(Type::Named(NamedType {
                                table: None,
                                name: "number".to_string(),
                                params: vec![]
                            }))
                        }],
                        init: vec![Expr::Nil]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_type_optional() {
        assert_parse!(
            "local x: number? = nil",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: Some(Type::Optional(Box::new(Type::Named(NamedType {
                                table: None,
                                name: "number".to_string(),
                                params: vec![]
                            }))))
                        }],
                        init: vec![Expr::Nil]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_type_optional_nest() {
        assert_parse!(
            "local x: number?? = nil",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: Some(Type::Optional(Box::new(Type::Optional(Box::new(
                                Type::Named(NamedType {
                                    table: None,
                                    name: "number".to_string(),
                                    params: vec![]
                                })
                            )))))
                        }],
                        init: vec![Expr::Nil]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_type_union() {
        assert_parse!(
            "local x: number | string | boolean = nil",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: Some(Type::Union(Box::new(UnionType {
                                left: Type::Union(Box::new(UnionType {
                                    left: Type::Named(NamedType {
                                        table: None,
                                        name: "number".to_string(),
                                        params: vec![]
                                    }),
                                    right: Type::Named(NamedType {
                                        table: None,
                                        name: "string".to_string(),
                                        params: vec![]
                                    })
                                })),
                                right: Type::Named(NamedType {
                                    table: None,
                                    name: "boolean".to_string(),
                                    params: vec![]
                                })
                            })))
                        }],
                        init: vec![Expr::Nil]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_type_intersection() {
        assert_parse!(
            "local x: number & string & boolean = nil",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: Some(Type::Intersection(Box::new(IntersectionType {
                                left: Type::Intersection(Box::new(IntersectionType {
                                    left: Type::Named(NamedType {
                                        table: None,
                                        name: "number".to_string(),
                                        params: vec![]
                                    }),
                                    right: Type::Named(NamedType {
                                        table: None,
                                        name: "string".to_string(),
                                        params: vec![]
                                    }),
                                })),
                                right: Type::Named(NamedType {
                                    table: None,
                                    name: "boolean".to_string(),
                                    params: vec![]
                                })
                            })))
                        }],
                        init: vec![Expr::Nil]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_type_mix_union_intersection() {
        assert_parse!(
            "local x: number | string & boolean = nil",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: Some(Type::Union(Box::new(UnionType {
                                left: Type::Named(NamedType {
                                    table: None,
                                    name: "number".to_string(),
                                    params: vec![]
                                }),
                                right: Type::Intersection(Box::new(IntersectionType {
                                    left: Type::Named(NamedType {
                                        table: None,
                                        name: "string".to_string(),
                                        params: vec![]
                                    }),
                                    right: Type::Named(NamedType {
                                        table: None,
                                        name: "boolean".to_string(),
                                        params: vec![]
                                    })
                                }))
                            })))
                        }],
                        init: vec![Expr::Nil]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_type_mix_union_intersection_with_optional() {
        assert_parse!(
            "local x: number | string & boolean? = nil",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: Some(Type::Union(Box::new(UnionType {
                                left: Type::Named(NamedType {
                                    table: None,
                                    name: "number".to_string(),
                                    params: vec![]
                                }),
                                right: Type::Intersection(Box::new(IntersectionType {
                                    left: Type::Named(NamedType {
                                        table: None,
                                        name: "string".to_string(),
                                        params: vec![]
                                    }),
                                    right: Type::Optional(Box::new(Type::Named(NamedType {
                                        table: None,
                                        name: "boolean".to_string(),
                                        params: vec![]
                                    })))
                                }))
                            })))
                        }],
                        init: vec![Expr::Nil]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_type_assertion() {
        assert_parse!(
            "local x = 1::number",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::TypeAssertion(Box::new(TypeAssertion {
                            expr: Expr::Number(1.0),
                            ty: Type::Named(NamedType {
                                table: None,
                                name: "number".to_string(),
                                params: vec![]
                            })
                        }))]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_type_table_prop() {
        assert_parse!(
            "local x: {a: number, b: string} = nil",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: Some(Type::Table(TableType {
                                props: vec![
                                    TableProp::Prop {
                                        key: "a".to_string(),
                                        value: Type::Named(NamedType {
                                            table: None,
                                            name: "number".to_string(),
                                            params: vec![]
                                        })
                                    },
                                    TableProp::Prop {
                                        key: "b".to_string(),
                                        value: Type::Named(NamedType {
                                            table: None,
                                            name: "string".to_string(),
                                            params: vec![]
                                        })
                                    }
                                ]
                            }))
                        }],
                        init: vec![Expr::Nil]
                    }),
                    vec![]
                )]
            }
        );
    }
    #[test]
    fn test_type_table_indexer() {
        assert_parse!(
            "local x: {[number]: number, [nil]: string} = nil",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: Some(Type::Table(TableType {
                                props: vec![
                                    TableProp::Indexer {
                                        key: Type::Named(NamedType {
                                            table: None,
                                            name: "number".to_string(),
                                            params: vec![]
                                        }),
                                        value: Type::Named(NamedType {
                                            table: None,
                                            name: "number".to_string(),
                                            params: vec![]
                                        })
                                    },
                                    TableProp::Indexer {
                                        key: Type::Nil,
                                        value: Type::Named(NamedType {
                                            table: None,
                                            name: "string".to_string(),
                                            params: vec![]
                                        })
                                    }
                                ]
                            }))
                        }],
                        init: vec![Expr::Nil]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_type_table_empty() {
        assert_parse!(
            "local x: {} = nil",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: Some(Type::Table(TableType { props: vec![] }))
                        }],
                        init: vec![Expr::Nil]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_function_typed() {
        assert_parse!(
            "function f(x: number, y: string): number end",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::FunctionDef(FunctionDef {
                        name: "f".to_string(),
                        body: FunctionBody {
                            ret_ty: Some(TypeOrPack::Type(Type::Named(NamedType {
                                table: None,
                                name: "number".to_string(),
                                params: vec![]
                            }))),
                            block: Block { stmt_ptrs: vec![] },
                            generics: vec![],
                            vararg: None,
                            params: vec![
                                Binding {
                                    name: "x".to_string(),
                                    ty: Some(Type::Named(NamedType {
                                        table: None,
                                        name: "number".to_string(),
                                        params: vec![]
                                    }))
                                },
                                Binding {
                                    name: "y".to_string(),
                                    ty: Some(Type::Named(NamedType {
                                        table: None,
                                        name: "string".to_string(),
                                        params: vec![]
                                    }))
                                }
                            ],
                        },
                        table: vec![],
                        is_method: false
                    }),
                    vec![]
                ),]
            }
        );
    }

    #[test]
    fn test_function_typed_generic() {
        assert_parse!(
            "function f<T, R>(x: T, y: T): R end",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::FunctionDef(FunctionDef {
                        name: "f".to_string(),
                        body: FunctionBody {
                            ret_ty: Some(TypeOrPack::Type(Type::Named(NamedType {
                                table: None,
                                name: "R".to_string(),
                                params: vec![]
                            }))),
                            block: Block { stmt_ptrs: vec![] },
                            generics: vec![
                                GenericParam::Name("T".to_string()),
                                GenericParam::Name("R".to_string()),
                            ],
                            vararg: None,
                            params: vec![
                                Binding {
                                    name: "x".to_string(),
                                    ty: Some(Type::Named(NamedType {
                                        table: None,
                                        name: "T".to_string(),
                                        params: vec![]
                                    }))
                                },
                                Binding {
                                    name: "y".to_string(),
                                    ty: Some(Type::Named(NamedType {
                                        table: None,
                                        name: "T".to_string(),
                                        params: vec![]
                                    }))
                                }
                            ],
                        },
                        table: vec![],
                        is_method: false
                    }),
                    vec![]
                ),]
            }
        );
    }

    #[test]
    fn test_function_typed_generic_pack() {
        assert_parse!(
            "function f<T, R, U...>(x: T, ...: U): R end",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::FunctionDef(FunctionDef {
                        name: "f".to_string(),
                        body: FunctionBody {
                            ret_ty: Some(TypeOrPack::Type(Type::Named(NamedType {
                                table: None,
                                name: "R".to_string(),
                                params: vec![]
                            }))),
                            block: Block { stmt_ptrs: vec![] },
                            generics: vec![
                                GenericParam::Name("T".to_string()),
                                GenericParam::Name("R".to_string()),
                                GenericParam::Pack("U".to_string()),
                            ],
                            vararg: Some(Some(Type::Named(NamedType {
                                table: None,
                                name: "U".to_string(),
                                params: vec![]
                            }))),
                            params: vec![Binding {
                                name: "x".to_string(),
                                ty: Some(Type::Named(NamedType {
                                    table: None,
                                    name: "T".to_string(),
                                    params: vec![]
                                }))
                            },],
                        },
                        table: vec![],
                        is_method: false
                    }),
                    vec![]
                ),]
            }
        );
    }

    #[test]
    fn test_function_typed_generic_pack_unpacked() {
        assert_parse!(
            "function f<T, R, U...>(x: T, ...: U...): R end",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::FunctionDef(FunctionDef {
                        name: "f".to_string(),
                        body: FunctionBody {
                            ret_ty: Some(TypeOrPack::Type(Type::Named(NamedType {
                                table: None,
                                name: "R".to_string(),
                                params: vec![]
                            }))),
                            block: Block { stmt_ptrs: vec![] },
                            generics: vec![
                                GenericParam::Name("T".to_string()),
                                GenericParam::Name("R".to_string()),
                                GenericParam::Pack("U".to_string()),
                            ],
                            vararg: Some(Some(Type::Pack(Box::new(Type::Named(NamedType {
                                table: None,
                                name: "U".to_string(),
                                params: vec![]
                            }))))),
                            params: vec![Binding {
                                name: "x".to_string(),
                                ty: Some(Type::Named(NamedType {
                                    table: None,
                                    name: "T".to_string(),
                                    params: vec![]
                                }))
                            },],
                        },
                        table: vec![],
                        is_method: false
                    }),
                    vec![]
                ),]
            }
        );
    }

    #[test]
    fn test_generic_param() {
        assert_parse!(
            "local x: A<number, string> = nil",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: Some(Type::Named(NamedType {
                                table: None,
                                name: "A".to_string(),
                                params: vec![
                                    TypeOrPack::Type(Type::Named(NamedType {
                                        table: None,
                                        name: "number".to_string(),
                                        params: vec![]
                                    })),
                                    TypeOrPack::Type(Type::Named(NamedType {
                                        table: None,
                                        name: "string".to_string(),
                                        params: vec![]
                                    })),
                                ]
                            }))
                        }],
                        init: vec![Expr::Nil]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_generic_param_typepack_empty() {
        assert_parse!(
            "local x: A<(), string> = nil",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: Some(Type::Named(NamedType {
                                table: None,
                                name: "A".to_string(),
                                params: vec![
                                    TypeOrPack::Pack(TypePack::Listed(TypeList {
                                        types: vec![],
                                        vararg: None
                                    })),
                                    TypeOrPack::Type(Type::Named(NamedType {
                                        table: None,
                                        name: "string".to_string(),
                                        params: vec![]
                                    })),
                                ]
                            }))
                        }],
                        init: vec![Expr::Nil]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_generic_param_typepack() {
        assert_parse!(
            "local x: A<(number, number), string> = nil",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: Some(Type::Named(NamedType {
                                table: None,
                                name: "A".to_string(),
                                params: vec![
                                    TypeOrPack::Pack(TypePack::Listed(TypeList {
                                        types: vec![
                                            Type::Named(NamedType {
                                                table: None,
                                                name: "number".to_string(),
                                                params: vec![]
                                            }),
                                            Type::Named(NamedType {
                                                table: None,
                                                name: "number".to_string(),
                                                params: vec![]
                                            })
                                        ],
                                        vararg: None
                                    })),
                                    TypeOrPack::Type(Type::Named(NamedType {
                                        table: None,
                                        name: "string".to_string(),
                                        params: vec![]
                                    })),
                                ]
                            }))
                        }],
                        init: vec![Expr::Nil]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_generic_param_typepack_variadic() {
        assert_parse!(
            "local x: A<(number, ...number), string> = nil",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: Some(Type::Named(NamedType {
                                table: None,
                                name: "A".to_string(),
                                params: vec![
                                    TypeOrPack::Pack(TypePack::Listed(TypeList {
                                        types: vec![Type::Named(NamedType {
                                            table: None,
                                            name: "number".to_string(),
                                            params: vec![]
                                        }),],
                                        vararg: Some(Type::Named(NamedType {
                                            table: None,
                                            name: "number".to_string(),
                                            params: vec![]
                                        }))
                                    })),
                                    TypeOrPack::Type(Type::Named(NamedType {
                                        table: None,
                                        name: "string".to_string(),
                                        params: vec![]
                                    })),
                                ]
                            }))
                        }],
                        init: vec![Expr::Nil]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_generic_param_variadic_typepack() {
        assert_parse!(
            "local x: A<...number, string> = nil",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: Some(Type::Named(NamedType {
                                table: None,
                                name: "A".to_string(),
                                params: vec![
                                    TypeOrPack::Pack(TypePack::Variadic(Type::Named(NamedType {
                                        table: None,
                                        name: "number".to_string(),
                                        params: vec![]
                                    }))),
                                    TypeOrPack::Type(Type::Named(NamedType {
                                        table: None,
                                        name: "string".to_string(),
                                        params: vec![]
                                    })),
                                ]
                            }))
                        }],
                        init: vec![Expr::Nil]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_generic_param_generic_typepack() {
        assert_parse!(
            "local x: A<number..., string> = nil",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: Some(Type::Named(NamedType {
                                table: None,
                                name: "A".to_string(),
                                params: vec![
                                    TypeOrPack::Pack(TypePack::Generic(Type::Named(NamedType {
                                        table: None,
                                        name: "number".to_string(),
                                        params: vec![]
                                    }))),
                                    TypeOrPack::Type(Type::Named(NamedType {
                                        table: None,
                                        name: "string".to_string(),
                                        params: vec![]
                                    })),
                                ]
                            }))
                        }],
                        init: vec![Expr::Nil]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_type_table_array() {
        assert_parse!(
            "local x: { number } = nil",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: Some(Type::Table(TableType {
                                props: vec![TableProp::Array(Type::Named(NamedType {
                                    table: None,
                                    name: "number".to_string(),
                                    params: vec![]
                                }))]
                            }))
                        }],
                        init: vec![Expr::Nil]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_type_function_simple() {
        assert_parse!(
            "local x: (string) -> number = nil",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: Some(Type::Function(Box::new(FunctionType {
                                generics: vec![],
                                params: TypeList {
                                    types: vec![Type::Named(NamedType {
                                        table: None,
                                        name: "string".to_string(),
                                        params: vec![]
                                    })],
                                    vararg: None
                                },
                                ret_ty: TypeOrPack::Type(Type::Named(NamedType {
                                    table: None,
                                    name: "number".to_string(),
                                    params: vec![]
                                }))
                            })))
                        }],
                        init: vec![Expr::Nil]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_type_function_multi_arg() {
        assert_parse!(
            "local x: (string, string, string) -> nil = nil",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: Some(Type::Function(Box::new(FunctionType {
                                generics: vec![],
                                params: TypeList {
                                    types: vec![
                                        Type::Named(NamedType {
                                            table: None,
                                            name: "string".to_string(),
                                            params: vec![]
                                        }),
                                        Type::Named(NamedType {
                                            table: None,
                                            name: "string".to_string(),
                                            params: vec![]
                                        }),
                                        Type::Named(NamedType {
                                            table: None,
                                            name: "string".to_string(),
                                            params: vec![]
                                        })
                                    ],
                                    vararg: None
                                },
                                ret_ty: TypeOrPack::Type(Type::Nil)
                            })))
                        }],
                        init: vec![Expr::Nil]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_type_function_generics() {
        assert_parse!(
            "local x: <T>(T) -> nil = nil",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: Some(Type::Function(Box::new(FunctionType {
                                generics: vec![GenericParam::Name("T".to_string())],
                                params: TypeList {
                                    types: vec![Type::Named(NamedType {
                                        table: None,
                                        name: "T".to_string(),
                                        params: vec![]
                                    })],
                                    vararg: None
                                },
                                ret_ty: TypeOrPack::Type(Type::Nil)
                            })))
                        }],
                        init: vec![Expr::Nil]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_type_function_generic_multi_with_pack() {
        assert_parse!(
            "local x: <T, U, C...>(T, U, C) -> nil = nil",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: Some(Type::Function(Box::new(FunctionType {
                                generics: vec![
                                    GenericParam::Name("T".to_string()),
                                    GenericParam::Name("U".to_string()),
                                    GenericParam::Pack("C".to_string()),
                                ],
                                params: TypeList {
                                    types: vec![
                                        Type::Named(NamedType {
                                            table: None,
                                            name: "T".to_string(),
                                            params: vec![]
                                        }),
                                        Type::Named(NamedType {
                                            table: None,
                                            name: "U".to_string(),
                                            params: vec![]
                                        }),
                                        Type::Named(NamedType {
                                            table: None,
                                            name: "C".to_string(),
                                            params: vec![]
                                        })
                                    ],
                                    vararg: None
                                },
                                ret_ty: TypeOrPack::Type(Type::Nil)
                            })))
                        }],
                        init: vec![Expr::Nil]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_type_function_generic_multi_with_pack_multi() {
        assert_parse!(
            "local x: <T, U..., C...>(T, U, C) -> nil = nil",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: Some(Type::Function(Box::new(FunctionType {
                                generics: vec![
                                    GenericParam::Name("T".to_string()),
                                    GenericParam::Pack("U".to_string()),
                                    GenericParam::Pack("C".to_string()),
                                ],
                                params: TypeList {
                                    types: vec![
                                        Type::Named(NamedType {
                                            table: None,
                                            name: "T".to_string(),
                                            params: vec![]
                                        }),
                                        Type::Named(NamedType {
                                            table: None,
                                            name: "U".to_string(),
                                            params: vec![]
                                        }),
                                        Type::Named(NamedType {
                                            table: None,
                                            name: "C".to_string(),
                                            params: vec![]
                                        })
                                    ],
                                    vararg: None
                                },
                                ret_ty: TypeOrPack::Type(Type::Nil)
                            })))
                        }],
                        init: vec![Expr::Nil]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_type_function_generic_multi_with_pack_multi_varargs() {
        assert_parse!(
            "local x: <T, U..., C...>(T, U, ...C) -> nil = nil",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: Some(Type::Function(Box::new(FunctionType {
                                generics: vec![
                                    GenericParam::Name("T".to_string()),
                                    GenericParam::Pack("U".to_string()),
                                    GenericParam::Pack("C".to_string()),
                                ],
                                params: TypeList {
                                    types: vec![
                                        Type::Named(NamedType {
                                            table: None,
                                            name: "T".to_string(),
                                            params: vec![]
                                        }),
                                        Type::Named(NamedType {
                                            table: None,
                                            name: "U".to_string(),
                                            params: vec![]
                                        }),
                                    ],
                                    vararg: Some(Type::Named(NamedType {
                                        table: None,
                                        name: "C".to_string(),
                                        params: vec![]
                                    }))
                                },
                                ret_ty: TypeOrPack::Type(Type::Nil)
                            })))
                        }],
                        init: vec![Expr::Nil]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_type_function_generic_multi_with_pack_multi_varargs_packret() {
        assert_parse!(
            "local x: <T, U..., C...>(T, U, ...C) -> (nil, ...nil) = nil",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "x".to_string(),
                            ty: Some(Type::Function(Box::new(FunctionType {
                                generics: vec![
                                    GenericParam::Name("T".to_string()),
                                    GenericParam::Pack("U".to_string()),
                                    GenericParam::Pack("C".to_string()),
                                ],
                                params: TypeList {
                                    types: vec![
                                        Type::Named(NamedType {
                                            table: None,
                                            name: "T".to_string(),
                                            params: vec![]
                                        }),
                                        Type::Named(NamedType {
                                            table: None,
                                            name: "U".to_string(),
                                            params: vec![]
                                        }),
                                    ],
                                    vararg: Some(Type::Named(NamedType {
                                        table: None,
                                        name: "C".to_string(),
                                        params: vec![]
                                    }))
                                },
                                ret_ty: TypeOrPack::Pack(TypePack::Listed(TypeList {
                                    types: vec![Type::Nil],
                                    vararg: Some(Type::Nil)
                                }))
                            })))
                        }],
                        init: vec![Expr::Nil]
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_type_def_simple() {
        assert_parse!(
            "type       MyNil         =         nil",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::TypeDef(TypeDef {
                        name: "MyNil".to_string(),
                        ty: Type::Nil,
                        generics: vec![],
                        is_exported: false
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_type_def_simple_exported() {
        assert_parse!(
            "export type MyNil = nil",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::TypeDef(TypeDef {
                        name: "MyNil".to_string(),
                        ty: Type::Nil,
                        generics: vec![],
                        is_exported: true
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_type_def_simple_exported_with_complicated_type() {
        assert_parse!(
            "export type MyNil = string | (number & boolean) | nil",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::TypeDef(TypeDef {
                        name: "MyNil".to_string(),
                        ty: Type::Union(Box::new(UnionType {
                            left: Type::Union(Box::new(UnionType {
                                left: Type::Named(NamedType {
                                    table: None,
                                    name: "string".to_string(),
                                    params: vec![]
                                }),
                                right: Type::Intersection(Box::new(IntersectionType {
                                    left: Type::Named(NamedType {
                                        table: None,
                                        name: "number".to_string(),
                                        params: vec![]
                                    }),
                                    right: Type::Named(NamedType {
                                        table: None,
                                        name: "boolean".to_string(),
                                        params: vec![]
                                    })
                                })),
                            })),
                            right: Type::Nil
                        })),
                        generics: vec![],
                        is_exported: true
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_type_def_generics_simple() {
        assert_parse!(
            "export type MyNil<T> = T | (number & boolean) | nil",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::TypeDef(TypeDef {
                        name: "MyNil".to_string(),
                        ty: Type::Union(Box::new(UnionType {
                            left: Type::Union(Box::new(UnionType {
                                left: Type::Named(NamedType {
                                    table: None,
                                    name: "T".to_string(),
                                    params: vec![]
                                }),
                                right: Type::Intersection(Box::new(IntersectionType {
                                    left: Type::Named(NamedType {
                                        table: None,
                                        name: "number".to_string(),
                                        params: vec![]
                                    }),
                                    right: Type::Named(NamedType {
                                        table: None,
                                        name: "boolean".to_string(),
                                        params: vec![]
                                    })
                                })),
                            })),
                            right: Type::Nil
                        })),
                        generics: vec![GenericDef {
                            param: GenericParam::Name("T".to_string()),
                            default: None,
                        }],
                        is_exported: true
                    }),
                    vec![]
                )]
            }
        );
    }

    #[test]
    fn test_type_def_generics_two_simple() {
        assert_parse!(
            "export type MyNil<T, U> = T | (U & boolean) | nil",
            Chunk {
                block: Block { stmt_ptrs: vec![0] },
                stmts: vec![StmtStatus::Some(
                    Stmt::TypeDef(TypeDef {
                        name: "MyNil".to_string(),
                        ty: Type::Union(Box::new(UnionType {
                            left: Type::Union(Box::new(UnionType {
                                left: Type::Named(NamedType {
                                    table: None,
                                    name: "T".to_string(),
                                    params: vec![]
                                }),
                                right: Type::Intersection(Box::new(IntersectionType {
                                    left: Type::Named(NamedType {
                                        table: None,
                                        name: "U".to_string(),
                                        params: vec![]
                                    }),
                                    right: Type::Named(NamedType {
                                        table: None,
                                        name: "boolean".to_string(),
                                        params: vec![]
                                    })
                                })),
                            })),
                            right: Type::Nil
                        })),
                        generics: vec![
                            GenericDef {
                                param: GenericParam::Name("T".to_string()),
                                default: None,
                            },
                            GenericDef {
                                param: GenericParam::Name("U".to_string()),
                                default: None,
                            }
                        ],
                        is_exported: true
                    }),
                    vec![]
                )]
            }
        );
    }

    // TODO: test the following
    // 1. type Iterator<K, V> = ({ [K]: V }, K?) -> (K?, V?)
}
