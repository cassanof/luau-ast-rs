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
    do_pretty_errors: bool,
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
    /// Constructs a new parser for the given source code.
    pub fn new(text: &'s str) -> Self {
        let ts = ts_parser();
        Self {
            ts,
            text,
            chunk: Chunk::default(),
            acc_trailing_comments: Vec::new(),
            do_pretty_errors: false,
        }
    }

    /// Enables/disables pretty errors, which means that the error message will contain
    /// the source code snippet where the error occurred, with squiggly lines
    /// under the problematic part. **Disabled by default.**
    pub fn set_pretty_errors(&mut self, pretty_errors: bool) {
        self.do_pretty_errors = pretty_errors;
    }

    //// Parses the file and returns the AST.
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

        if self.do_pretty_errors {
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
            _ => Err(self.error(node)),
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

        enum State<'ts> {
            Init,
            BoundFound(tree_sitter::Node<'ts>),
        }

        let mut state = State::Init;

        for child in node.children(cursor) {
            let kind = child.kind();
            match (kind, &state) {
                ("variadic", State::Init) => {
                    typelist.vararg = Some(
                        self.parse_type(child.child(1).ok_or_else(|| self.error(child))?, unp)?,
                    )
                }
                ("(" | ")" | "->" | ",", State::Init) => {}
                ("comment", _) => self.parse_comment_tr(child),
                ("name", State::Init) => state = State::BoundFound(child),
                (":", State::BoundFound(_)) => {}
                (_, State::BoundFound(name)) => {
                    let ty = self.parse_type(child, unp)?;
                    let btype = BoundType {
                        name: self.extract_text(*name).to_string(),
                        ty: Box::new(ty),
                    };
                    typelist.types.push(Type::Bound(btype));
                    state = State::Init;
                }
                (_, State::Init) => typelist.types.push(self.parse_type(child, unp)?),
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
