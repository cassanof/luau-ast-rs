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
        match kind {
            "local_var_stmt" => Ok((Stmt::Local(self.parse_local(node)?), Vec::new())),
            "call_stmt" => Ok((Stmt::Call(self.parse_call(node)?), Vec::new())),
            "fn_stmt" => {
                let (func, to_be_parsed) = self.parse_function_def(node)?;
                Ok((Stmt::FunctionDef(func), to_be_parsed))
            }
            _ => todo!("parse_stmt: {}", kind),
        }
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

    fn parse_function_def(
        &mut self,
        node: tree_sitter::Node<'ts>,
    ) -> Result<(FunctionDef, UnparsedStmts<'ts>)> {
        let cursor = &mut node.walk();

        // vars to fill
        let mut name = String::new();
        let mut body = None;
        let mut to_be_parsed = Vec::new();
        let mut params = Vec::new();

        let mut parsing_params = false;
        for child in node.children(cursor) {
            let kind = child.kind();
            match kind {
                "name" => name.push_str(self.extract_text(child)),
                "(" => parsing_params = true,
                ")" => parsing_params = false,
                "param" => {
                    if parsing_params {
                        params.push(self.parse_binding(child)?);
                    } else {
                        todo!("parse_function_def (parsing params): {}", kind);
                    }
                }
                "block" => {
                    let (parsed_body, unparsed_stmts) = self.parse_block(child)?;
                    body = Some(parsed_body);
                    to_be_parsed.extend(unparsed_stmts);
                }
                "end" | "function" => {}
                _ => todo!("parse_function_def: {}", kind),
            }
        }
        Ok((
            FunctionDef {
                name,
                params,
                body: body.ok_or(self.error(node))?,
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
                            _ if parsing_args => args.push(self.parse_expr(arg)?),
                            "," => {}
                            _ => todo!("parse_call (arglist): {}", kind),
                        }
                    }
                }
                _ => todo!("parse_call: {}", kind),
            }
        }
        Ok(Call {
            func: expr.ok_or(self.error(node))?,
            args,
        })
    }

    fn parse_binding(&self, node: tree_sitter::Node<'ts>) -> Result<Binding> {
        let cursor = &mut node.walk();
        let mut name = String::new();
        let mut parsing_type = false;
        for child in node.children(cursor) {
            let kind = child.kind();
            match kind {
                "name" => name.push_str(self.extract_text(child)),
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
            // delegate to parse_var
            "var" => Ok(Expr::Var(self.parse_var(node)?)),
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
                        params: vec![Binding {
                            name: "n".to_string(),
                            ty: None
                        }],
                        body: Block {
                            stmt_ptrs: vec![2], // print(n)
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
                            op: BinOpType::Add,
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
                            op: BinOpType::Sub,
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
                            op: BinOpType::Mul,
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
                            op: BinOpType::Div,
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
                            op: BinOpType::Mod,
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
                            op: BinOpType::Pow,
                            rhs: Box::new(Expr::Number(2.0))
                        }),]
                    })),
                ]
            }
        );
    }
}
