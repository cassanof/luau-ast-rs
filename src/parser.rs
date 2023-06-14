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
    input: &'s str,
}

impl<'ts, 's> Parser<'s> {
    pub fn new() -> Self {
        Self {
            ts: ts_parser(),
            input: "",
        }
    }

    pub fn parse(&mut self, text: &'s str) -> Result<Chunk> {
        self.input = text;
        let tree = self.ts.parse(self.input, None).ok_or(ParseError::TSError)?;
        let root = tree.root_node();
        let block = self.parse_block(root)?;
        Ok(Chunk { block })
    }

    fn extract_text(&self, node: tree_sitter::Node<'ts>) -> &'s str {
        let start = node.start_byte();
        let end = node.end_byte();
        &self.input[start..end]
    }

    fn error(&self, node: tree_sitter::Node<'ts>) -> ParseError {
        let start = node.start_position();
        let end = node.end_position();

        let text = self.input;

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

    fn parse_block(&self, node: tree_sitter::Node<'ts>) -> Result<Block> {
        let mut stmts = Vec::new();
        let cursor = &mut node.walk();
        for child in node.children(cursor) {
            let stmt = self.parse_stmt(child)?;
            stmts.push(stmt);
        }
        Ok(Block { stmts })
    }

    fn parse_stmt(&self, node: tree_sitter::Node<'ts>) -> Result<Stmt> {
        let kind = node.kind();
        match kind {
            "local_var_stmt" => {
                let local = self.parse_local(node)?;
                Ok(Stmt::Local(local))
            }
            // "assignment_statement" => {
            // let assign = self.parse_assign(cursor, node)?;
            // Ok(Stmt::Assign(assign))
            // }
            _ => todo!("parse_stmt: {}", kind),
        }
    }

    fn parse_binding(&self, node: tree_sitter::Node<'ts>) -> Result<Binding> {
        let cursor = &mut node.walk();
        let mut name = String::new();
        let mut parsing_type = false;
        for child in node.children(cursor) {
            let kind = child.kind();
            match kind {
                "name" => name = self.extract_text(child).to_string(),
                ":" => {
                    parsing_type = true;
                }
                _ if parsing_type => {
                    // todo!("parse_binding (type delegation): {}", kind);
                    return Err(self.error(child));
                }
                _ => todo!("parse_binding: {}", kind),
            }
        }
        Ok(Binding { name, ty: None })
    }

    fn parse_local(&self, node: tree_sitter::Node<'ts>) -> Result<Local> {
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

    fn parse_expr(&self, node: tree_sitter::Node<'ts>) -> Result<Expr> {
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
                debug_assert!(matches!(&self.input[start..end], "true" | "false"));

                Ok(Expr::Bool(end - start == 4))
            }
            // delegate to parse_var
            "var" => Ok(Expr::Var(self.parse_var(node)?)),
            _ => todo!("parse_expr: {}", kind),
        }
    }

    fn parse_var(&self, node: tree_sitter::Node<'ts>) -> Result<Var> {
        let mut name = String::new();
        let cursor = &mut node.walk();
        for child in node.children(cursor) {
            let kind = child.kind();
            match kind {
                "name" => name = self.extract_text(child).to_string(),
                _ => todo!("parse_var: {}", kind),
            }
        }
        Ok(Var::Name(name))
    }
}

impl Default for Parser<'_> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use crate::parser::*;

    macro_rules! assert_parse {
        ($source:expr, $expected:expr) => {
            let ast = Parser::new().parse($source);
            assert_eq!(ast.unwrap(), $expected);
        };
    }

    #[test]
    fn parse_locals() {
        assert_parse!(
            "local a = 1\nlocal b, c = 2, 3",
            Chunk {
                block: Block {
                    stmts: vec![
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "a".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::Number(1.0)]
                        }),
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
                        })
                    ]
                }
            }
        );
    }

    #[test]
    fn parse_string() {
        assert_parse!(
            "local a = \"hello\"",
            Chunk {
                block: Block {
                    stmts: vec![Stmt::Local(Local {
                        bindings: vec![Binding {
                            name: "a".to_string(),
                            ty: None
                        }],
                        init: vec![Expr::String("hello".to_string())]
                    })]
                }
            }
        );
    }

    #[test]
    fn parse_bools() {
        assert_parse!(
            "local a = true\nlocal b = false",
            Chunk {
                block: Block {
                    stmts: vec![
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "a".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::Bool(true)]
                        }),
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "b".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::Bool(false)]
                        })
                    ]
                }
            }
        );
    }

    #[test]
    fn parse_var() {
        assert_parse!(
            "local a = 1\nlocal b = a",
            Chunk {
                block: Block {
                    stmts: vec![
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "a".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::Number(1.0)]
                        }),
                        Stmt::Local(Local {
                            bindings: vec![Binding {
                                name: "b".to_string(),
                                ty: None
                            }],
                            init: vec![Expr::Var(Var::Name("a".to_string()))]
                        })
                    ]
                }
            }
        );
    }

    #[test]
    fn parse_fn_and_call() {
        let source = "function f()\nprint(\"hi\")\nend\nf()";
        let mut parser = ts_parser();
        let tree = parser.parse(source, None).unwrap();
        println!("{:#?}", tree.root_node().to_sexp());
    }
}
