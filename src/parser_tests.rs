use std::vec;

use crate::ast::*;
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
                        else_if_blocks: vec![(Expr::Bool(false), Block { stmt_ptrs: vec![2] },)],
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
                        else_if_blocks: vec![(Expr::Bool(false), Block { stmt_ptrs: vec![2] },)],
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
                                StringInterpPart::String("The lock combination is ".to_string()),
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
                                        expr: Expr::Var(Var::FieldAccess(Box::new(FieldAccess {
                                            expr: Expr::Var(Var::Name("t".to_string())),
                                            field: "a".to_string()
                                        }))),
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
                                "\"cache size cannot be a negative number or a NaN\"".to_string()
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
fn test_type_function_bound_param() {
    assert_parse!(
        "local x: (a: string) -> number = nil",
        Chunk {
            block: Block { stmt_ptrs: vec![0] },
            stmts: vec![StmtStatus::Some(
                Stmt::Local(Local {
                    bindings: vec![Binding {
                        name: "x".to_string(),
                        ty: Some(Type::Function(Box::new(FunctionType {
                            generics: vec![],
                            params: TypeList {
                                types: vec![Type::Bound(BoundType {
                                    name: "a".to_string(),
                                    ty: Box::new(Type::Named(NamedType {
                                        table: None,
                                        name: "string".to_string(),
                                        params: vec![]
                                    }))
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
fn test_type_function_bound_params() {
    assert_parse!(
        "local x: (a: string,b: string, c: string) -> number = nil",
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
                                    Type::Bound(BoundType {
                                        name: "a".to_string(),
                                        ty: Box::new(Type::Named(NamedType {
                                            table: None,
                                            name: "string".to_string(),
                                            params: vec![]
                                        }))
                                    }),
                                    Type::Bound(BoundType {
                                        name: "b".to_string(),
                                        ty: Box::new(Type::Named(NamedType {
                                            table: None,
                                            name: "string".to_string(),
                                            params: vec![]
                                        }))
                                    }),
                                    Type::Bound(BoundType {
                                        name: "c".to_string(),
                                        ty: Box::new(Type::Named(NamedType {
                                            table: None,
                                            name: "string".to_string(),
                                            params: vec![]
                                        }))
                                    }),
                                ],
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
fn test_type_function_bound_params_some_unbound() {
    assert_parse!(
        "local x: (a: string, string, c: string) -> number = nil",
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
                                    Type::Bound(BoundType {
                                        name: "a".to_string(),
                                        ty: Box::new(Type::Named(NamedType {
                                            table: None,
                                            name: "string".to_string(),
                                            params: vec![]
                                        }))
                                    }),
                                    Type::Named(NamedType {
                                        table: None,
                                        name: "string".to_string(),
                                        params: vec![]
                                    }),
                                    Type::Bound(BoundType {
                                        name: "c".to_string(),
                                        ty: Box::new(Type::Named(NamedType {
                                            table: None,
                                            name: "string".to_string(),
                                            params: vec![]
                                        }))
                                    }),
                                ],
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

#[test]
fn test_type_def_generics_two_variadic() {
    assert_parse!(
        "export type MyNil<T, U..., C...> = T | (U & boolean) | nil",
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
                            param: GenericParam::Pack("U".to_string()),
                            default: None,
                        },
                        GenericDef {
                            param: GenericParam::Pack("C".to_string()),
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

#[test]
fn test_type_def_generics_default() {
    assert_parse!(
        "export type MyNil<T = string> = T | (number & boolean) | nil",
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
                        default: Some(TypeOrPack::Type(Type::Named(NamedType {
                            table: None,
                            name: "string".to_string(),
                            params: vec![]
                        }))),
                    }],
                    is_exported: true
                }),
                vec![]
            )]
        }
    );
}

#[test]
fn test_type_def_generics_two_variadic_defaults() {
    assert_parse!(
        "export type MyNil<T = nil, U... = ...nil, C... = ...nil> = T | (U & boolean) | nil",
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
                            default: Some(TypeOrPack::Type(Type::Nil)),
                        },
                        GenericDef {
                            param: GenericParam::Pack("U".to_string()),
                            default: Some(TypeOrPack::Pack(TypePack::Variadic(Type::Nil))),
                        },
                        GenericDef {
                            param: GenericParam::Pack("C".to_string()),
                            default: Some(TypeOrPack::Pack(TypePack::Variadic(Type::Nil))),
                        }
                    ],
                    is_exported: true
                }),
                vec![]
            )]
        }
    );
}

// type Iterator<K, V> = ({ [K]: V }, K?) -> (K?, V?)
#[test]
fn test_type_def_generics_iterator() {
    assert_parse!(
        "type Iterator<K, V> = ({ [K]: V }, K?) -> (K?, V?)",
        Chunk {
            block: Block { stmt_ptrs: vec![0] },
            stmts: vec![StmtStatus::Some(
                Stmt::TypeDef(TypeDef {
                    name: "Iterator".to_string(),
                    ty: Type::Function(Box::new(FunctionType {
                        generics: vec![],
                        params: TypeList {
                            types: vec![
                                Type::Table(TableType {
                                    props: vec![TableProp::Indexer {
                                        key: Type::Named(NamedType {
                                            table: None,
                                            name: "K".to_string(),
                                            params: vec![],
                                        }),
                                        value: Type::Named(NamedType {
                                            table: None,
                                            name: "V".to_string(),
                                            params: vec![],
                                        })
                                    }]
                                }),
                                Type::Optional(Box::new(Type::Named(NamedType {
                                    table: None,
                                    name: "K".to_string(),
                                    params: vec![]
                                })))
                            ],
                            vararg: None
                        },
                        ret_ty: TypeOrPack::Pack(TypePack::Listed(TypeList {
                            types: vec![
                                Type::Optional(Box::new(Type::Named(NamedType {
                                    table: None,
                                    name: "K".to_string(),
                                    params: vec![]
                                }))),
                                Type::Optional(Box::new(Type::Named(NamedType {
                                    table: None,
                                    name: "V".to_string(),
                                    params: vec![]
                                })))
                            ],
                            vararg: None
                        }))
                    })),
                    generics: vec![
                        GenericDef {
                            param: GenericParam::Name("K".to_string()),
                            default: None,
                        },
                        GenericDef {
                            param: GenericParam::Name("V".to_string()),
                            default: None,
                        }
                    ],
                    is_exported: false
                }),
                vec![]
            )]
        }
    );
}
