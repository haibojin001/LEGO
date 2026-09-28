# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg159::ast.Call+ast.Load+ast.Name
# name: ast_primitive
# summary: Uses ast.Call, ast.Load, ast.Name across 2 repos
# anchor_symbols: ['ast.Call', 'ast.Load', 'ast.Name']
# observed in 2 repos: ['microsoft__nni', 'vaexio__vaex']...

# --- from vaexio__vaex::packages/vaex-core/vaex/expresso.py::call ---
def call(fname, args):
    return ast.Call(func=ast.Name(id=fname, ctx=ast.Load()), args=args)

# --- from microsoft__nni::nni/common/concrete_trace_utils/operator_patcher.py::TransformerOp.visit_UnaryOp ---
def visit_UnaryOp(self, node: ast.UnaryOp):
        if self.is_incond_status != 0:
            # in branch cond test expr, need no replacement
            self.is_incond_status = 2
            return self.generic_visit(node)
        elif _orig_isinstance(node.op, ast.Not):
            self.is_transformed = True
            return self.generic_visit(ast.Call(
                func=ast.Name(id='not_', ctx=ast.Load()),
                args=[node.operand],
                keywords=[],
            ))
        else:
            return self.generic_visit(node)

# --- from microsoft__nni::nni/common/concrete_trace_utils/operator_patcher.py::TransformerOp.visit_Call ---
def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == 'super' and _orig_len(node.args) == 0:
            return self.generic_visit(ast.Call(
                func=ast.Name(id='super', ctx=ast.Load()),
                args=[
                    ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='__class__', ctx=ast.Load()),
                    ast.Name(id='self', ctx=ast.Load()),
                ],
                keywords=node.keywords,
            ))
        elif not isinstance(node.func, ast.Name) or node.func.id != 'patch_run':
            self.is_transformed = True
            return self.generic_visit(ast.Call(
                func=ast.Name(id='patch_run', ctx=ast.Load()),
                args=[node.func, *node.args],
                keywords=node.keywords,
            ))
        else:
            return self.generic_visit(node)
