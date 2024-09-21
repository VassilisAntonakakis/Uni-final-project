
0.583412

logicOp = "and" | "or"
logicExpr = expression + logicOp + expression
expression = expression | id | logicExpr | left_paren + logicExpr + right_paren