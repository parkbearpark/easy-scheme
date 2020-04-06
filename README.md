# easy-scheme

## BNF

EXPR = EXPR3;

EXPR3 = "(", ("+" | "-"), " ", EXPR2, " ", EXPR2, ")" | EXPR2;

EXPR2 = "(", ("\*" | "/"), " ", EXPR1, " ", EXPR1, ")" | EXPR1;

EXPR1 = ("+" | "-"), ATOM | ATOM;

ATOM = UNUMBER | EXPR3;

UNUMBER = DIGIT, {DIGIT};

DIGIT = "0" .. "9";
