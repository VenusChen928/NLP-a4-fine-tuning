import re

# Ordinary text markers preserve operators that T5 encodes as UNK:
SQL_PREPROCESSING_MODES = ("none", "operators_v1")
SQL_OPERATOR_TOKENS = {"<=": "SQL_LE", "<": "SQL_LT"}
SQL_TOKEN_OPERATORS = {token: operator for operator, token in SQL_OPERATOR_TOKENS.items()}
SQL_OPERATOR_PATTERN = re.compile(r"<=|<")
SQL_MARKER_PATTERN = re.compile(r"\b(?:SQL_LE|SQL_LT)\b")

# Quoted strings, quoted identifiers and comments are left untouched:
SQL_PROTECTED_PATTERN = re.compile(r"('(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`|\[[^\]]*\]|--[^\r\n]*|/\*[\s\S]*?\*/)")


def validate_sql_preprocessing(mode):
    if mode not in SQL_PREPROCESSING_MODES:
        raise ValueError(f"Unknown SQL preprocessing mode: {mode!r}!")
    return mode


def resolve_sql_preprocessing(config, requested = None):
    # Old checkpoints default to none
    # Explicit mismatches are rejected:
    saved_mode = validate_sql_preprocessing(config.get("sql_preprocessing", "none"))

    if requested is not None and validate_sql_preprocessing(requested) != saved_mode:
        raise ValueError(f"Requested SQL preprocessing {requested!r} differs from saved mode {saved_mode!r}!")

    return saved_mode


def encode_sql(sql, mode = "none"):
    validate_sql_preprocessing(mode)

    if mode == "none":
        return sql

    if any(token in sql for token in SQL_TOKEN_OPERATORS):
        raise ValueError("Original SQL contains a reserved SQL processing marker!")

    parts = SQL_PROTECTED_PATTERN.split(sql)
    for index in range(0, len(parts), 2):
        part = parts[index]

        def replace_operator(match):
            # Separate markers from adjacent identifiers or numbers:
            before = " " if match.start() > 0 and re.match(r"\w", part[match.start() - 1]) else ""
            after = " " if match.end() < len(part) and re.match(r"\w", part[match.end()]) else ""

            return before + SQL_OPERATOR_TOKENS[match.group()] + after

        parts[index] = SQL_OPERATOR_PATTERN.sub(replace_operator, part)

    return "".join(parts)


def decode_sql(text, mode = "none"):
    validate_sql_preprocessing(mode)

    if mode == "none":
        return text

    # Restore complete markers before SQL execution and official scoring:
    parts = SQL_PROTECTED_PATTERN.split(text)
    for index in range(0, len(parts), 2):
        parts[index] = SQL_MARKER_PATTERN.sub(lambda match: SQL_TOKEN_OPERATORS[match.group()], parts[index])

    return "".join(parts)
