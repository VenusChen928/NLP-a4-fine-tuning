"""Regression checks for SQL representation and bounded database execution."""

import hashlib
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

PART2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART2))

from sql_processing import encode_sql, decode_sql, resolve_sql_preprocessing
import utils


@pytest.mark.parametrize("query", [
    "SELECT 1 < 2, 2 <= 2, 3 > 2, 3 >= 3, 1 <> 2",
    "SELECT 'it''s < text', '<=', 1 < 2 /* keep <= here */ -- keep < here",
    'SELECT 1 AS "<", 2 AS [<=], 3 AS `x<y`',
    "SELECT 1<2, 2<=2, 1<>2"
])
def test_processing_preserves_sql_meaning(query):
    # SQL processing: preserve quoted text, comments and query results
    encoded = encode_sql(query, "operators_v1")
    restored = decode_sql(encoded, "operators_v1")
    with sqlite3.connect(":memory:") as connection:
        assert connection.execute(restored).fetchall() == connection.execute(query).fetchall()
    if " < " in query or " AS " in query:
        assert restored == query
    assert encode_sql(query, "none") == decode_sql(query, "none") == query


def test_saved_mode_and_marker_validation():
    assert resolve_sql_preprocessing({}) == "none"
    assert resolve_sql_preprocessing({"sql_preprocessing": "operators_v1"}) == "operators_v1"
    with pytest.raises(ValueError):
        resolve_sql_preprocessing({"sql_preprocessing": "operators_v1"}, requested = "none")
    with pytest.raises(ValueError):
        encode_sql("SELECT SQL_LT", "operators_v1")
    with pytest.raises(ValueError):
        encode_sql("SELECT 1", "unknown")
    assert decode_sql("SELECT 'SQL_LT', SQL_LT_backup, 1 SQL_LT 2", "operators_v1") == "SELECT 'SQL_LT', SQL_LT_backup, 1 < 2"


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "queries.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE items (value INTEGER)")
        connection.execute("INSERT INTO items VALUES (7)")
    return path


def test_sql_output_order_errors_and_read_only(database, monkeypatch):
    monkeypatch.setattr(utils, "DB_PATH", str(database))
    original_hash = hashlib.sha256(database.read_bytes()).hexdigest()
    queries = ["SELECT 2", "SELECT value FROM items", "SELECT missing FROM items", "DELETE FROM items", "SELECT 1"]
    records, errors = utils.compute_records(queries)
    assert records == [[(2,)], [(7,)], [], [], [(1,)]]
    assert [bool(error) for error in errors] == [False, False, True, True, False]
    assert hashlib.sha256(database.read_bytes()).hexdigest() == original_hash
    assert utils.compute_records([]) == ([], [])


@pytest.mark.parametrize("workers", [1, 2])
def test_timed_out_sql_does_not_block_process_exit(database, workers):
    # SQL execution: verify running and queued queries cannot keep Python alive
    code = """
import sys
sys.path.insert(0, sys.argv[1])
import utils
utils.DB_PATH = sys.argv[2]
workers = int(sys.argv[3])
query = 'WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) SELECT sum(x) FROM n'
records, errors = utils.compute_records([query, 'SELECT 2'], timeout_secs = 0.1, num_threads = workers)
assert records[0] == [] and errors[0] == 'Query timed out'
assert records[1] == ([(2,)] if workers == 2 else [])
assert errors[1] == ('' if workers == 2 else 'Query timed out')
"""
    subprocess.run([sys.executable, "-B", "-c", code, str(PART2), str(database), str(workers)], check = True, timeout = 15)
