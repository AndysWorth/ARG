"""Compatibility tests for the kuzu graph database library.

Verifies the exact API surface ARG uses so breaking changes are caught when
the dependency is updated. If any of these tests fail after a kuzu upgrade,
the corresponding call sites in arg/graph/knowledge_graph.py must be updated.

Known Kuzu quirks documented here (and in knowledge_graph.py):
- Variable-length path depth (``*1..N``) must be a literal integer in the
  query string, not a query parameter — the parser rejects ``*1..$depth``.
- Primary-key properties cannot appear in SET clauses of MERGE statements.
"""

from __future__ import annotations

import kuzu
import pytest

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def conn(tmp_path):
    db = kuzu.Database(str(tmp_path / "test.kuzu"))
    c = kuzu.Connection(db)
    yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_database_and_connection_constructable(tmp_path) -> None:
    db = kuzu.Database(str(tmp_path / "db.kuzu"))
    c = kuzu.Connection(db)
    assert c is not None


def test_create_node_table_if_not_exists_is_idempotent(conn) -> None:
    """Running CREATE NODE TABLE IF NOT EXISTS twice must not raise."""
    ddl = "CREATE NODE TABLE IF NOT EXISTS Doc(id STRING, label STRING, PRIMARY KEY(id))"
    conn.execute(ddl)
    conn.execute(ddl)  # second time must be a no-op


def test_create_rel_table_if_not_exists_is_idempotent(conn) -> None:
    conn.execute("CREATE NODE TABLE IF NOT EXISTS A(id STRING, PRIMARY KEY(id))")
    conn.execute("CREATE NODE TABLE IF NOT EXISTS B(id STRING, PRIMARY KEY(id))")
    ddl = "CREATE REL TABLE IF NOT EXISTS LINKS(FROM A TO B)"
    conn.execute(ddl)
    conn.execute(ddl)


def test_merge_node_does_not_create_duplicates(conn) -> None:
    conn.execute("CREATE NODE TABLE IF NOT EXISTS Doc(id STRING, label STRING, PRIMARY KEY(id))")
    conn.execute("MERGE (d:Doc {id: $id})", {"id": "alpha"})
    conn.execute("MERGE (d:Doc {id: $id})", {"id": "alpha"})
    result = conn.execute("MATCH (d:Doc) RETURN count(d)")
    count = result.get_next()[0]
    assert count == 1, f"MERGE created duplicate node; count={count}"


def test_merge_rel_does_not_create_duplicates(conn) -> None:
    conn.execute("CREATE NODE TABLE IF NOT EXISTS Doc(id STRING, PRIMARY KEY(id))")
    conn.execute("CREATE REL TABLE IF NOT EXISTS LINKS_TO(FROM Doc TO Doc)")
    conn.execute("MERGE (d:Doc {id: $id})", {"id": "a"})
    conn.execute("MERGE (d:Doc {id: $id})", {"id": "b"})
    for _ in range(3):
        conn.execute(
            "MATCH (s:Doc {id: $s}), (t:Doc {id: $t}) MERGE (s)-[:LINKS_TO]->(t)",
            {"s": "a", "t": "b"},
        )
    result = conn.execute("MATCH ()-[r:LINKS_TO]->() RETURN count(r)")
    count = result.get_next()[0]
    assert count == 1, f"MERGE created duplicate edges; count={count}"


def test_skip_and_limit_are_independent(conn) -> None:
    """SKIP N LIMIT M must skip N then return at most M rows."""
    conn.execute("CREATE NODE TABLE IF NOT EXISTS Item(id INT64, PRIMARY KEY(id))")
    for i in range(10):
        conn.execute("MERGE (x:Item {id: $id})", {"id": i})
    result = conn.execute("MATCH (x:Item) RETURN x.id ORDER BY x.id SKIP 3 LIMIT 2")
    rows = []
    while result.has_next():
        rows.append(result.get_next()[0])
    assert rows == [3, 4], f"expected [3, 4], got {rows}"


def test_variable_length_path_with_formatted_depth(conn) -> None:
    """*1..N path syntax with N as a formatted int literal must work."""
    conn.execute("CREATE NODE TABLE IF NOT EXISTS N(id STRING, PRIMARY KEY(id))")
    conn.execute("CREATE REL TABLE IF NOT EXISTS E(FROM N TO N)")
    for node in ("a", "b", "c"):
        conn.execute("MERGE (x:N {id: $id})", {"id": node})
    conn.execute("MATCH (s:N {id: $s}), (t:N {id: $t}) MERGE (s)-[:E]->(t)", {"s": "a", "t": "b"})
    conn.execute("MATCH (s:N {id: $s}), (t:N {id: $t}) MERGE (s)-[:E]->(t)", {"s": "b", "t": "c"})
    depth = 2
    query = f"MATCH (s:N {{id: $id}})-[:E*1..{int(depth)}]->(t:N) RETURN DISTINCT t.id"
    result = conn.execute(query, {"id": "a"})
    reached = set()
    while result.has_next():
        reached.add(result.get_next()[0])
    assert reached == {"b", "c"}, f"expected {{b, c}} at depth 2, got {reached}"


def test_variable_length_path_depth_param_rejected(conn) -> None:
    """Kuzu rejects ``*1..$depth`` — depth must be a literal.

    This test documents the quirk so we don't accidentally try to pass depth
    as a parameter in the future.
    """
    conn.execute("CREATE NODE TABLE IF NOT EXISTS P(id STRING, PRIMARY KEY(id))")
    conn.execute("CREATE REL TABLE IF NOT EXISTS F(FROM P TO P)")
    with pytest.raises(RuntimeError):
        conn.execute(
            "MATCH (s:P {id: $id})-[:F*1..$depth]->(t:P) RETURN DISTINCT t.id",
            {"id": "x", "depth": 2},
        )
