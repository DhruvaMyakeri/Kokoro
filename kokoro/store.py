"""
State persistence layer for Kokoro.

Stores per-user emotional state vectors in a local SQLite database.
Fully local, zero external dependencies beyond the standard library and numpy.

Schema (table: user_states):
  user_id       TEXT PRIMARY KEY
  state_vector  BLOB            numpy array serialised with ndarray.tobytes()
  state_dim     INTEGER         used to verify dtype/shape on load
  session_count INTEGER         incremented on every save
  valence       REAL            most recent valence from the linear probe
  arousal       REAL            most recent arousal from the linear probe
  arc_history   TEXT            JSON list of up to 20 [valence, arousal] pairs,
                                oldest first
  created_at    TEXT            ISO-8601 UTC timestamp
  updated_at    TEXT            ISO-8601 UTC timestamp

Design decisions:
  - SQLite WAL mode for safe concurrent readers.
  - state_vector stored as raw bytes (float32 little-endian via tobytes()).
    state_dim is stored alongside so we can reconstruct the exact array on load
    and detect accidental dimension mismatches.
  - arc_history is a JSON array rather than a separate table.  It is small
    (at most 20 × 2 floats), changes every save, and is always read alongside
    the main record — a join would add complexity for no benefit.
  - Timestamps are stored as ISO-8601 strings in UTC.  SQLite has no native
    datetime type; strings are human-readable and sortable.
  - All public methods are atomic: they open, execute, and close a connection
    in a single with-block.  No persistent connection is held, which keeps the
    class safe to use from multiple processes.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_DB: Path = Path.home() / ".kokoro" / "kokoro.db"
_ARC_HISTORY_MAX: int = 20
_STATE_DTYPE: np.dtype = np.dtype("float32")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS user_states (
    user_id       TEXT    PRIMARY KEY,
    state_vector  BLOB    NOT NULL,
    state_dim     INTEGER NOT NULL,
    session_count INTEGER NOT NULL DEFAULT 0,
    valence       REAL    NOT NULL DEFAULT 0.0,
    arousal       REAL    NOT NULL DEFAULT 0.0,
    arc_history   TEXT    NOT NULL DEFAULT '[]',
    created_at    TEXT    NOT NULL,
    updated_at    TEXT    NOT NULL
)
"""

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _vec_to_blob(arr: np.ndarray) -> bytes:
    return arr.astype(_STATE_DTYPE).tobytes()


def _blob_to_vec(blob: bytes, state_dim: int) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=_STATE_DTYPE).copy()
    if arr.size != state_dim:
        raise ValueError(
            f"Stored vector has {arr.size} elements but state_dim={state_dim}. "
            "The database may be corrupted or was written by a different model."
        )
    return arr


def _append_arc_history(
    history_json: str,
    valence: float,
    arousal: float,
) -> str:
    """Append [valence, arousal] to the history list, capped at _ARC_HISTORY_MAX."""
    history: list[list[float]] = json.loads(history_json)
    history.append([float(valence), float(arousal)])
    if len(history) > _ARC_HISTORY_MAX:
        history = history[-_ARC_HISTORY_MAX:]
    return json.dumps(history)


# ---------------------------------------------------------------------------
# StateStore
# ---------------------------------------------------------------------------

class StateStore:
    """
    Persistent store for per-user Kokoro state vectors.

    Usage:
        store = StateStore()                          # ~/.kokoro/kokoro.db
        store = StateStore("/path/to/custom.db")

        store.save("user_42", state_vec, valence=0.6, arousal=0.1)
        vec  = store.load("user_42")                  # np.ndarray or None
        info = store.get_info("user_42")              # dict or None
        store.reset("user_42")                        # keep user, clear state
        store.delete("user_42")
        ids  = store.list_users()                     # list[str]
    """

    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        """
        Args:
            db_path: Path to the SQLite database file.
                     Defaults to ~/.kokoro/kokoro.db.
                     The parent directory is created automatically if absent.
        """
        if db_path is None:
            self._db_path = _DEFAULT_DB
        else:
            self._db_path = Path(db_path)

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self._db_path))
        con.execute("PRAGMA journal_mode=WAL")
        con.row_factory = sqlite3.Row
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute(_CREATE_TABLE)
            con.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        user_id: str,
        state_vector: np.ndarray,
        valence: float,
        arousal: float,
    ) -> None:
        """
        Upsert the state for a user.

        On first call: creates a new record with session_count=1 and
        arc_history=[[valence, arousal]].

        On subsequent calls: increments session_count, appends
        [valence, arousal] to arc_history (capped at 20 entries),
        and updates updated_at.

        Args:
            user_id:      Unique string identifier for the user.
            state_vector: 1-D float32 numpy array (e.g. shape (384,)).
            valence:      Most recent valence in [-1, 1].
            arousal:      Most recent arousal in [-1, 1].

        Raises:
            ValueError: if state_vector is not a 1-D array.
        """
        if state_vector.ndim != 1:
            raise ValueError(
                f"state_vector must be 1-D, got shape {state_vector.shape}"
            )

        blob      = _vec_to_blob(state_vector)
        state_dim = state_vector.shape[0]
        now       = _now_iso()

        with self._connect() as con:
            existing = con.execute(
                "SELECT session_count, arc_history FROM user_states WHERE user_id = ?",
                (user_id,),
            ).fetchone()

            if existing is None:
                # INSERT — new user
                new_history = json.dumps([[float(valence), float(arousal)]])
                con.execute(
                    """
                    INSERT INTO user_states
                        (user_id, state_vector, state_dim, session_count,
                         valence, arousal, arc_history, created_at, updated_at)
                    VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?)
                    """,
                    (user_id, blob, state_dim,
                     float(valence), float(arousal),
                     new_history, now, now),
                )
            else:
                # UPDATE — existing user
                new_count   = existing["session_count"] + 1
                new_history = _append_arc_history(
                    existing["arc_history"], valence, arousal
                )
                con.execute(
                    """
                    UPDATE user_states
                    SET state_vector  = ?,
                        state_dim     = ?,
                        session_count = ?,
                        valence       = ?,
                        arousal       = ?,
                        arc_history   = ?,
                        updated_at    = ?
                    WHERE user_id = ?
                    """,
                    (blob, state_dim, new_count,
                     float(valence), float(arousal),
                     new_history, now, user_id),
                )
            con.commit()

    def load(self, user_id: str) -> Optional[np.ndarray]:
        """
        Return the stored state vector for a user, or None if not found.

        Returns:
            1-D float32 numpy array of shape (state_dim,), or None.

        Raises:
            ValueError: if the stored blob is inconsistent with state_dim
                        (indicates database corruption or model mismatch).
        """
        with self._connect() as con:
            row = con.execute(
                "SELECT state_vector, state_dim FROM user_states WHERE user_id = ?",
                (user_id,),
            ).fetchone()

        if row is None:
            return None
        return _blob_to_vec(bytes(row["state_vector"]), row["state_dim"])

    def get_info(self, user_id: str) -> Optional[dict]:
        """
        Return metadata for a user without loading the full state vector.

        Returns:
            dict with keys:
                session_count (int)
                valence       (float)
                arousal       (float)
                arc_history   (list of [valence, arousal] pairs, oldest first)
                created_at    (str, ISO-8601 UTC)
                updated_at    (str, ISO-8601 UTC)
                state_dim     (int)
            or None if the user does not exist.
        """
        with self._connect() as con:
            row = con.execute(
                """
                SELECT session_count, valence, arousal, arc_history,
                       created_at, updated_at, state_dim
                FROM user_states WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()

        if row is None:
            return None
        return {
            "session_count": row["session_count"],
            "valence":       row["valence"],
            "arousal":       row["arousal"],
            "arc_history":   json.loads(row["arc_history"]),
            "created_at":    row["created_at"],
            "updated_at":    row["updated_at"],
            "state_dim":     row["state_dim"],
        }

    def reset(self, user_id: str) -> bool:
        """
        Reset a user's state to cold-start without deleting the record.

        Sets state_vector to a zero vector, session_count to 0,
        arc_history to [], valence and arousal to 0.0, and updates updated_at.

        Args:
            user_id: The user to reset.

        Returns:
            True if the user existed and was reset, False if not found.
        """
        with self._connect() as con:
            row = con.execute(
                "SELECT state_dim FROM user_states WHERE user_id = ?",
                (user_id,),
            ).fetchone()

            if row is None:
                return False

            zero_blob = _vec_to_blob(np.zeros(row["state_dim"], dtype=_STATE_DTYPE))
            con.execute(
                """
                UPDATE user_states
                SET state_vector  = ?,
                    session_count = 0,
                    valence       = 0.0,
                    arousal       = 0.0,
                    arc_history   = '[]',
                    updated_at    = ?
                WHERE user_id = ?
                """,
                (zero_blob, _now_iso(), user_id),
            )
            con.commit()
        return True

    def delete(self, user_id: str) -> bool:
        """
        Permanently delete a user record.

        Returns:
            True if a record was deleted, False if the user was not found.
        """
        with self._connect() as con:
            cursor = con.execute(
                "DELETE FROM user_states WHERE user_id = ?", (user_id,)
            )
            con.commit()
        return cursor.rowcount > 0

    def list_users(self) -> list[str]:
        """
        Return a list of all stored user_ids, ordered by created_at ascending.
        """
        with self._connect() as con:
            rows = con.execute(
                "SELECT user_id FROM user_states ORDER BY created_at ASC"
            ).fetchall()
        return [row["user_id"] for row in rows]

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"StateStore(db_path={str(self._db_path)!r})"


# ---------------------------------------------------------------------------
# __main__ — self-test
# ---------------------------------------------------------------------------

def _approx(x: float, rel: float = 1e-5) -> object:
    """Inline float comparator so the test block has no pytest dependency."""
    class _A:
        def __eq__(self, other: object) -> bool:
            return abs(float(other) - x) <= rel * max(abs(x), 1e-12)
        def __repr__(self) -> str:
            return f"~{x}"
    return _A()


if __name__ == "__main__":
    import sys

    passed = 0
    failed = 0

    def check(label: str, condition: bool, detail: str = "") -> None:
        global passed, failed
        if condition:
            print(f"  [PASS] {label}")
            passed += 1
        else:
            msg = f"  [FAIL] {label}"
            if detail:
                msg += f"  ({detail})"
            print(msg)
            failed += 1

    print("\n=== StateStore self-test ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store   = StateStore(db_path)

        # ------------------------------------------------------------------
        # 1. Basic round-trip
        # ------------------------------------------------------------------
        print("--- Round-trip lossless ---")
        STATE_DIM = 384
        rng       = np.random.default_rng(42)

        vec0 = rng.standard_normal(STATE_DIM).astype(np.float32)
        store.save("alice", vec0, valence=0.6, arousal=0.2)
        loaded = store.load("alice")

        check("load returns ndarray",        isinstance(loaded, np.ndarray))
        check("shape preserved",             loaded.shape == (STATE_DIM,),
              f"got {loaded.shape}")
        check("dtype is float32",            loaded.dtype == np.float32)
        check("values lossless",             np.allclose(vec0, loaded, atol=1e-7),
              f"max diff={np.abs(vec0 - loaded).max():.2e}")

        # ------------------------------------------------------------------
        # 2. session_count increments correctly
        # ------------------------------------------------------------------
        print("\n--- session_count ---")
        for i in range(1, 5):
            store.save("alice", rng.standard_normal(STATE_DIM).astype(np.float32),
                       valence=float(i) * 0.1, arousal=float(i) * 0.05)

        info = store.get_info("alice")
        check("session_count == 5",          info["session_count"] == 5,
              f"got {info['session_count']}")
        check("arc_history has 5 entries",   len(info["arc_history"]) == 5,
              f"got {len(info['arc_history'])}")
        check("arc_history oldest first",    info["arc_history"][0][0] == _approx(0.6),
              f"first entry: {info['arc_history'][0]}")

        # ------------------------------------------------------------------
        # 3. arc_history capped at 20
        # ------------------------------------------------------------------
        print("\n--- arc_history cap ---")
        for i in range(20):
            store.save("alice", rng.standard_normal(STATE_DIM).astype(np.float32),
                       valence=float(i) * 0.02, arousal=0.0)

        info = store.get_info("alice")
        check("session_count == 25",         info["session_count"] == 25,
              f"got {info['session_count']}")
        check("arc_history capped at 20",    len(info["arc_history"]) == 20,
              f"got {len(info['arc_history'])}")
        # After 25 saves (indices 0-24) the oldest kept should be index 5.
        # valence for save index 5 (in the 20-loop): i=0 gives 0.0 but the first
        # of the 20-loop saves happens after the 5 earlier saves; the 20-loop
        # starts at i=0→valence=0.0.  The 20 kept are the last 20 of the 25.
        check("arc_history entries are lists", all(
              isinstance(e, list) and len(e) == 2
              for e in info["arc_history"]),
              f"bad entries: {info['arc_history'][:2]}")

        # ------------------------------------------------------------------
        # 4. delete
        # ------------------------------------------------------------------
        print("\n--- delete ---")
        result = store.delete("alice")
        check("delete returns True for existing user", result is True)
        check("load returns None after delete",        store.load("alice") is None)
        check("get_info returns None after delete",    store.get_info("alice") is None)
        result2 = store.delete("alice")
        check("delete returns False for missing user", result2 is False)

        # ------------------------------------------------------------------
        # 5. list_users
        # ------------------------------------------------------------------
        print("\n--- list_users ---")
        store.save("bob",   rng.standard_normal(STATE_DIM).astype(np.float32), 0.1, 0.1)
        store.save("carol", rng.standard_normal(STATE_DIM).astype(np.float32), 0.2, 0.2)
        store.save("dave",  rng.standard_normal(STATE_DIM).astype(np.float32), 0.3, 0.3)
        users = store.list_users()
        check("list_users returns 3 users",             len(users) == 3,
              f"got {users}")
        check("list_users contains bob",                "bob"   in users)
        check("list_users contains carol",              "carol" in users)
        check("list_users contains dave",               "dave"  in users)

        # ------------------------------------------------------------------
        # 6. reset
        # ------------------------------------------------------------------
        print("\n--- reset ---")
        # Give bob a few sessions first
        for i in range(4):
            store.save("bob", rng.standard_normal(STATE_DIM).astype(np.float32),
                       valence=float(i) * 0.1, arousal=0.0)
        info_before = store.get_info("bob")
        check("bob has 5 sessions before reset",      info_before["session_count"] == 5,
              f"got {info_before['session_count']}")

        reset_ok = store.reset("bob")
        check("reset returns True for existing user", reset_ok is True)

        info_after = store.get_info("bob")
        check("session_count reset to 0",             info_after["session_count"] == 0)
        check("arc_history cleared",                  info_after["arc_history"] == [])
        check("valence reset to 0.0",                 info_after["valence"] == 0.0)
        check("arousal reset to 0.0",                 info_after["arousal"] == 0.0)

        bob_vec_after = store.load("bob")
        check("state_vector reset to zeros",          np.all(bob_vec_after == 0.0),
              f"non-zero elements: {np.count_nonzero(bob_vec_after)}")

        reset_missing = store.reset("nobody")
        check("reset returns False for missing user", reset_missing is False)

        # user record still exists after reset
        check("bob still in list_users after reset",  "bob" in store.list_users())

        # ------------------------------------------------------------------
        # 7. Two users simultaneously — no cross-contamination
        # ------------------------------------------------------------------
        print("\n--- Two users simultaneously ---")
        vec_u1 = rng.standard_normal(STATE_DIM).astype(np.float32)
        vec_u2 = rng.standard_normal(STATE_DIM).astype(np.float32)

        store.save("user_1", vec_u1, valence=0.9,  arousal=0.5)
        store.save("user_2", vec_u2, valence=-0.7, arousal=0.1)

        loaded_u1 = store.load("user_1")
        loaded_u2 = store.load("user_2")

        check("user_1 loads its own vector",
              np.allclose(vec_u1, loaded_u1, atol=1e-7))
        check("user_2 loads its own vector",
              np.allclose(vec_u2, loaded_u2, atol=1e-7))
        check("user_1 valence correct",
              store.get_info("user_1")["valence"] == _approx(0.9))
        check("user_2 valence correct",
              store.get_info("user_2")["valence"] == _approx(-0.7))
        check("user_1 and user_2 vectors differ",
              not np.allclose(loaded_u1, loaded_u2))

        # ------------------------------------------------------------------
        # 8. timestamps
        # ------------------------------------------------------------------
        print("\n--- Timestamps ---")
        info_u1 = store.get_info("user_1")
        check("created_at is a string",    isinstance(info_u1["created_at"], str))
        check("updated_at is a string",    isinstance(info_u1["updated_at"], str))
        check("created_at ends with Z",    info_u1["created_at"].endswith("Z"))
        check("updated_at ends with Z",    info_u1["updated_at"].endswith("Z"))

        import time
        time.sleep(0.01)
        store.save("user_1", rng.standard_normal(STATE_DIM).astype(np.float32),
                   valence=0.8, arousal=0.4)
        info_u1b = store.get_info("user_1")
        check("updated_at changes on save",
              info_u1b["updated_at"] != info_u1["updated_at"])
        check("created_at unchanged on update",
              info_u1b["created_at"] == info_u1["created_at"])

        # ------------------------------------------------------------------
        # 9. ValueError on bad input
        # ------------------------------------------------------------------
        print("\n--- Input validation ---")
        try:
            store.save("bad", np.zeros((4, 384), dtype=np.float32), 0.0, 0.0)
            check("2-D array raises ValueError", False, "no error raised")
        except ValueError:
            check("2-D array raises ValueError", True)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = passed + failed
    print(f"\n{'='*40}")
    print(f"  {passed}/{total} tests passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
        sys.exit(1)
    else:
        print("  -- all clear")
    print(f"{'='*40}\n")


def _approx(x, rel=1e-5):
    """Tiny inline approx helper so the __main__ block has no pytest dep."""
    class _Approx:
        def __eq__(self, other):
            return abs(other - x) <= rel * max(abs(x), 1e-12)
        def __repr__(self):
            return f"~{x}"
    return _Approx()
