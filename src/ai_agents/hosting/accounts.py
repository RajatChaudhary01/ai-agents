"""User accounts and login sessions stored in SQLite.

* Passwords are hashed with scrypt (memory-hard, from the standard library) and a
  per-user random salt; plain passwords are never stored or logged.
* Login sessions are random tokens. Only a SHA-256 of each token is stored, so a leaked
  database cannot be used to log in, and sessions can be revoked server-side.
* Usernames are case-insensitive, so ``Alice`` and ``alice`` cannot both exist.
"""

import hashlib
import hmac
import re
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

USERNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,31}$")
MIN_PASSWORD_LENGTH = 10
MAX_PASSWORD_LENGTH = 128
SESSION_TTL_SECONDS = 7 * 24 * 3600

_SCRYPT = {"n": 2**14, "r": 8, "p": 1, "dklen": 32}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username      TEXT PRIMARY KEY,
    display_name  TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    created_at    REAL NOT NULL,
    created_ip    TEXT,
    disabled      INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    username   TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
    expires_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS sessions_by_user ON sessions(username);
"""


class AccountError(ValueError):
    """A signup or login problem that is safe to show to the user."""


class AccountStoreError(ValueError):
    """The accounts database cannot be opened (a deployment problem)."""


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, **_SCRYPT)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt_hex, digest_hex = stored.split("$")
    except ValueError:
        return False
    if scheme != "scrypt":
        return False
    digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), **_SCRYPT)
    return hmac.compare_digest(digest.hex(), digest_hex)


# Compared against when the username does not exist, so a failed login takes the same
# time either way and does not reveal which usernames are registered.
_DUMMY_HASH = hash_password(secrets.token_hex(16))


def validate_new_account(username: str, password: str) -> None:
    if not USERNAME_RE.fullmatch(username or ""):
        raise AccountError(
            "Username must be 3-32 characters: letters, digits, '.', '_' or '-', "
            "starting with a letter or digit."
        )
    if not MIN_PASSWORD_LENGTH <= len(password or "") <= MAX_PASSWORD_LENGTH:
        raise AccountError(
            f"Password must be {MIN_PASSWORD_LENGTH}-{MAX_PASSWORD_LENGTH} characters long."
        )
    if password.casefold() == username.casefold() or username.casefold() in password.casefold():
        raise AccountError("Password must not contain your username.")


@dataclass(frozen=True)
class User:
    username: str
    display_name: str
    created_at: float
    disabled: bool


class AccountStore:
    def __init__(self, path: str | Path, clock: Callable[[], float] = time.time) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            try:
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise AccountStoreError(
                    f"Cannot create the directory for the accounts database {self.path}: {exc}"
                ) from exc
        self._clock = clock
        self._lock = threading.Lock()
        try:
            self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
            self._db.execute("PRAGMA foreign_keys = ON")
            if self.path != ":memory:":
                self._db.execute("PRAGMA journal_mode = WAL")
            self._db.executescript(_SCHEMA)
        except sqlite3.Error as exc:
            raise AccountStoreError(
                f"Cannot open the accounts database at {self.path} ({exc}). Make sure its "
                "directory is writable; in Docker, mount a volume at /data."
            ) from exc

    # ------------------------------------------------------------------ users
    def create_user(self, username: str, password: str, ip: str | None = None) -> User:
        validate_new_account(username, password)
        key = username.casefold()
        password_hash = hash_password(password)
        now = self._clock()
        with self._lock:
            try:
                self._db.execute(
                    "INSERT INTO users (username, display_name, password_hash, created_at, "
                    "created_ip) VALUES (?, ?, ?, ?, ?)",
                    (key, username, password_hash, now, ip),
                )
            except sqlite3.IntegrityError:
                raise AccountError("That username is already taken.") from None
        return User(key, username, now, False)

    def get_user(self, username: str) -> User | None:
        with self._lock:
            row = self._db.execute(
                "SELECT username, display_name, created_at, disabled FROM users WHERE username = ?",
                (username.casefold(),),
            ).fetchone()
        return User(row[0], row[1], row[2], bool(row[3])) if row else None

    def authenticate(self, username: str, password: str) -> User | None:
        """Return the user for valid, enabled credentials, else None (in constant-ish time)."""
        with self._lock:
            row = self._db.execute(
                "SELECT username, display_name, created_at, disabled, password_hash "
                "FROM users WHERE username = ?",
                ((username or "").casefold(),),
            ).fetchone()
        if row is None:
            verify_password(password or "", _DUMMY_HASH)
            return None
        if not verify_password(password or "", row[4]) or row[3]:
            return None
        return User(row[0], row[1], row[2], bool(row[3]))

    def set_disabled(self, username: str, disabled: bool = True) -> bool:
        with self._lock:
            cur = self._db.execute(
                "UPDATE users SET disabled = ? WHERE username = ?",
                (int(disabled), username.casefold()),
            )
            if disabled:
                self._db.execute("DELETE FROM sessions WHERE username = ?", (username.casefold(),))
        return cur.rowcount > 0

    def set_password(self, username: str, password: str) -> bool:
        validate_new_account(username, password)
        password_hash = hash_password(password)
        with self._lock:
            cur = self._db.execute(
                "UPDATE users SET password_hash = ? WHERE username = ?",
                (password_hash, username.casefold()),
            )
            # Changing the password logs out every existing session.
            self._db.execute("DELETE FROM sessions WHERE username = ?", (username.casefold(),))
        return cur.rowcount > 0

    def list_users(self) -> list[User]:
        with self._lock:
            rows = self._db.execute(
                "SELECT username, display_name, created_at, disabled FROM users ORDER BY created_at"
            ).fetchall()
        return [User(r[0], r[1], r[2], bool(r[3])) for r in rows]

    # --------------------------------------------------------------- sessions
    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    def create_session(self, username: str, ttl: float = SESSION_TTL_SECONDS) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._db.execute(
                "INSERT INTO sessions (token_hash, username, expires_at) VALUES (?, ?, ?)",
                (self._token_hash(token), username.casefold(), self._clock() + ttl),
            )
        return token

    def session_user(self, token: str | None) -> str | None:
        """Username for a valid session token, or None if missing, expired or disabled."""
        if not token or len(token) > 128:
            return None
        with self._lock:
            row = self._db.execute(
                "SELECT s.username, s.expires_at, u.disabled FROM sessions s "
                "JOIN users u ON u.username = s.username WHERE s.token_hash = ?",
                (self._token_hash(token),),
            ).fetchone()
        if row is None or row[2] or row[1] < self._clock():
            return None
        return row[0]

    def delete_session(self, token: str | None) -> None:
        if not token:
            return
        with self._lock:
            self._db.execute(
                "DELETE FROM sessions WHERE token_hash = ?", (self._token_hash(token),)
            )

    def purge_expired_sessions(self) -> int:
        with self._lock:
            cur = self._db.execute("DELETE FROM sessions WHERE expires_at < ?", (self._clock(),))
        return cur.rowcount
