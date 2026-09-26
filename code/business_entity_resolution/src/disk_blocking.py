"""Disk-backed candidate generation for the multi-million-row datasets.

The in-memory blocker is useful for small experiments but cannot index the real
challenge pool. This module stores normalized records, exact blocking keys, and
trigram FTS indexes in SQLite so the source pool is never loaded into pandas.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sqlite3
from pathlib import Path
from typing import Iterator

from blocking import _first_two_tokens, _first_token, _sorted_token_key
from features import extract_postal, normalize_address, normalize_name, tokens

LOG = logging.getLogger(__name__)
TRIGRAM_RE = re.compile(r"(?=(.{3}))")
SCHEMA = """
CREATE TABLE records (
    entity_id TEXT PRIMARY KEY,
    business_name TEXT NOT NULL,
    business_address TEXT NOT NULL,
    country TEXT NOT NULL
);
CREATE TABLE exact_keys (
    key_type TEXT NOT NULL,
    key_value TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    PRIMARY KEY (key_type, key_value, entity_id)
);
CREATE INDEX exact_keys_lookup ON exact_keys(key_type, key_value);
CREATE VIRTUAL TABLE name_fts USING fts5(entity_id UNINDEXED, value, tokenize='trigram');
CREATE VIRTUAL TABLE address_fts USING fts5(entity_id UNINDEXED, value, tokenize='trigram');
"""


def _rows(path: str | Path) -> Iterator[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = ["entity_id", "business_name", "business_address", "country"]
        if reader.fieldnames != expected:
            raise ValueError(f"{path}: expected columns {expected}, got {reader.fieldnames}")
        for row in reader:
            yield {key: (row.get(key) or "").strip() for key in expected}


def _keys(name: str, address: str, country: str) -> set[tuple[str, str]]:
    normalized_name = normalize_name(name)
    normalized_address = normalize_address(address)
    result: set[tuple[str, str]] = set()
    for key_type, value in (
        ("name", normalized_name),
        ("name_stripped", normalize_name(name, strip_suffix=True)),
        ("name_sorted", _sorted_token_key(normalized_name)),
        ("name_first", _first_token(normalized_name)),
        ("name_first_two", _first_two_tokens(normalized_name)),
        ("postal", extract_postal(normalized_address, country)),
    ):
        if value:
            result.add((key_type, value))
    for token in set(tokens(normalized_address)):
        if len(token) >= 4:
            result.add(("address_token", token))
    return result


def build_index(source_paths: list[str | Path], database_path: str | Path, commit_rows: int = 20_000) -> None:
    """Build or replace a SQLite index from Source 2/3 TSVs."""
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if database_path.exists():
        database_path.unlink()
    connection = sqlite3.connect(database_path)
    try:
        connection.executescript(SCHEMA)
        records: list[tuple[str, str, str, str]] = []
        exact: list[tuple[str, str, str]] = []
        names: list[tuple[str, str]] = []
        addresses: list[tuple[str, str]] = []
        total = 0
        for path in source_paths:
            for row in _rows(path):
                entity_id = row["entity_id"]
                name = normalize_name(row["business_name"])
                address = normalize_address(row["business_address"])
                records.append((entity_id, row["business_name"], row["business_address"], row["country"]))
                exact.extend((key_type, value, entity_id) for key_type, value in _keys(
                    row["business_name"], row["business_address"], row["country"]
                ))
                names.append((entity_id, name))
                addresses.append((entity_id, address))
                total += 1
                if len(records) >= commit_rows:
                    _insert_batch(connection, records, exact, names, addresses)
                    records.clear(); exact.clear(); names.clear(); addresses.clear()
                    LOG.info("Indexed %,d records", total)
        if records:
            _insert_batch(connection, records, exact, names, addresses)
        connection.commit()
        LOG.info("Finished SQLite index with %,d records at %s", total, database_path)
    finally:
        connection.close()


def _insert_batch(connection: sqlite3.Connection, records: list[tuple], exact: list[tuple], names: list[tuple], addresses: list[tuple]) -> None:
    connection.executemany("INSERT INTO records VALUES (?, ?, ?, ?)", records)
    connection.executemany("INSERT OR IGNORE INTO exact_keys VALUES (?, ?, ?)", exact)
    connection.executemany("INSERT INTO name_fts VALUES (?, ?)", names)
    connection.executemany("INSERT INTO address_fts VALUES (?, ?)", addresses)
    connection.commit()


def _fts_query(value: str, limit: int) -> list[str]:
    grams = list(dict.fromkeys(match.group(1) for match in TRIGRAM_RE.finditer(value)))
    return [f'"{gram.replace(chr(34), chr(34) * 2)}"' for gram in grams[:24]]


def candidates(connection: sqlite3.Connection, name: str, address: str, country: str, *, name_k: int = 30, address_k: int = 20) -> set[str]:
    """Return a union of exact and trigram candidates for one Source 1 row."""
    normalized_name = normalize_name(name)
    normalized_address = normalize_address(address)
    result: set[str] = set()
    for key in _keys(name, address, country):
        result.update(row[0] for row in connection.execute(
            "SELECT entity_id FROM exact_keys WHERE key_type = ? AND key_value = ?",
            key,
        ))
    query = " OR ".join(_fts_query(normalized_name, name_k))
    if query:
        result.update(row[0] for row in connection.execute(
            "SELECT entity_id FROM name_fts WHERE name_fts MATCH ? LIMIT ?", (query, name_k)
        ))
    query = " OR ".join(_fts_query(normalized_address, address_k))
    if query:
        result.update(row[0] for row in connection.execute(
            "SELECT entity_id FROM address_fts WHERE address_fts MATCH ? LIMIT ?", (query, address_k)
        ))
    return result


def generate_candidates_file(
    source1_path: str | Path,
    database_path: str | Path,
    output_path: str | Path,
    *,
    name_k: int = 30,
    address_k: int = 20,
) -> None:
    """Stream Source 1 rows and write the final candidate set as TSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    try:
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["source1_entity_id", "candidate_entity_ids"])
            for index, row in enumerate(_rows(source1_path), start=1):
                ids = sorted(candidates(
                    connection,
                    row["business_name"],
                    row["business_address"],
                    row["country"],
                    name_k=name_k,
                    address_k=address_k,
                ))
                writer.writerow([row["entity_id"], ",".join(ids)])
                if index % 10_000 == 0:
                    LOG.info("Generated candidates for %,d Source 1 rows", index)
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument("--source1", help="optional Source 1 TSV to process after indexing")
    parser.add_argument("--output", help="candidate TSV output when --source1 is provided")
    parser.add_argument("--name-k", type=int, default=30)
    parser.add_argument("--address-k", type=int, default=20)
    parser.add_argument("sources", nargs="+", help="Source 2/3 TSV files")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    build_index(args.sources, args.database)
    if bool(args.source1) != bool(args.output):
        parser.error("--source1 and --output must be supplied together")
    if args.source1:
        generate_candidates_file(
            args.source1,
            args.database,
            args.output,
            name_k=args.name_k,
            address_k=args.address_k,
        )


if __name__ == "__main__":
    main()