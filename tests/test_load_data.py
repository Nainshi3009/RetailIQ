"""Tests for CSV loading and error handling."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.load_data import load_csv


class LoadCsvTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.temp_path = Path(self.temp_dir.name)

    def test_loads_utf8_csv(self) -> None:
        csv_path = self.temp_path / "utf8.csv"
        csv_path.write_text("customer,city\nA,Montréal\n", encoding="utf-8")

        data = load_csv(csv_path)

        self.assertEqual(data.shape, (1, 2))
        self.assertEqual(data.loc[0, "city"], "Montréal")

    def test_falls_back_to_cp1252_csv(self) -> None:
        csv_path = self.temp_path / "cp1252.csv"
        csv_path.write_bytes("customer,city\nA,Montréal\n".encode("cp1252"))

        data = load_csv(csv_path)

        self.assertEqual(data.loc[0, "city"], "Montréal")

    def test_does_not_ignore_bytes_invalid_in_both_encodings(self) -> None:
        csv_path = self.temp_path / "invalid-encoding.csv"
        csv_path.write_bytes(b"customer,city\nA,\x81\n")

        with self.assertRaises(UnicodeDecodeError):
            load_csv(csv_path)

    def test_missing_file_raises_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_csv(self.temp_path / "missing.csv")

    def test_empty_file_raises_empty_data_error(self) -> None:
        csv_path = self.temp_path / "empty.csv"
        csv_path.write_bytes(b"")

        with self.assertRaises(pd.errors.EmptyDataError):
            load_csv(csv_path)

    def test_malformed_file_raises_parser_error(self) -> None:
        csv_path = self.temp_path / "malformed.csv"
        csv_path.write_text('customer,city\n"A,Montréal\n', encoding="utf-8")

        with self.assertRaises(pd.errors.ParserError):
            load_csv(csv_path)


if __name__ == "__main__":
    unittest.main()
