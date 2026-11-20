"""Independent release checks using only the standard library.

Run: python -m unittest discover -s tests -p test_independent_data.py -v
These deliberately fail until the prepared dataset exists.
"""
from pathlib import Path
import csv
import itertools
import re
import unicodedata
import unittest

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = {"example_id", "text", "label", "group_id"}
SPLITS = ("train", "validation", "test")


def canonical_input(text):
    text = unicodedata.normalize("NFKC", text).casefold().replace("\u2019", "'")
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


class IndependentDataReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = {}
        cls.columns = {}
        for split in SPLITS:
            path = ROOT / "data" / "processed" / f"{split}.csv"
            if not path.is_file():
                raise AssertionError(f"Missing required frozen split: {path}")
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                cls.columns[split] = set(reader.fieldnames or [])
                cls.rows[split] = list(reader)

    def test_schema_and_nonempty_fields(self):
        for split in SPLITS:
            with self.subTest(split=split):
                self.assertEqual(self.columns[split], REQUIRED_COLUMNS)
                self.assertTrue(self.rows[split])
                for row in self.rows[split]:
                    self.assertTrue(all(str(row.get(k, "")).strip() for k in REQUIRED_COLUMNS))

    def test_ids_are_unique_across_all_partitions(self):
        ids = [row["example_id"] for rows in self.rows.values() for row in rows]
        self.assertEqual(len(ids), len(set(ids)), "Example IDs collide or rows repeat across splits")

    def test_group_ids_do_not_cross_partitions(self):
        groups = {split: {r["group_id"] for r in self.rows[split]} for split in SPLITS}
        for left, right in itertools.combinations(SPLITS, 2):
            with self.subTest(left=left, right=right):
                self.assertFalse(groups[left] & groups[right], "A group crosses partitions")

    def test_canonical_request_text_does_not_cross_partitions(self):
        texts = {split: {canonical_input(r["text"]) for r in self.rows[split]} for split in SPLITS}
        for left, right in itertools.combinations(SPLITS, 2):
            with self.subTest(left=left, right=right):
                self.assertFalse(texts[left] & texts[right], "Equivalent request text crosses partitions")

    def test_all_27_intents_present_in_every_partition(self):
        labels = {split: {r["label"] for r in self.rows[split]} for split in SPLITS}
        self.assertEqual(len(labels["train"]), 27)
        for split in ("validation", "test"):
            with self.subTest(split=split):
                self.assertEqual(labels[split], labels["train"])

    def test_no_conflicting_labels_for_same_canonical_request(self):
        mapping = {}
        for rows in self.rows.values():
            for row in rows:
                key = canonical_input(row["text"])
                mapping.setdefault(key, set()).add(row["label"])
        conflicts = sum(len(labels) > 1 for labels in mapping.values())
        self.assertEqual(conflicts, 0, "Conflicting canonical inputs need explicit quarantine or adjudication")


if __name__ == "__main__":
    unittest.main()
