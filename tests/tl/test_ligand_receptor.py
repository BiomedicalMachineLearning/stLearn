#!/usr/bin/env python

"""Tests for LRPairs and associated functions."""

import unittest

import numpy as np
import pandas as pd

from stlearn.tl.cci import LRPairs

# Rows in the layout of connectomeDB2025's csv download (current release).
TEST_DATABASE = pd.DataFrame(
    {
        "Interaction ID": ["CDB00001", "CDB00002", "CDB00003", "CDB00004"],
        "LR Pair": ["CCN2 LRP1", "THBS1 LRP1", "CXCL12 CXCR4", "Ccl21a Ccr7"],
        "Evidence": ["Direct", "Inferred", "Direct", "absent in human"],
        "Ligand Symbols": [
            "CCN2 (CTGF, HCS24)",
            "THBS1 (TSP1)",
            "CXCL12 (SDF1)",
            "no human ortholog",
        ],
        "Receptor Symbols": [
            "LRP1 (A2MR, APOER)",
            "LRP1 (A2MR, APOER)",
            "CXCR4",
            "no human ortholog",
        ],
    }
)


def lr_test_pairs() -> LRPairs:
    return LRPairs.from_frame(
        TEST_DATABASE,
        "Ligand Symbols",
        "Receptor Symbols",
        species="human",
        source="cdb",
    )


class TestLRPairs(unittest.TestCase):
    def test_from_frame_connectomedb2025_layout(self):
        """Both releases' column spellings are found, aliases are dropped, and pairs
        without a human ortholog are skipped; other columns are kept."""
        cdb2025v1 = TEST_DATABASE.rename(columns=lambda name: name.replace(" ", "_"))
        for df in [TEST_DATABASE, cdb2025v1]:
            lrs = LRPairs.from_frame(
                df, "Ligand Symbols", "Receptor Symbols", species="human"
            )
            np.testing.assert_array_equal(
                lrs.names(), ["CCN2_LRP1", "CXCL12_CXCR4", "THBS1_LRP1"]
            )
            columns = list(lrs.to_frame().columns)
            self.assertEqual(columns[:3], ["ligand", "receptor", "source"])
            self.assertIn("Evidence", columns)

    def test_filter_union_genes(self):
        lrs = lr_test_pairs()
        direct = lrs.filter(lambda d: d["Evidence"] == "Direct")
        np.testing.assert_array_equal(direct.names(), ["CCN2_LRP1", "CXCL12_CXCR4"])
        self.assertEqual((len(lrs), direct.species), (3, "human"))

        masked = lrs.filter(np.array([True, False, True]))
        np.testing.assert_array_equal(masked.names(), direct.names())

        mine = LRPairs.from_names(
            ["CXCL12_ACKR3", "CCN2_LRP1"], species="human", source="hand-picked"
        )
        both = direct | mine
        np.testing.assert_array_equal(
            both.names(), ["CCN2_LRP1", "CXCL12_ACKR3", "CXCL12_CXCR4"]
        )
        np.testing.assert_array_equal(
            both.genes(), ["ACKR3", "CCN2", "CXCL12", "CXCR4", "LRP1"]
        )

        none = lrs.filter(lambda d: d["Evidence"] == "Retracted")
        self.assertEqual((len(none), none.names().size), (0, 0))

    def test_sources_are_tracked(self):
        """Each pair keeps where it came from, through filter and union; a pair from
        more than one place lists them all."""
        lrs = lr_test_pairs()
        mine = LRPairs.from_names(
            ["CXCL12_ACKR3", "CCN2_LRP1"], species="human", source="hand-picked"
        )
        extra = LRPairs.from_names(
            ["CCN2_LRP1", "SPP1_CD44"], species="human", source="extra.csv"
        )
        combined = lrs | mine | extra
        self.assertEqual(combined.sources, ("cdb", "hand-picked", "extra.csv"))
        np.testing.assert_array_equal(
            combined.source_of(["CCN2_LRP1", "CXCL12_ACKR3", "SPP1_CD44", "A_B"]),
            np.array(
                ["cdb | hand-picked | extra.csv", "hand-picked", "extra.csv", None]
            ),
        )
        # A merged pair keeps the first value given for the other columns.
        evidence = combined.to_frame().set_index(["ligand", "receptor"])["Evidence"]
        self.assertEqual(evidence[("CCN2", "LRP1")], "Direct")

        # Sources are in the order added, not the order of the pairs they overlap.
        late = LRPairs.from_names(["CXCL12_CXCR4"], species="human", source="late")
        early = LRPairs.from_names(["CCN2_LRP1"], species="human", source="early")
        self.assertEqual((lrs | late | early).sources, ("cdb", "late", "early"))
        self.assertEqual(lrs.filter(lambda d: d["ligand"] == "CCN2").sources, ("cdb",))

        only_mine = combined.filter(lambda d: d["source"].str.contains("hand-picked"))
        np.testing.assert_array_equal(only_mine.names(), ["CCN2_LRP1", "CXCL12_ACKR3"])
        self.assertEqual(
            repr(only_mine), "LRPairs(2 human pairs from cdb, hand-picked, extra.csv)"
        )

    def test_species_must_match(self):
        human = lr_test_pairs()
        mouse = LRPairs.from_names(["Cxcl12_Cxcr4"], species="mouse", source="mm")
        with self.assertRaisesRegex(
            ValueError, r"different species: human \(cdb\), mouse \(mm\)"
        ):
            human | mouse
        with self.assertRaisesRegex(ValueError, "different species"):
            human.union(human, mouse)
        with self.assertRaises(TypeError):
            LRPairs.from_names(["CXCL12_CXCR4"])  # species is required
        self.assertEqual(
            LRPairs.from_names(["Cxcl12_Cxcr4"], species="Mouse").species, "mouse"
        )

    def test_immutable(self):
        lrs = lr_test_pairs()
        frame = lrs.to_frame()
        frame.loc[0, "ligand"] = "XXX"

        def keep_all_after_overwriting(d: pd.DataFrame) -> np.ndarray:
            d["ligand"] = "XXX"
            return np.ones(len(d), dtype=bool)

        kept = lrs.filter(keep_all_after_overwriting)
        np.testing.assert_array_equal(kept.names(), lrs.names())
        self.assertIn("CCN2_LRP1", lrs)
        self.assertEqual(len(lrs), 3)
        with self.assertRaises(AttributeError):
            lrs.species = "mouse"

    def test_from_names(self):
        lrs = LRPairs.from_names(
            np.array(["CXCL12_CXCR4", "CXCL12_CXCR4", "CCN2_LRP1"]), species="human"
        )
        np.testing.assert_array_equal(lrs.names(), ["CCN2_LRP1", "CXCL12_CXCR4"])
        self.assertEqual(repr(lrs), "LRPairs(2 human pairs from names)")
        for bad in ["CXCL12", "A_B_C", "CXCL12_ CXCR4"]:
            with self.assertRaisesRegex(ValueError, "LIGAND_RECEPTOR"):
                LRPairs.from_names([bad], species="human")

    def test_bad_input(self):
        with self.assertRaisesRegex(ValueError, "1 column"):
            LRPairs.from_frame(pd.DataFrame({"ligand": ["CXCL12"]}), species="human")
        with self.assertRaisesRegex(ValueError, "No 'Ligand' column"):
            LRPairs.from_frame(TEST_DATABASE, "Ligand", "Receptor", species="human")
        # Pair labels rather than ligand and receptor columns.
        with self.assertRaisesRegex(ValueError, "No ligand-receptor pairs"):
            LRPairs.from_frame(TEST_DATABASE, species="human")
        with self.assertRaisesRegex(ValueError, "contain '_'"):
            LRPairs.from_frame(
                pd.DataFrame({"l": ["CXCL12"], "r": ["integrin_a4b1"]}),
                species="human",
            )
        with self.assertRaisesRegex(ValueError, "needs 'ligand' and 'receptor'"):
            LRPairs(pd.DataFrame({"a": ["CXCL12"]}), species="human")
        with self.assertRaisesRegex(ValueError, "species must name a species"):
            LRPairs.from_names(["CXCL12_CXCR4"], species="")


if __name__ == "__main__":
    unittest.main()
