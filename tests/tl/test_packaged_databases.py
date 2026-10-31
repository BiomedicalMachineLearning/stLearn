#!/usr/bin/env python

"""Tests loading packaged databases."""

import re
import unittest

import numpy as np

import stlearn as st


class TestPackagedDatabases(unittest.TestCase):
    # Testing loading lr database (deprecated load_lrs).
    def test_load_lrs(self):
        sizes = [2293, 3550, 3228]  # 2020 and 2025 db sizes (human then mouse).
        with self.assertWarns(FutureWarning):
            lrs = st.tl.cci.load_lrs(["connectomeDB2020"])
        self.assertEqual(sizes[0], len(lrs))

        lrs = st.tl.cci.load_lrs(["connectomeDB2025"])
        self.assertEqual(sizes[1], len(lrs))

        # Default is the old 2020 database.
        lrs = st.tl.cci.load_lrs()
        self.assertEqual(sizes[0], len(lrs))

        # Test without a database but just species
        with self.assertRaisesRegex(ValueError, "connectomeDB2020 only has human LRs."):
            st.tl.cci.load_lrs(species="mouse")

        # Testing loading mouse as species - there are 24 repeated lines.
        lrs = st.tl.cci.load_lrs("connectomeDB2025", species="mouse")
        self.assertEqual(sizes[2], len(lrs))

        bad = [lr for lr in lrs if not re.fullmatch(r'[^\s_()"<>]+_[^\s_()"<>]+', lr)]
        self.assertEqual(bad, [])

    def test_default_is_connectomedb2025_human(self):
        lrs = st.tl.cci.lr_pairs()
        self.assertEqual((lrs.species, lrs.sources), ("human", ("connectomeDB2025",)))
        self.assertGreater(len(lrs), 1000)
        self.assertIn("CXCL12_CXCR4", lrs)
        for lr in lrs.names():
            self.assertRegex(lr, r"^[^\s_()]+_[^\s_()]+$")

    def test_connectomedb2020(self):
        lrs = st.tl.cci.lr_pairs("connectomeDB2020")
        self.assertEqual(lrs.sources, ("connectomeDB2020",))

    def test_load_lrs_previous(self):
        """The previous db load_lrs returns what lr_pairs(...).names() does."""
        with self.assertWarns(FutureWarning):
            legacy = st.tl.cci.load_lrs(["connectomeDB2020"])
        put = st.tl.cci.lr_pairs("connectomeDB2020")
        np.testing.assert_array_equal(legacy, put.names())

    def test_species_have_their_own_tables(self):
        with self.assertRaisesRegex(ValueError, "has human LRs."):
            st.tl.cci.lr_pairs("connectomeDB2020", species="mouse")
        with self.assertRaisesRegex(ValueError, "has human and mouse LRs"):
            st.tl.cci.lr_pairs(species="unicorn")
        with self.assertRaisesRegex(ValueError, "Unknown LR database"):
            st.tl.cci.lr_pairs("connectomeDB2019")
        self.assertEqual(st.tl.cci.lr_pairs(species="Human").species, "human")

    def test_mouse_table(self):
        mouse = st.tl.cci.lr_pairs(species="mouse")
        self.assertEqual(mouse.species, "mouse")
        self.assertIn("Cxcl12_Cxcr4", mouse)
        self.assertNotIn("CXCL12_CXCR4", mouse)
        with self.assertRaisesRegex(ValueError, "different species"):
            st.tl.cci.lr_pairs() | mouse

    def test_mouse_table_legacy(self):
        mouse = st.tl.cci.load_lrs("connectomeDB2025", species="mouse")
        self.assertIn("H2-Ea_Cd4", mouse)
        self.assertNotIn("H2-ea_Cd4", mouse)


if __name__ == "__main__":
    unittest.main()
