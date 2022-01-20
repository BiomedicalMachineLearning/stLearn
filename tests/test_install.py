"""
Tests that everything is installed correctly.
"""

import unittest


class TestCCI(unittest.TestCase):
    """Tests for `stlearn` importability, i.e. correct installation."""

    def test_SME(self):
        import stlearn.spatials.SME.normalize as sme_normalise

    def test_cci(self):
        """Tests CCI can be imported."""
        import stlearn.tools.microenv.cci.analysis as an
