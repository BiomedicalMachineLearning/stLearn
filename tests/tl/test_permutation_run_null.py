import unittest

import numpy as np

import stlearn as st
from stlearn.tl.cci.permutation import empirical_pvals
from tests.utils import CCI_LRS, cci_table


class TestRunNull(unittest.TestCase):
    """st.tl.cci.run passes its `null` on, "codetected" by default."""

    @staticmethod
    def pvals(**kwargs):
        settings = {"min_spots": 2, "distance": 1.5, "n_pairs": 100, "verbose": False}
        lr = st.tl.cci.run(cci_table(), CCI_LRS, **settings, **kwargs)
        lr = lr[:, CCI_LRS.names()]
        return st.tl.cci.lr_result(lr, "p_vals"), lr

    def test_null_reaches_the_test(self):
        default, _ = self.pvals()
        codetected, _ = self.pvals(null="codetected")
        random, lr = self.pvals(null="random", save_bg=True)
        np.testing.assert_array_equal(default, codetected)
        for null, pvals in (("random", random), ("codetected", codetected)):
            for j, name in enumerate(CCI_LRS.names()):
                spots = lr.uns["lr_spot_indices"][name]
                scores = st.tl.cci.lr_result(lr, "lr_scores", name)[spots]
                expected = empirical_pvals(scores, lr.uns["lrs_to_bg"][name], null)
                np.testing.assert_array_equal(pvals[spots, j], expected, err_msg=null)
        self.assertFalse(np.array_equal(random, codetected))
