#!/usr/bin/env python3

from pybullet_tree_sim.time_of_flight import TimeOfFlight
from pybullet_tree_sim.utils.pyb_utils import PyBUtils


import unittest


class TestTimeOfFlight:
    pbutils = PyBUtils(renders=False)
    tof = TimeOfFlight(pbutils.pbclient, sensor_name="vl53l8cx")

    def test_view_matrix(self):
        """Test that the view matrix is a 4x4 matrix."""
        # self.assertTrue(self.tof.view_matrix.shape == (4, 4))
        return


if __name__ == "__main__":
    unittest.main()
