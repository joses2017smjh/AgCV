#!/usr/bin/env python3
from pybullet_tree_sim.camera import Camera
from pybullet_tree_sim.utils.pyb_utils import PyBUtils

import numpy as np
import pprint as pp
import unittest


class TestCamera(unittest.TestCase):
    pbutils = PyBUtils(renders=False)
    camera = Camera(pbutils.pbclient, sensor_name="realsense_d435i")

    def test_depth_pixel_coords_range(self):
        """Test that the pixel coordinates are in the range x=[0, depth_width] and y=[0, depth_height]."""
        self.assertTrue(self.camera.depth_pixel_coords[0, 0] == 0)
        self.assertTrue(self.camera.depth_pixel_coords[0, 1] == 0)
        self.assertTrue(self.camera.depth_pixel_coords[-1, 0] == self.camera.depth_width - 1)
        self.assertTrue(self.camera.depth_pixel_coords[-1, 1] == self.camera.depth_height - 1)
        return

    def test_depth_film_coords_range(self):
        """Test that the film coordinates are in the range [-1, 1]."""
        self.assertTrue(self.camera.depth_film_coords[0, 0] > -1)
        self.assertTrue(self.camera.depth_film_coords[0, 1] > -1)
        self.assertTrue(self.camera.depth_film_coords[-1, 0] < 1)
        self.assertTrue(self.camera.depth_film_coords[-1, 1] < 1)
        return

    def test_xy_depth_projection(self):
        """Test whether a depth pixel has be adequately scaled to xy"""
        # depth_width = 8
        # depth_height = 8
        # xy_pixels_order_C = np.array(list(np.ndindex((self.camera.depth_width, self.camera.depth_height))), dtype=int)
        # print(xy_pixels_order_C)
        print(self.camera.depth_pixel_coords)

        return


if __name__ == "__main__":
    unittest.main()
