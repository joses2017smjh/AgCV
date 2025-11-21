#!/usr/bin/env python3
from pybullet_tree_sim.utils import helpers

import unittest
from toolz import partial


class TestHelpers(unittest.TestCase):
    def test_negative_dfov(self):
        dfov = -65
        cam_w = 100
        cam_h = 100

        with self.assertRaises(ValueError) as cm:
            helpers.get_fov_from_dfov(cam_w, cam_h, dfov)
        return

    def test_negative_cam_width(self):
        dfov = 65
        cam_w = -100
        cam_h = 100

        with self.assertRaises(ValueError) as cm:
            helpers.get_fov_from_dfov(cam_w, cam_h, dfov)
        return

    def test_negative_cam_height(self):
        dfov = 65
        cam_w = 100
        cam_h = -100

        with self.assertRaises(ValueError) as cm:
            helpers.get_fov_from_dfov(cam_w, cam_h, dfov)
        return


# class TestStringMethods(unittest.TestCase):

#     def test_upper(self):
#         self.assertEqual('foo'.upper(), 'FOO')

#     def test_isupper(self):
#         self.assertTrue('FOO'.isupper())
#         self.assertFalse('Foo'.isupper())

#     def test_split(self):
#         s = 'helloworld'
#         self.assertEqual(s.split(), ['hello', 'world'])
#         # check that s.split fails when the separator is not a string
#         with self.assertRaises(TypeError):
#             s.split(2)

if __name__ == "__main__":
    unittest.main()
