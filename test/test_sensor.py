#!/usr/bin/env python3
from pybullet_tree_sim.sensor import Sensor
from pybullet_tree_sim.utils.pyb_utils import PyBUtils

import unittest


class TestSensor(unittest.TestCase):
    # pbutils = PyBUtils(renders=False)

    def test_file_load(self):
        sensor = Sensor(sensor_name="vl53l8cx", sensor_type="tof")
        self.assertTrue(sensor.params is not None)

        with self.assertRaises(FileNotFoundError):
            sensor = Sensor(sensor_name="hello", sensor_type="world")
        return


if __name__ == "__main__":
    unittest.main()
