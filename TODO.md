# TODO:
Look up Cantera

urdf generic launch CLI test:
`xacro robot.urdf.xacro > test.urdf end_effector_type:=mock_pruner eef_parent:=ur5e__tool0 arm_type:=ur5 ur_type:=ur5e tf_prefix:=ur5e__ base_attachment_type:=linear_slider`

## TODO
1. Figure out better sensor -> camera -> ToF inheritance (ask Cindy)
1. Tree/robot/env interaction
1. For Claire: 
    1. Figure out best way to manage tree/robot/environment interaction. I removed robot from penv, but self.trees still exists. 
    1. Fill out the `object_loader.py` class. Activate/deactivate trees, supports, robots.
    1. Find the `TODO`s in all the code. Ask Luke what they meana and discuss solutions.
1. Format the final approach controller as a python subpackage?
    1. https://packaging.python.org/en/latest/guides/packaging-namespace-packages/#packaging-namespace-packages
1. Add basic cylinder to world. Dynamically create URDF.
1. Dynamically populate UR URDF. Allow for various end-effectors and robot configurations.
    1. Make sure to include camera and other sensors. (Source manifold mesh -- utils -> camera class (C++))
    1. Dynamic parent joint for Panda to slider/farm-ng (like UR5)
1. Add generic robot class (from Abhinav's code)
    1. Panda/UR5
    1. End-effector
1. Make mini orchard from a set of URDFs
    1. Space out like normal orchard
    1. Be aware of memory issues in PyB
1. Change pkl files to hdf5. (lower priority) -- Ask Abhinav for code
1. Test
    1. Various tof configurations


### Useful RegEx commands...
For replacing .mtl file paths
```
From: ([a-zA-Z0-9_]*).mtl\n
To:   ../mtl/$1.mtl\n

```
