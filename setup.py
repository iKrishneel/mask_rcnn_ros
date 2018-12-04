
## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=['object_detector'],
    package_dir={'': 'scripts'},
    requires=['std_msgs', 'rospy', 'sensor_msgs', 'geometry_msgs', 'cv_bridge']
)

setup(**setup_args)
