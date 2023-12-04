from glob import glob
import os
from setuptools import setup

package_name = 'assignment_4'

setup(
    name=package_name,
    version='3.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.py')),
        (os.path.join('share', package_name), glob('maps/*.json')),
        (os.path.join('share', package_name), glob('urdf/*.xacro')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jenkin',
    maintainer_email='jenkin@yorku.ca',
    description='Code associated with Chapter 2 of CPMR 3rd Edition',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'path_finder = assignment_4.path_finder:main',
            'state_machine = assignment_4.state_machine:main',
            'add_obstacle = assignment_4.add_obstacle:main',
            'build_map = assignment_4.build_map:main',
            'destroy_map = assignment_4.destroy_map:main',
        ],
    },
)
