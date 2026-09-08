from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'waiter_robot'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml') + glob('config/*.xml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='HRAI Student',
    maintainer_email='student@hrai.local',
    description='Emotionally-aware waiter robot',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'arm_controller = waiter_robot.arm_controller:main',
            'grasp_test = waiter_robot.grasp_test:main',
        ],
    },
)
