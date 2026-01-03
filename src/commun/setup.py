from setuptools import find_packages, setup

package_name = 'commun'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='leonard',
    maintainer_email='ngoak@islab.snu.ac.kr',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'latency = commun.latency:main',
            'sub = commun.image_code_subscriber:main',
            'pub = commun.image_code_publisher:main',
            'all = commun.image_code_all:main',
            'listener = commun.code_subscriber:main',
            'talker = commun.code_publisher:main',
        ],
    },
)
