from distutils.core import setup

setup(
    name='correct_il',
    version='0.0.1',
    packages=['correct_il', 'correct_il.models', 'correct_il.envs'],
    license='MIT',
    long_description=open('README.md').read(),
    install_requires=[
        "tqdm",
        "tabulate",
        "torch==2.1.0",
        "gym==0.19.0",
        "matplotlib==3.7.2",
        "pygame==2.5.0",
        "PyYAML",
        "d4rl==1.1",
        "tensorboard",
        "scipy",
        "mjrl @ git+https://github.com/aravindr93/mjrl@3871d93763d3b49c4741e6daeaebbc605fe140dc",
        "d3rlpy @ git+https://github.com/personalrobotics/d3rlpy.git@e89a7da6e4c1020f5cf999346c0ebd32e0bbc7c1#egg=d3rlpy",
        "f110_gym @ git+https://github.com/f1tenth/f1tenth_gym.git@cd56335eda43ff4e401331c461877227474a3ed4#egg=f110_gym",
        "metaworld @ git+https://github.com/rlworkgroup/metaworld.git@84bda2c3bd32fc03bb690d6188b22c7946cdb020",
        "gym-pybullet-drones @ git+https://github.com/personalrobotics/gym-pybullet-drones.git@2d4f34529722d57875073024e5c13fa80edf1072#egg=gym_pybullet_drones"
    ]
)
