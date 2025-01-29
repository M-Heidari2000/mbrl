from setuptools import setup, find_packages

setup(
    name="mbrl",
    version="0.0.0",
    description="envs and utils for Model Based RL",
    url="https://github.com/M-Heidari2000/mbrl",
    packages=find_packages(),
    requires=[
        "gymnasium",
        "gymnasium[mujoco]",
        "numpy",
    ]
)