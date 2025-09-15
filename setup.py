from setuptools import setup, find_packages

setup(
    name="JobSubmitor",
    version="1.0",
    description="A simple job submission tool for managing and scheduling tasks.",

    # 你要安装的包，通过 setuptools.find_packages 找到当前目录下有哪些包
    packages=find_packages()
)