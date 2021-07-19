from setuptools import find_packages, setup

setup(
    name="autobert",
    package_dir={"": "src"},
    packages=find_packages("src"),
    version="0.0.1",
    license="MIT",
    description="AutoBert_pytorch",
    author="Jun Yu",
    author_email="573009727@qq.com",
    url="https://github.com/JunnYu/AutoBert_pytorch",
    keywords=["autobert", "pytorch"],
    install_requires=["transformers>=4.8.0"],
)
