import setuptools

setuptools.setup(
    name="impactdeal",
    version="0.0.1",
    author="Dario Cannone",
    description="Tools for data science projects @ImpactDeal2022",
    url="https://github.com/darioka/impactdeal-2022",
    packages=["impactdeal", "impactdeal.config"],
    python_requires=">=3.7",
    install_requires=[
        "pandas>=1.3.5",
    ]
)
