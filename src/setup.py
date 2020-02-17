from setuptools import setup
import pandas as pd

ser_ver = pd.read_json("./linder/linder_version.json", typ="series")
print(ser_ver)
__version__ = f"{ser_ver.ver_milestone}.{ser_ver.ver_major}.{ser_ver.ver_minor}{ser_ver.ver_remark}"


def readme():
    with open("../README.md", encoding="utf-8") as f:
        return f.read()


setup(
    name="linder",
    version=__version__,
    description="linder is a machine-learning based land use/land cover (LULC) classifier using Sentinel imagery.",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="",
    author=", ".join(["Dr Hamidreza Omidvar", "Dr Ting Sun", ]),
    author_email=", ".join(
        [
            "h.omidvar@reading.ac.uk",
            "ting.sun@reading.ac.uk",
        ]
    ),
    license="GPL-V3.0",
    packages=["linder"],
    package_data={"linder": ["*.json","model.pkl"]},
    # distclass=BinaryDistribution,
    ext_modules=[],
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "eo-learn",
        "xarray",
        "click",  # cmd tool
    ],
    entry_points={
        #   command line tools
        # "console_scripts": [
        #     "suews-run=supy.cmd.SUEWS:SUEWS",
        #     "suews-convert=supy.cmd.table_converter:convert_table_cmd",
        # ]
    },
    include_package_data=True,
    test_suite="nose.collector",
    tests_require=["nose"],
    python_requires="~=3.6",
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
    ],
    zip_safe=False,
)
