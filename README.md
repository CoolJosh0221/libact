# libact: Pool-based Active Learning in Python

authors: [Yao-Yuan Yang](http://yyyang.me), Shao-Chuan Lee, Yu-An Chung, Tung-En Wu, Si-An Chen, [Hsuan-Tien Lin](http://www.csie.ntu.edu.tw/~htlin)

[![Build Status](https://github.com/ntucllab/libact/actions/workflows/tests.yml/badge.svg)](https://github.com/ntucllab/libact/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/libact/badge/?version=latest)](http://libact.readthedocs.org/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/libact.svg)](https://badge.fury.io/py/libact)
[![codecov.io](https://codecov.io/github/ntucllab/libact/coverage.svg?branch=master)](https://codecov.io/github/ntucllab/libact?branch=master)

## Introduction

`libact` is a Python package designed to make active learning easier for
real-world users. The package not only implements several popular active learning strategies, but also features the [active-learning-by-learning](http://www.csie.ntu.edu.tw/~htlin/paper/doc/aaai15albl.pdf)
meta-algorithm that assists the users to automatically select the best strategy
on the fly. Furthermore, the package provides a unified interface for implementing more strategies, models and application-specific labelers. The package is open-source along with issue trackers on github, and can be easily installed from Python Package Index repository.

## Platform Support

libact supports the following platforms:
- **Linux** (x86_64) - Full support with pre-built wheels
- **macOS** (Intel x86_64 and Apple Silicon ARM64) - Full support with pre-built wheels
- **Windows** (x86_64) - Core features supported, optional BLAS features may require additional setup

## Documentation

The technical report associated with the package is on [arXiv](https://arxiv.org/abs/1710.00379), and the documentation for the latest release is available on [readthedocs](http://libact.readthedocs.org/en/latest/).
Comments and questions on the package is welcomed at `libact-users@googlegroups.com`. All contributions to the documentation are greatly appreciated!

## Installation

### Quick Install (Recommended)

The easiest way to install libact is via pip:

```shell
pip install libact
```

This will install pre-built wheels with all dependencies. Core features work out of the box on all platforms.

### Checking Available Features

After installation, you can check which optional features are available:

```python
import libact
libact.check_optional_features()
```

This will show output like:
```
libact Optional Features Status
========================================
  variance_reduction: ✓ Available
  hintsvm: ✓ Available
========================================
```

### Platform-Specific Notes

#### Linux (Debian/Ubuntu)

For full feature support including HintSVM and VarianceReduction:

```shell
# Install BLAS/LAPACK (optional, for full features)
sudo apt-get install libopenblas-dev liblapacke-dev

# Install libact
pip install libact
```

#### Linux (Fedora/RHEL/CentOS)

```shell
# Install BLAS/LAPACK (optional, for full features)
sudo dnf install openblas-devel lapack-devel

# Install libact
pip install libact
```

#### Linux (Arch Linux)

```shell
# Install BLAS/LAPACK (optional, for full features)
sudo pacman -S openblas lapacke

# Install libact
pip install libact
```

#### macOS

Using Homebrew (recommended):

```shell
# Install BLAS/LAPACK (optional, for full features)
brew install openblas

# Install libact
pip install libact
```

macOS also uses Apple's Accelerate framework by default, which provides BLAS/LAPACK functionality.

#### Windows

Basic installation:

```shell
pip install libact
```

For full feature support on Windows, use conda:

```shell
# Using conda (recommended for full features)
conda install -c conda-forge openblas
pip install libact
```

Alternatively, install Intel MKL:

```shell
pip install mkl
pip install libact
```

**Note:** On Windows without BLAS/LAPACK, the core libact features work perfectly. Only HintSVM and VarianceReduction query strategies require BLAS/LAPACK.

### Install from Source

For development or to install from the latest source:

```shell
# Clone the repository
git clone https://github.com/ntucllab/libact.git
cd libact

# Create and activate conda environment (recommended)
conda env create -f environment.yml
conda activate libact

# Install in development mode
pip install --no-build-isolation -e .
```

### Install Specific Version

```shell
# Install a specific version
pip install libact==0.1.6

# Install the latest development version
pip install git+https://github.com/ntucllab/libact.git
```

## Python Version Support

- Python 3.9
- Python 3.10
- Python 3.11
- Python 3.12

## Dependencies

### Required Dependencies (installed automatically)

- numpy >= 2
- scipy >= 1.13
- scikit-learn >= 1.6
- matplotlib >= 3.8
- joblib == 1.5.1

### Optional Dependencies (for full features)

- BLAS library (OpenBLAS, MKL, or Apple Accelerate)
- LAPACK library

## Build Options

This package supports the following build options:

- `blas`: BLAS library to use (default='auto'). Options: `auto`, `openblas`, `Accelerate`, `mkl`, `lapack`, `blis`.
- `lapack`: LAPACK library to use (default=`auto`). Options: `auto`, `openblas`, `Accelerate`, `mkl`, `lapack`, `blis`.
- `variance_reduction`: Build variance reduction module (default: true)
- `hintsvm`: Build hintsvm module (default: true)

### Examples

To install `libact` with the default configuration:

```shell
pip install libact
```

Install without optional modules:

```shell
pip install libact --config-settings=setup-args="-Dvariance_reduction=false" \
                    --config-settings=setup-args="-Dhintsvm=false"
```

## Usage

The main usage of `libact` is as follows:

```python
from libact.base.dataset import Dataset
from libact.query_strategies import UncertaintySampling

# Create a query strategy instance
qs = UncertaintySampling(trn_ds, method='lc')

# Let the query strategy suggest a data point to query
ask_id = qs.make_query()

# Get the data and query its label
X, y = zip(*trn_ds.data)
lb = lbr.label(X[ask_id])

# Update the dataset with the newly queried label
trn_ds.update(ask_id, lb)
```

Some examples are available under the `examples` directory. Before running, use
`examples/get_dataset.py` to retrieve the dataset used by the examples.

### Available Query Strategies

**Core Strategies (always available):**
- `UncertaintySampling` - Query instances with highest uncertainty
- `RandomSampling` - Query random instances
- `QueryByCommittee` - Query instances with highest disagreement among committee
- `QUIRE` - Query by informative and representative examples
- `DWUS` - Density-weighted uncertainty sampling
- `ActiveLearningByLearning` - Meta-algorithm for strategy selection
- `DensityWeightedMeta` - Density-weighted meta-learning

**Optional Strategies (require BLAS/LAPACK):**
- `HintSVM` - Hint-based SVM for active learning
- `VarianceReduction` - Variance reduction sampling

### Available Examples

- [plot](examples/plot.py): Basic usage of libact with pool-based active learning
- [label_digits](examples/label_digits.py): Human-in-the-loop labeling example
- [albl_plot](examples/albl_plot.py): Comparison of ALBL with other algorithms
- [multilabel_plot](examples/multilabel_plot.py): Multi-label active learning
- [alce_plot](examples/alce_plot.py): Cost-sensitive multi-class active learning

## Running Tests

To run the test suite:

```shell
python -m unittest -v
```

To run pylint:

```shell
pip install pylint
pylint libact
```

To measure test code coverage:

```shell
pip install coverage
python -m coverage run --source libact --omit */tests/* -m unittest
python -m coverage report
```

## Citing

If you find this package useful, please cite the original works (see Reference of each strategy) as well as the following:

```bibtex
@techreport{YY2017,
  author = {Yao-Yuan Yang and Shao-Chuan Lee and Yu-An Chung and Tung-En Wu and Si-An Chen and Hsuan-Tien Lin},
  title = {libact: Pool-based Active Learning in Python},
  institution = {National Taiwan University},
  url = {https://github.com/ntucllab/libact},
  note = {available as arXiv preprint \url{https://arxiv.org/abs/1710.00379}},
  month = oct,
  year = 2017
}
```

## Acknowledgments

The authors thank Chih-Wei Chang and other members of the [Computational Learning Lab](https://learner.csie.ntu.edu.tw/) at National Taiwan University for valuable discussions and various contributions to making this package better.
