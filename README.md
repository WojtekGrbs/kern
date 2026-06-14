# kern

`kern` is a minimalistic kernel density estimation package backed by optimized C
implementations. It provides exact, approximate, bounded, and multivariate KDE
estimators with a scikit-learn-like API.

## Installation

```console
python -m pip install kern
```

## Example usage
```python
from kern import KernelDensity

model = KernelDensity(bandwidth=0.3).fit([0.0, 0.2, 1.0])
density = model.evaluate([0.1, 0.5])
```

See the [documentation](https://wojtekgrbs.github.io/kern/) for the user guide,
API reference, and source-build options.
