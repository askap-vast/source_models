# source_models
Source models for fitting

## Contents:
* `nova.py`: self-similar nova model with free-free absorption/emission, following [Hjellming et al. (1979)](https://ui.adsabs.harvard.edu/abs/1979AJ.....84.1619H/abstract)


## Examples:
```
from nova import NovaShell
n = NovaShell(
    Te=1e4 * u.K,
    M=8.6e-5 * u.Msun,
    d=800 * u.pc,
    v2=450 * u.km / u.s,
    v1=0.44 * 450 * u.km / u.s,
    )
print(n.Snu(1e9*u.Hz, np.array([1000,2000])*u.d))
```
