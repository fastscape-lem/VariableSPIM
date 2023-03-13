# VariableSPIM
Variable Discharge Stream Power Incision Model

This repository contains the code for a FastScape process that includes the effect of discharge variability arising from a variable rainfall on erosional efficiency assuming a discharge threshold that may depend on slope. The full mathematical description of the model can be obtained from the following preprint of a manuscript submitted to JGR-Earth Surface on February 28, 2023:
[Archive Doc](https://essopenarchive.org/doi/full/10.22541/essoar.167810329.99741471/v1)

The main inputs to the model are:
- daily rainfall frequency
- storm depth
- storm size
- soil moisture capacity
- potential evapo-transpiration
- daily stream power law exponent
- recession exponent (values of b=1 or b=2 only implemented)
- threshold erosion rate (rescaled discharge threshold)
- slope dependence of threshold (exponent)

in addition to the usual SPIM slope (n) and area (m) exponents and the rate coefficient, Kf

