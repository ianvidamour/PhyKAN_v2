# PhyKAN_v2: Analogue Electronic Filter-Based Kolmogorov-Arnold Networks

**PhyKAN_v2** is a computationally optimised and refactored implementation of Physical Kolmogorov-Arnold Networks (PhyKAN). This framework is specifically designed to emulate analogue electronic filter-based KANs, modelling the physical behaviour of real-world analogue hardware. 

The codebase refactors the original legacy implementation to maximise computational efficiency, conform to standard PyTorch design patterns, and provide an accessible tool for others to implement analogue filter emulations.

---

## Hardware Lookup Tables

The repository includes two look-up table files containing discrete filter configurations:
*   `HighFilterVals.npy`: Discrete filter values achievable for high-pass stages.
*   `LowFilterVals.npy`: Discrete filter values achievable for low-pass stages.

> **Hardware Context:** These values directly map to the discrete hardware constraints of the **Anadigm AN231K04-QUAD4** Field-Programmable Analog Array (FPAA) equipment.

---

## Key Enhancements in Version 2

Compared to the original legacy implementation, `PhyKAN_v2` introduces several critical structural and algorithmic improvements:

*   **Native PyTorch Parameter Management:** Replaces raw Python parameter lists with `nn.ParameterList`, ensuring that model parameters are natively tracked by PyTorch optimizers.
*   **Automatic GPU Migration:** Employs `self.register_buffer` for lookup tables and edge masks, enabling automated, error-free device casting (CPU to GPU).
*   **Vectorised Discretisation via Broadcasting:** Replaced slow nested Python loops with tensor broadcasting to calculate difference matrices instantaneously when matching continuous values to hardware lookups.
*   **Elimination of Complex Numbers:** Restructured the frequency magnitude response maths to use completely real-valued operations via `torch.hypot`. This removes complex number overhead and safeguards against precision collapse.
*   **Optimised Penalty Regularisation:** Reimplemented L1 and entropy penalty terms to use vectorised operations across entire layers at once rather than indexing them piece-by-piece.

---

## Core Architecture & API

The network maps an input tensor to frequency space and applies simulated band-pass filter properties (gains and cutoff frequencies) along its edges.

### Initialisation
```python
from PhyKAN_v2 import PhyKAN_v2
import numpy as np

# Load the Anadigm hardware lookup tables
low_pass_lookup = np.load('LowFilterVals.npy')
high_pass_lookup = np.load('HighFilterVals.npy')

model = PhyKAN_v2(
    KANshape=[2, 5, 1],          # Architecture layout
    filters_per_unit=4,          # Sub-filters per edge
    low_pass_lookup=low_pass_lookup,
    high_pass_lookup=high_pass_lookup,
    lr=1e-3
)