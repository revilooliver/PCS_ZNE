# Project Context for Claude

## Project Overview
This is a quantum computing research project developing **Pauli Check Extrapolation (PCE)**, a novel error mitigation technique that extends Pauli Check Sandwiching (PCS) using extrapolation methods. The work is being prepared for journal submission.

## Key Research Contributions
- **PCE**: Combines PCS with extrapolation to the "maximum check" limit (similar to how ZNE extrapolates to zero noise)
- **Completed work**: PCE vs Robust Shadow (RS) estimation on VQE circuits (already written up)
- **Current focus**: Adding PCE vs Zero Noise Extrapolation (ZNE) comparisons for the paper

## Current Active Experiments (PCE vs ZNE)
- **ZNE_vs_PCE_random_cliff_auto.ipynb**: PCE vs ZNE on random Clifford circuits
- **QAOA_exp.ipynb**: PCE vs ZNE on QAOA circuits with p=1
- These are the main experiments for the new results being added to the paper

## Completed Work (Not Currently Active)
- **VQE circuits**: H2 (4-qubit) and H2O (8-qubit) comparisons with Robust Shadow estimation
- This work is finished and already written up in the paper draft

## Key Advantages of PCE
1. No calibration procedure required (unlike RS estimation)
2. Can protect entire circuit, not just portions
3. Works well with inhomogeneous noise models
4. Extrapolates to "maximum checks" rather than requiring noise amplification like ZNE

## Key Files for Current Work
- `ZNE_vs_PCE_random_cliff_auto.ipynb`: Main random Clifford experiments
- `QAOA_exp.ipynb`: QAOA circuit experiments
- `utils/pce_vs_zne_utils.py`: Core utilities for PCE vs ZNE comparisons
- `utils/pauli_checks.py`: PCS implementation

## Paper Status
- Draft available as "PCE_journal_submission (2).pdf"
- PCE vs RS results complete and written
- Currently adding PCE vs ZNE results to strengthen the comparison section