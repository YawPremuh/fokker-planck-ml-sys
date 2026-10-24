# FluxFP — Fokker–Planck ML Systems

A Mac-first C++ and PyTorch project investigating whether a neural network can approximate probability forecasts faster than a numerical Fokker–Planck solver while maintaining useful accuracy.

The project models a one-dimensional Ornstein–Uhlenbeck process: noisy dynamics that tend to return toward a long-term average. It predicts a probability distribution rather than a single future value.

## Pipeline

```text
Process parameters
       ↓
C++ Fokker–Planck solver
       ↓
Numerical training dataset
       ↓
Training / validation / test split
       ↓
PyTorch neural surrogate
       ↓
Accuracy evaluation and CPU/MPS benchmarks
```

## Progress

- [x] Implement the analytical Gaussian reference.
- [x] Implement the C++ numerical solver.
- [x] Test probability conservation, nonnegative densities, and grid convergence.
- [x] Plot numerical and analytical distributions.
- [x] Generate and validate 3,000 numerical examples.
- [x] Prepare training, validation, and test splits.
- [x] Standardize inputs using training-set statistics only.
- [x] Train the neural surrogate on CPU.
- [x] Save the checkpoint with the lowest validation error.
- [x] Evaluate the selected model on the untouched test set.
- [x] Compare model predictions against the analytical reference.
- [ ] Benchmark inference against numerical and analytical methods.
- [ ] Compare CPU and Apple MPS performance where available.

## Numerical solver

The solver uses conservative finite volumes with upwind drift, centered diffusion, explicit time stepping, and zero-flux boundaries on `[-6, 6]`.

For the default forecast at time 1:

| Grid cells | L1 error against analytical reference |
|---:|---:|
| 100 | 0.0606662 |
| 200 | 0.0316461 |
| 400 | 0.0162245 |
| 800 | 0.00823309 |

The error approximately halves when the grid resolution doubles, showing approximately first-order convergence for this example.

The 400-cell run used 1,309 time steps and preserved total probability to numerical precision.

## Dataset

Each example contains four inputs and a numerical density evaluated on 256 grid cells.

| Input | Sampled range |
|---|---|
| Mean-reversion strength (`theta`) | 0.5–2.0 |
| Long-term mean (`mean`) | −1.0–1.0 |
| Noise strength (`sigma`) | 0.4–1.0 |
| Forecast time | 0.1–2.0 |

The initial distribution is fixed: Gaussian with mean 0 and standard deviation 0.3.

| Split | Examples |
|---|---:|
| Training | 2,100 |
| Validation | 450 |
| Test | 450 |

Dataset checks cover shape, finite values, nonnegative densities, parameter ranges, probability normalization, and duplicate inputs.

The numerical labels have a mean L1 error of **0.0298408** and a maximum of **0.0849119** against the analytical reference.

Input scaling is fitted only on training data. Target densities are converted into cell probabilities before training.

## Neural surrogate

The network maps four standardized inputs to 256 cell probabilities:

```text
4 inputs → 128 + SiLU → 128 + SiLU → 256 → Log-softmax
```

The model contains **50,176 trainable parameters**. Exponentiating its outputs produces nonnegative probabilities that sum to approximately one.

Training uses Adam and cross-entropy against the numerical target distributions. Checkpoint selection uses validation L1 distance.

### CPU training results

| Metric | Recorded result |
|---|---:|
| Training epochs | 300 |
| Batch size | 128 |
| CPU threads | 1 |
| Best epoch | 297 |
| Best validation L1 | 0.009197 |
| Training-loop time | 2.56 seconds |

These are results from one local CPU run. Training time excludes installation and data preparation and is not an inference benchmark.

The checkpoint is saved to `artifacts/cpu/model.pt`, with training history and settings in `artifacts/cpu/training.json`.

**Validation L1 measures agreement with the numerical solver, not the analytical solution. It is a distribution distance, not an accuracy percentage. Test-set results and inference speedups have not yet been established.**

### Held-out test evaluation

The selected checkpoint from **epoch 297** was evaluated on **450 test examples** that were excluded from training and checkpoint selection.

| Comparison | Mean L1 | Median L1 | 95th-percentile L1 | Maximum L1 |
|---|---:|---:|---:|---:|
| Model vs numerical solver | 0.008956 | 0.007544 | 0.019172 | 0.045389 |
| Numerical solver vs analytical reference | 0.030808 | 0.029030 | 0.056792 | 0.071535 |
| Model vs analytical reference | 0.031935 | 0.029986 | 0.057671 | 0.098562 |

The model’s mean test L1 against the solver was close to its best validation L1 of **0.009197**, supporting generalization to unseen combinations within the sampled parameter ranges.

The model closely approximates its numerical teacher, but inherits numerical-label limitations. Its mean error against the analytical reference is slightly higher than the solver’s, and its worst-case error is larger. These pairwise errors are not additive.

Additional checks:

| Metric | Result |
|---|---:|
| Model mean absolute mean error vs analytical reference | 0.006074 |
| Model mean absolute tail-probability error vs analytical reference | 0.001982 |
| Maximum model probability-mass error | 4.639e-07 |

Tail probability is approximated using cells whose centers satisfy `x > 1`. The reported error is absolute; it does not establish relative accuracy for extremely rare events.

Evaluation runs on CPU without updating model weights:

```bash
python python/evaluate.py
```

The script verifies the dataset fingerprint and test split, then saves:

- `results/evaluation_cpu/metrics.json`
- `results/evaluation_cpu/predictions.npz`

These results measure accuracy only. No inference speedup has been demonstrated yet.

## Build and run

Requires a C++20 compiler, CMake, and a compatible Python environment.

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy matplotlib torch
```

Build and test the C++ programs:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Generate and plot a numerical forecast:

```bash
./build/fluxfp
python python/plot_comparison.py
```

Generate, validate, and prepare the dataset:

```bash
./build/fluxfp_dataset
python python/check_dataset.py
python python/prepare_data.py
```

Train the CPU model:

```bash
python python/train.py --device cpu
```

The training script also supports MPS when available:

```bash
python python/train.py --device mps
```

MPS training has not yet been evaluated in the reported results. Running these commands again overwrites the corresponding generated outputs.

## Limitations

- The numerical method introduces artificial diffusion, which the surrogate can learn from its training labels.
- The solver uses finite reflecting boundaries, while the analytical solution assumes an infinite domain.
- Held-out combinations within the sampled ranges measure interpolation, not extrapolation.
- Small overall distribution error does not guarantee accurate relative probabilities for extremely rare events.
- Ornstein–Uhlenbeck dynamics already have an inexpensive analytical solution. That formula will be included as a performance baseline.
- This is a numerical computing and ML systems experiment using synthetic data, not evidence of a profitable trading strategy.

