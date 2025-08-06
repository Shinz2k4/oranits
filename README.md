
# ITS Simulation: DRL & Metaheuristic for Joint Task Handling

This project simulates joint task handling and mission processing in Intelligent Transportation Systems (ITS) using Deep Reinforcement Learning (DRL) and metaheuristic approaches.

---

## 1. Environment Setup

```
source setup.sh
```

## 2. Project Structure

```
.
├── src/
│   ├── DRL/                      # DRL algorithms & training scripts
│   ├── physic_definition/       # ITS environment simulation
│   ├── meta_heuristic/          # Metaheuristic methods & analysis
│   └── ...
├── configs/                     # Configuration files
├── task/                        # Output results (auto-created)
├── logs/                        # Runtime logs (auto-created)
├── run.py                       # Main entry point
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 3. Run Simulations

Run simulations with `run.py`:

```bash
python run.py -i <method> [--verbose] [-device <cuda_id>] [-a <analysis_mode>] [-c <comparison_mode>]
```

### Examples:

- Run DDQN:

  ```bash
  python run.py -i ddqn
  ```

- Run many metaheuristics:

  ```bash
  python run.py -i many_metaheuristics
  ```

- Run evaluation of DDQN results:

  ```bash
  python run.py -i eval_ddqn
  ```

- Compare DRL and metaheuristics:

  ```bash
  python run.py -i meta_heuristic_proposal -c drl_and_meta_heuristic_proposal
  ```

- Run analysis mode 1:

  ```bash
  python run.py -i None -a 1
  ```

> `-a` triggers statistical analysis and plotting  
> `-device -1` uses CPU, `0` uses first CUDA GPU

---

## 4. Logging

All logs are automatically saved in the `logs/` folder, for example:

```
./logs/run_log_20250806_153245.log
```

---

## 5. Plotting Style

This project uses the `scienceplots` package for publication-ready matplotlib styles.

---
