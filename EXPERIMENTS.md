# NRC2 experiments

We compare Simple SF against the identical agent with NRC2 alignment of hidden features to the vector SF readouts on Walker Run and Cheetah Run. The pilot sweeps coefficients 0, 0.01, 0.1, and 1 over three seeds (24 runs, 2M frames each); select a coefficient, confirm it on fresh seeds, then test adaptation when forward/backward rewards alternate.

## Run on Saadiyat / Jubail

```bash
git clone https://github.com/Okyumi/simple_successor_features.git
cd simple_successor_features
bash hpc/setup_saadiyat.sh                # once: creates simple_sfs
bash hpc/submit_saadiyat.sh --stage smoke # 4 short checks, then inspect logs
bash hpc/submit_saadiyat.sh --stage pilot # 24 jobs, at most 4 concurrently
```

The scripts follow `sgcrl`'s Jubail examples: `nvidia`, one GPU, eight CPUs, 64GB, 48 hours, CUDA 11.8, and conda-gcc/11.2.0. They have been checked locally, but have not yet run on the cluster; the smoke stage checks the actual GPU environment before you spend a full training budget.

## Adjust hyperparameters

Edit defaults in `experiment_configs.py` or pass flags; `--set` accepts ordinary Hydra settings and applies the same value to every comparison:

```bash
# Preview only; no training or submission.
python experiment_configs.py --stage pilot --domains walker --seeds 0 1 2 \
  --coefficients 0 0.03 0.1 --set agent.lr=0.0003 --set agent.batch_size=512
# Submit that custom grid.
MAX_PARALLEL=2 bash hpc/submit_saadiyat.sh --stage pilot --domains walker \
  --seeds 0 1 2 --coefficients 0 0.03 0.1 --set agent.lr=0.0003 --set agent.batch_size=512
# 0.1 is an example: replace it with the coefficient selected on pilot seeds.
bash hpc/submit_saadiyat.sh --stage confirm --coefficients 0 0.1
bash hpc/submit_saadiyat.sh --stage transfer --coefficients 0 0.1
```

Other useful overrides: `--set num_train_frames=1000000`, `--set agent.lr_task=0.00001`, `--set agent.hidden_dim=1024`, `--set agent.sf_dim=64`, and `--set agent.nrc2_target=both` (or `feature`/`successor`). Keep architecture and optimization settings matched between zero and nonzero coefficients.

Each submission saves its exact grid, per-run commands, environment versions, TensorBoard metrics, JSONL evaluations, and task-end checkpoints under `/scratch/$USER/simple_sf_runs/`. Override `SF_RUNS_DIR` or `SF_CONDA_ENV` if needed; don't change the checkout while its jobs are queued/running, because the launcher checks the submitted commit. Interrupted jobs are not automatically resumed; a new submission starts fresh.

Compare return AUC and final return across seeds, alongside rank99, feature variance, NRC2 alignment, and readout conditioning. Transfer uses four task periods (8M frames per run); pixels are available via `--stage pixels`, but the initial state-based pilot and penultimate-layer intervention are not exact reproductions of the CRL paper's intermediate residual-block setting.
