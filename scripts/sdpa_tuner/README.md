# SDPA forward config tuner

Tools for the tile selection of the fused forward SDPA kernel in
`src/gpu/intel/sdpa/select.cpp`.

The library picks the microkernel tile configuration in two steps: a lookup
table keyed on the full problem description (`fwd_table_data[]`, filled by
the sweeps below), then the hand-tuned table in `configs.cpp` for every
shape without an entry. A cost model ranks the valid configs of a problem;
it is not a selection step in the library, it is what the sweeps use to
choose candidates, and it is exported through the `dnnl_impl_sdpa_fwd_*`
functions of a DNNL_DEV_MODE build for the scripts here. Out of sample it
is 1.08x off the best measured config on the A750 and 1.10x on PVC where
the hand-tuned table is 1.26x and 1.24x, but a config the microkernel
generator rejects fails primitive creation, and only table entries have
been built and verified, so the table is what ships.

Dev-mode knobs: `SDPA_CONFIG=8 ints` and `QUANTIZED_SDPA_CONFIG` force a
config (existing knobs, what the sweeps use to time candidates). Every
selection prints one `fwd_select,...` line at `ONEDNN_VERBOSE=debuginfo=4`
carrying the lookup key, the chosen config and its source (`table`,
`table_dev`, `legacy` or `env`), the hand-tuned pick (`legacy:`), the
config's derived quantities, the device description (`hw:`) and the
problem description (`prb:`). A table key may end in `:eu<N>` to apply only
to devices with N EUs; such a line is preferred over the arch-wide one
(`src:table_dev`). The key ends in `:tk0` or `:tk1` for the K layout: with
K rows contiguous (benchdnn `--ktag=abcd`, the driver default) or with the
head dimension contiguous (`--ktag=abdc`, what frameworks pass). The kernel
differs per layout and decode runs up to 3x faster with the second, so
sweep the layouts your workload uses; `problems_ci.txt` covers both.

## Files

- `sdpa_model.py`: ctypes binding to the cost model in the library. The model
  exists once, in `select.cpp`; a DNNL_DEV_MODE build exports it through the
  `dnnl_impl_sdpa_fwd_*` functions in `select_api.cpp`. The scripts find
  `build/src/dnnl.dll` or `libdnnl.so` under the repo, or take `--lib` /
  `$DNNL_LIB`.
- `tune_fwd_config.py`: runs benchdnn's `--sdpa` driver over problems and
  configs and appends rows to a CSV.
- `fit_fwd_model.py`: reports regret/rank correlation of the model and fits
  coefficients. `--check-recorded` flags a CSV whose recorded estimates came
  from a different model version than the loaded library.
- `analyze_landscape.py`: top configs and one-factor marginals per problem.
- `bake_fit.py`: writes a fit's coefficients into the architecture's case
  of `seed_coefs()` and merges `--table-out` lines into the built-in table.
- `report_fails.py`: configs a sweep could not build that the tiler still
  accepts, and configs that ran but the tiler rejects; both lists should
  be empty before a fit is baked.
- `problems_to_batch.py` and `compare_perf.py`: a benchdnn batch file from
  a problems file, and a problem-by-problem comparison of two perf logs
  (`table` versus `legacy` selection).
- `problems_compute.txt`: compute-bound shapes that pin down the systolic
  terms; `problems_ci.txt`: coverage shapes from the benchdnn harness.
- `problems_f32.txt`: the KPI and coverage regions in f32, which always
  takes the FMA kernel; `problems_quant.txt`: int4/int8 K/V with grouped
  scales and zero points through the in-tree graph cases;
  `problems_f16acc.txt`: f16 accumulation through the graph cases'
  `accumulation_mode`, reachable only on parts without systolic arrays
  (xe_lpg). Fits that use them need `--unfix fma_cycles_per_flop
  --unfix unroll_overhead --unfix dequant_cycles_per_elem`.
- `problems_kpi.txt`: the SDPA KPI shapes (SD UNet, SDXL head sizes, LLM
  prefill and decode with explicit masks, Whisper) reduced to one line per
  selector key, plus int8 K/V cases in the `graph=` form, which runs a
  verbatim benchdnn `--graph` command line for what the `--sdpa` driver
  cannot express. Replace `KPI_DIR` with the directory of the json cases.

## Workflow

1. Build with `DNNL_DEV_MODE=ON` and the GPU runtime.
2. Sweep. Either list problems or draw a design:

       python scripts/sdpa_tuner/tune_fwd_config.py \
           --benchdnn build/tests/benchdnn/benchdnn.exe \
           --design 60 --seed 1 --dtype f16 --topk 12 --random 8 \
           --out sweep_a750.csv --table-out table_a750.txt

   The reference config is re-timed every `--ref-every` runs; stop if the
   drift warning fires repeatedly (the part is throttling). The check only
   sees drift within a problem: the same config measured 0.099 ms in one
   A750 session and 0.044 ms in another on a Whisper decode shape, so keep
   the GPU state constant across the sweeps that feed one fit (no display
   or other GPU load, fixed clocks where the driver allows it) and treat
   step 4's `--compare` as the acceptance test of a table entry. Measurements use
   benchdnn perf mode P with `--max-ms` per config (default 300 ms); the
   10 ms fast mode (`--perf-mode F`) is quicker but leaves the GPU clock
   unsettled and measured 1.5x off on the A750.
   Candidates that fail to build show `status=fail`; `report_fails.py`
   lists the families the tiler still accepts, add them to `fwd_describe()`
   so it stops emitting them.
3. Score the model and fit:

       python scripts/sdpa_tuner/fit_fwd_model.py sweep_a750.csv --check-recorded -v
       python scripts/sdpa_tuner/fit_fwd_model.py sweep_a750.csv --fit --folds 4 --restarts 1

   Before baking, check that the tiler agrees with the sweep on what
   builds, and add a rule to `fwd_describe()` for any family it still
   accepts:

       python scripts/sdpa_tuner/report_fails.py sweep_a750.csv

4. Bake the fit and promote the verified winners to the lookup table:

       python scripts/sdpa_tuner/fit_fwd_model.py sweep_a750.csv --fit --folds 4 --restarts 1 > fit_a750.txt
       python scripts/sdpa_tuner/bake_fit.py --arch xe_hpg --device A750 --fit fit_a750.txt --table table_a750.txt
       clang-format -i src/gpu/intel/sdpa/select.cpp

   Rebuild, run the `sdpa_select` tests, confirm the `seed:` regret of
   `fit_fwd_model.py` now matches the fit's `fitted/train` line, and time
   the shipped selection against the hand-tuned configs on the swept
   shapes; every covered shape should be at or below 1.0:

       python scripts/sdpa_tuner/tune_fwd_config.py --benchdnn ... --problems problems_kpi.txt --compare
5. Second SKU of an architecture (A770 after A750, B570 after B580): run
   `--compare` on the same problems. Shapes that regress get a device-tagged
   line from a sweep with `--table-tag-device`; the arch-wide lines stay
   shared. Occupancy-bound shapes with few work-groups
   are the ones that differ between SKUs.

## Metrics

Regret is `time(config the model ranks first) / time(best measured config)`
per problem. Table entries are 1.0 by construction. Targets before the
model could become a selection step of its own: geomean regret at or below
1.05, worst case at or below 1.15, and a fallback to the hand-tuned config
when the generator rejects the pick.

Fit with `--folds 4 --restarts 1` and quote the cross-validated line: it is
the out-of-sample regret over every problem, each scored by a fit that did
not see it. The in-sample `fitted/train` line is always better and is only
the number to bake.

## Model form and A750 results

The cost model (`fwd_estimate_cost` in `select.cpp`, called from Python
through `sdpa_model.py`) is, per subgroup and per key iteration: a systolic k-block
chain (each k-block issues `(unroll_m/8) x (unroll_n/simd)` dpas and takes at
least the pipe latency), K/V loads as one message per k-block of `unroll_m`
rows (latency or bandwidth share), softmax, the S round trip through SLM and
two barriers whose cost grows with the subgroups taking part. Work-groups
are scheduled in waves with a contention term;
memory time is bytes over bandwidth with the achieved bandwidth bounded by
bytes in flight. Compute and memory combine with a smooth max. The FMA path
keeps a flop-based term that no sweep has covered yet.

A750 (xe_hpg), 116 problems, 2677 timed configs: the generic sweep, the
SDPA KPI shapes (`problems_kpi.txt`), the coverage shapes derived from the
benchdnn harness (`problems_ci.txt`: both K layouts, grouped-query
attention, head sizes 16 to 512), the compute-bound shapes
(`problems_compute.txt`: long sequences and large head sizes at 500 to
8000 flops per byte) and full landscapes of ten problems. The data are
the CSVs the workflow above writes (`--out` of the top-k sweeps and of the
`--all` landscape sweeps); they are not kept in the tree. Rerunning the
sweeps and `fit_fwd_model.py --fit --folds 4 --restarts 1` reproduces the
table:

| selector | regret geomean | within 5% | within 15% | worst |
|---|---|---|---|---|
| legacy table | 1.259 | 25% | 46% | 2.29 |
| model, cross-validated | 1.082 | 52% | 76% | 1.49 |
| model, fitted on all (baked seeds) | 1.079 | 53% | 76% | 1.47 |
| built-in table hit | 1.000 by construction | | | |

Two lessons from widening the set. Dropping the barrier-per-subgroup and
causal-tail terms moves the cross-validated regret from 1.086 to 1.094 on
the 60-problem set, so both stay. The compute-bound shapes only pin down
the systolic terms when the throughput floor `mma_cycles_per_flop` is free:
held at its first-principles value the same data cross-validate at 1.105
with a worst case of 2.04, free they give the 1.082 above, with the floor
settling at half the nominal rate. The fitter therefore leaves it free by
default. Without compute-bound shapes a fit cannot tell the chain terms
from the floor at all, which is what the first PVC fit showed.

The remaining misses are short-key prefill (K of 77 and 197) and long
prefill at D 80 to 128 in the framework K layout, where the model prefers
a 128-column Q tile over the 64-column tile that wins; neighbouring tiles
of the same size are also confused at D=512. More measurements rather
than more terms are the next step there.

The model's top 12 plus 8 random candidates contained the best measured
config on every problem, which is how the 154 built-in table entries were
produced.

## xe2 results (BMG and LNL)

The table key carries the integrated flag (`:i0`/`:i1`), so BMG lines never
apply on LNL and each needed its own sweep; the coefficients are fitted on
both (`fit_fwd_model.py` names a problem per device, `@eu<N>`, when the
CSVs come from more than one; `--unfix mem_bw_integrated` on the fit and
on `bake_fit.py`). 171 problems, 3896 timed configs:

| selector | regret geomean |
|---|---|
| legacy table | 1.409 |
| model, cross-validated | 1.194 |
| model, fitted on all (baked seeds) | 1.177 |

`--compare` of the shipped table against the hand-tuned configs: BMG 0.758
over 77 problems, LNL 0.736 over 78, none slower by more than 5%. The model
misses the 1.10 target: it prefers 16-row KQ or 32-column VS tiles where
`32,32,32,16` and `32,32,16,32` win, on BMG by about 2x.

Rules found on xe2 (in `fwd_describe()`): decode over 2-byte aligned 16-bit
K with more than 64 keys is not dispatched on xe2/xe3 (mirrors `micro.cpp`);
a KQ subgroup count along M that is not a power of two (3, 5, 6) builds but
fails correctness from xe2 on (also on PTL and NVL); the PVC rule for
1024-element KQ tiles over 2-byte aligned K does not hold on BMG. The shipped
xe2 decode config `64,16,16,16,8,1,4,1` uses fewer VS than KQ subgroups,
which the tiler cannot express; it stays the legacy pick.

Other architectures use first-principles seeds until swept.
