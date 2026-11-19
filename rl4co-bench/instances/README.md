# Fixed benchmark instances and source provenance

From the project root, verify the read-only RL4CO source and all test instances
before a measurement session:

```bash
python -B rl4co-bench/instances/fixed_inputs.py verify
```

The verifier requires Python with NumPy and Git. It checks file content against
the outer project's Git index and reconstructs the vendor tree. The tree must
equal `194dc6ae564df2c836c8f3da1ce2496ed983c477` from project commit
`1bfbb3415e997ee042e46ff286fa33adaa04bd17`. The Git tree is a pin for the
project's vendored checkout. It does not assert equivalence to an upstream
commit or an editable installation elsewhere.

The original local checkout has a `configs/env/` overlay omitted from this
tree by the vendored ignore rules. If present, the overlay must match
`local-hydra-overlay.json` in full. A clean checkout can omit it, as the
profiling driver instantiates environments through the Python API. Platform
caches and non-code assets outside the package/configuration directories are
outside this pin.

## Test population

Each NPZ contains exactly 1,024 instances. A result identifies its input by the
NPZ manifest ID and its zero-based row index. Use the same row order across
conditions. Batch size changes grouping and never generates fresh inputs.

| Manifest ID | Problem | Locations per row | Population |
| --- | --- | ---: | ---: |
| `tsp50` | TSP | 50 cities | 1,024 |
| `tsp100` | TSP | 100 cities | 1,024 |
| `tsp200` | TSP | 200 cities | 1,024 |
| `cvrp100` | CVRP | 100 customers and a depot | 1,024 |

For both problems, `locs` is float32 with shape `[batch, locations, 2]`. CVRP
also has `depot` (float32, `[batch, 2]`), `demand` (int64, `[batch, locations]`,
uniform integers from 1 to 9) and `capacity` (int64, `[batch, 1]`, 50). All
coordinates are sampled uniformly on a 24-bit grid in the unit square. The
CVRP convention matches the demand support and capacity of the vendored
RL4CO data generator for this size.

`fixed/manifest.json` records seeds and SHA-256 digests of the canonical
archives. Copy the NPZ directory to an unsynced local disk for timed runs and
verify that copy before use:

```bash
python -B rl4co-bench/instances/fixed_inputs.py verify --directory /path/to/fixed
```

For independent reproduction, the PCG64 generator writes deterministic NPZs
to a fresh directory and refuses to overwrite existing outputs:

```bash
python -B rl4co-bench/instances/fixed_inputs.py generate --directory /path/to/recreated
python -B rl4co-bench/instances/fixed_inputs.py verify --directory /path/to/recreated
```

## Python contract

Add `rl4co-bench/instances` to the import path. Call
`fixed_inputs.load_instance_set("tsp100")` to get a verified `(arrays, manifest_entry)`
pair. Pass `directory` to use the unsynced copy. For classical references use
the raw arrays. For RL4CO, slice every array to the same rows and pass the
result to `fixed_inputs.as_rl4co_tensordict()` before `env.reset()`. The adapter
normalises CVRP demand by capacity for the default unit-capacity RL4CO
environment. The archive retains the integer quantities so both solvers use
the same underlying instance.
