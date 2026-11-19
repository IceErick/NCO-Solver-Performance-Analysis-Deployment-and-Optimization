# Measurement environment and calibration

Use [the RTX 4090 runbook](../RUN_NOVEMBER.md) to create the environment
and run a session. The current protocol records one NVIDIA workstation. The
older `capture_env.py`, macOS JSON, PARITY and CALIBRATION documents are
historical records from the cross-backend design.

`session.py` checks the vendored source, collects NVIDIA telemetry, and times
five blocks of ten resident 4096 × 4096 FP32 matrix multiplications. Five
warmup calls precede the timing. Each block synchronises before its elapsed
time is read. Inputs have a fixed seed and TF32 is disabled.

The first session explicitly enrolls a reference. Pass that JSON file with
`--reference` at the start of each subsequent measurement session. A
comparison checks the hardware UUID, driver, OS, source pin and core software
versions. It flags more than 10% change in median matmul time, more than 8 °C
change in comparable temperature snapshots, or more than 10% change in
comparable post-workload clocks. The thresholds are screening rules. Snapshot
clocks are sampled after the workload and do not replace sustained telemetry.

A drift or incomparable result preserves its JSON and stops the session.
Review the reason, restore a comparable workload and machine state, then
start a fresh session. Record a new reference explicitly if a driver or
software change was intentional. The CUDA runner keeps a copy of its session
record alongside raw traces. A CPU smoke calibration is tagged separately.
