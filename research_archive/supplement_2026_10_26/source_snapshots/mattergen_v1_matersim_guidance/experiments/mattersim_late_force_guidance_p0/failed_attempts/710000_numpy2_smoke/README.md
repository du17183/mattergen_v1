# Retained technical smoke failure: NumPy 2 compatibility

The first diagnostic seed-710000 smoke stopped before producing a trajectory
because a frozen MatterGen dependency accesses `np.math`, which NumPy 2 no
longer exposes. The runner now installs the narrow compatibility alias
`np.math = math` before MatterGen imports. The same seed then completed under
the otherwise unchanged frozen sampling protocol. No metric from the failed
attempt is reported.
