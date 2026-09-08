# Baseline v1

Baseline v1 freezes the first complete experimental system before calculation-audit corrections or
new forward-speed models. The immutable Git tag is `baseline-v1`.

It preserves:

- the 10,411-parameter PyTorch dilated-CNN checkpoint;
- the training, IEKF, alignment, blackout-selection, GNSS and HMM configurations;
- the locked Vta08 and S3c 20-second blackout intervals;
- Vta08 and S3c cumulative metrics and trajectories;
- the final presentation summary and plots.

The checkpoint remains at `results/models/adapter_cnn_v1.pt`. Its SHA-256 is recorded in
`manifest.json`. To restore the entire baseline, including the checkpoint and results, use the Git
tag rather than rerunning training:

```bash
git switch --detach baseline-v1
```

Future experiments must report changes against the frozen Vta08 pre-map drift of 20.965754%,
Vta08 post-map drift of 11.145913%, S3c pre-map drift of 106.201443%, and S3c post-map drift of
93.524308% on the exact intervals in the manifest.

Raw IO-VNBD datasets are deliberately not part of the baseline commit.
