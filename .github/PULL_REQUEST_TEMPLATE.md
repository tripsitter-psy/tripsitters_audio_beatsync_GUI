### Summary
Please describe the change and why it is needed.

### Checklist
- [ ] I have added/updated tests
- [ ] I have updated documentation

### GPU / CUDA / ONNX Note
If your PR touches GPU code (CUDA, ONNX runtime, device providers, .cu files, or environment flags like `BEATSYNC_ONNX_USE_CUDA`):
- Add the label **`test-cuda`** to the PR so the gated CUDA integration workflow runs on a GPU-enabled runner.
- In the **Summary** section above, include GPU/driver testing details:
  - GPU model (e.g., RTX 4090, GTX 1080)
  - CUDA toolkit version (e.g., 12.4)
  - Driver version (e.g., 560.94)
  - Any special setup steps or environment variables used

Thanks!
<details>
<summary>Optional: PR Metadata (Type, Related Issues, Breaking Changes)</summary>

### Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

### Related Issues
Link any related issues (optional). For example:
- Closes #123

### Breaking Changes
If this PR introduces breaking changes, briefly document migration steps or upgrade guidance here. If not applicable, leave blank.

</details>