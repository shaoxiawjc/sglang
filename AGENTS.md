# Repository Guidelines

## Project Structure & Module Organization
Core Python code lives in `python/sglang/`. Tests are under `test/`, with runtime suites in `test/srt/`, unit tests in `test/unit/`, and registry-based CI coverage in `test/registered/`. Benchmarks and evaluation scripts live in `benchmark/`. Documentation sources are in `docs/`, examples in `examples/`, container assets in `docker/`, and branding files in `assets/`. Two subprojects have their own workflows: `sgl-kernel/` for native kernels and `sgl-model-gateway/` for the Rust gateway.

## Build, Test, and Development Commands
Install the main package from source with `cd python && pip install -e ".[dev]"`. Run repository checks with `pre-commit run --all-files`; this applies import sorting, lint fixes, notebook cleanup, spelling checks, and formatting hooks. Run focused Python tests with `python3 test/srt/test_srt_endpoint.py` or suite-based coverage with `python3 test/run_suite.py --hw cuda --suite stage-b-test-small-1-gpu`. Build docs with `make -C docs html`. Subprojects use their own entrypoints: `make -C sgl-kernel build` and `make -C sgl-model-gateway test`.

## Coding Style & Naming Conventions
Use 4-space indentation for Python and follow existing module patterns. Prefer descriptive snake_case for Python files, functions, and tests; keep backend-specific additions isolated in new files such as `allocator_ascend.py` instead of widening shared conditionals. Run `pre-commit` before pushing. Formatting is enforced with `isort`, `black-jupyter`, `ruff` for selected Python checks, `clang-format` for C/CUDA, and `codespell`.

## Testing Guidelines
This repository primarily uses `unittest`, executed as direct scripts from `test/`; some components also use `pytest` where already configured. Name files `test_*.py` and keep them fast, focused, and reusable across suites. If you add CI coverage, register or list the test in the relevant `run_suite.py` and keep test entries alphabetized. Reuse existing server launches where possible to avoid slow GPU test startup.

## Commit & Pull Request Guidelines
Recent history favors concise, scoped subjects such as `[NPU] Bump SGL-Kernel-NPU version...`, `[Feature] Integrate ...`, or `feat: fix ...`. Use a clear prefix for subsystem or change type, then a brief imperative summary. Open PRs from a branch, not `main`, and include the problem, the fix, test evidence, and any hardware or accuracy impact. Add screenshots only for docs or UI-facing changes. For CI, maintainers may apply the `run-ci` label or slash commands described in `docs/developer_guide/contribution_guide.md`.
