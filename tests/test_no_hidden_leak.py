"""Tests to verify inference logic does not leak training information.

These tests use AST static analysis and runtime inspection to ensure that
the predict function and its call chain in denoise_baseline.py do not:
- Reference ground-truth coordinate variables (clean, gt, ground_truth, target)
- Use sigma (noise level) for adaptive behavior during inference
- Compute or use chamfer_distance or other evaluation metrics to guide denoising

All tests run locally without GPU.
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DENOISE_BASELINE = PROJECT_ROOT / "denoise_baseline.py"

# Variable names that indicate ground-truth leakage in predict context
GT_LEAK_NAMES = {"clean", "gt", "ground_truth", "target"}

# Parameter names that should NOT appear in predict function signatures
FORBIDDEN_PREDICT_PARAMS = {"ground_truth", "gt", "clean", "target", "sigma", "cd_score"}

# Function/method calls that indicate metric-guided denoising
FORBIDDEN_METRIC_CALLS = {"chamfer_distance", "chamfer_l2", "earth_mover_distance", "hausdorff_distance"}

# Names used as sigma / noise-level variable in predict that indicate leakage
SIGMA_NAMES = {"sigma", "noise_level", "noise_sigma", "gt_sigma"}


# ---------------------------------------------------------------------------
# Helper: Parse AST
# ---------------------------------------------------------------------------


def _parse_baseline_ast() -> ast.Module:
    """Parse denoise_baseline.py into an AST."""
    source = DENOISE_BASELINE.read_text()
    return ast.parse(source, filename=str(DENOISE_BASELINE))


def _find_function_defs(tree: ast.Module, names: set[str]) -> list[ast.FunctionDef]:
    """Find all function/method definitions matching given names."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in names:
                found.append(node)
    return found


def _get_all_names_in_node(node: ast.AST) -> set[str]:
    """Collect all Name identifiers used within an AST node."""
    names = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
    return names


def _get_all_call_names(node: ast.AST) -> set[str]:
    """Collect all function/method call names within an AST node."""
    calls = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                calls.add(child.func.id)
            elif isinstance(child.func, ast.Attribute):
                calls.add(child.func.attr)
    return calls


def _get_function_params(node: ast.FunctionDef) -> set[str]:
    """Get all parameter names of a function definition."""
    params = set()
    for arg in node.args.args:
        params.add(arg.arg)
    for arg in node.args.posonlyargs:
        params.add(arg.arg)
    for arg in node.args.kwonlyargs:
        params.add(arg.arg)
    if node.args.vararg:
        params.add(node.args.vararg.arg)
    if node.args.kwarg:
        params.add(node.args.kwarg.arg)
    return params


# ---------------------------------------------------------------------------
# AST Static Analysis Tests
# ---------------------------------------------------------------------------


@pytest.mark.leak_check
class TestNoGTLeakStatic:
    """Static analysis: predict functions must not reference GT variables."""

    def test_baseline_file_exists(self):
        """denoise_baseline.py must exist for analysis."""
        assert DENOISE_BASELINE.exists(), (
            f"denoise_baseline.py not found at {DENOISE_BASELINE}"
        )

    def test_predict_no_gt_variable_usage(self):
        """predict() function must not use clean/gt/ground_truth/target as input variables."""
        tree = _parse_baseline_ast()
        predict_fns = _find_function_defs(tree, {"predict", "predict_points_in_chunks"})
        assert predict_fns, "Could not find predict functions in denoise_baseline.py"

        for fn in predict_fns:
            names_used = _get_all_names_in_node(fn)
            # 'clean' as a standalone Name (not attribute access) in predict is a leak
            leaked = GT_LEAK_NAMES & names_used
            # Filter: if 'clean' only appears in string literals or comments, that's fine.
            # Check actual Name nodes specifically
            actual_name_refs = set()
            for child in ast.walk(fn):
                if isinstance(child, ast.Name) and child.id in GT_LEAK_NAMES:
                    # Verify it's used as a variable, not just part of a string
                    actual_name_refs.add(child.id)
            assert not actual_name_refs, (
                f"Function '{fn.name}' references GT variables as identifiers: "
                f"{actual_name_refs}. Prediction must not access ground truth data."
            )

    def test_predict_no_sigma_parameter(self):
        """predict() and predict_points_in_chunks() must not accept sigma/noise params."""
        tree = _parse_baseline_ast()
        predict_fns = _find_function_defs(tree, {"predict", "predict_points_in_chunks"})
        assert predict_fns, "Could not find predict functions"

        for fn in predict_fns:
            params = _get_function_params(fn)
            leaked_params = FORBIDDEN_PREDICT_PARAMS & params
            assert not leaked_params, (
                f"Function '{fn.name}' has forbidden parameters: {leaked_params}. "
                f"Predict must not receive GT/sigma/cd_score inputs."
            )

    def test_predict_no_sigma_usage(self):
        """predict() must not use sigma-related variables for adaptive denoising."""
        tree = _parse_baseline_ast()
        predict_fns = _find_function_defs(tree, {"predict", "predict_points_in_chunks"})

        for fn in predict_fns:
            actual_sigma_refs = set()
            for child in ast.walk(fn):
                if isinstance(child, ast.Name) and child.id in SIGMA_NAMES:
                    actual_sigma_refs.add(child.id)
            assert not actual_sigma_refs, (
                f"Function '{fn.name}' uses sigma/noise-level variables: "
                f"{actual_sigma_refs}. Predict must not adapt based on known noise level."
            )

    def test_predict_no_metric_calls(self):
        """predict() must not call evaluation metrics (chamfer_distance, etc.)."""
        tree = _parse_baseline_ast()
        predict_fns = _find_function_defs(tree, {"predict", "predict_points_in_chunks"})

        for fn in predict_fns:
            calls = _get_all_call_names(fn)
            leaked_calls = FORBIDDEN_METRIC_CALLS & calls
            assert not leaked_calls, (
                f"Function '{fn.name}' calls evaluation metrics: {leaked_calls}. "
                f"Predict must not use metrics to guide denoising."
            )

    def test_execute_no_gt_in_forward(self):
        """ResidualDenoiser.execute() must not reference GT variables."""
        tree = _parse_baseline_ast()
        # Find the execute method inside ResidualDenoiser class
        execute_fns = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "ResidualDenoiser":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "execute":
                        execute_fns.append(item)

        assert execute_fns, "Could not find ResidualDenoiser.execute()"

        for fn in execute_fns:
            params = _get_function_params(fn)
            # 'noisy' and 'return_offset' are expected; GT vars are not
            leaked_params = FORBIDDEN_PREDICT_PARAMS & params
            assert not leaked_params, (
                f"ResidualDenoiser.execute() has forbidden parameters: {leaked_params}"
            )

            # Check for GT variable usage in the body
            actual_refs = set()
            for child in ast.walk(fn):
                if isinstance(child, ast.Name) and child.id in GT_LEAK_NAMES:
                    actual_refs.add(child.id)
            assert not actual_refs, (
                f"ResidualDenoiser.execute() references GT variables: {actual_refs}"
            )


# ---------------------------------------------------------------------------
# Runtime Inspection Tests
# ---------------------------------------------------------------------------


@pytest.mark.leak_check
class TestNoGTLeakRuntime:
    """Runtime inspection of predict function signatures."""

    @pytest.fixture(autouse=True)
    def _ensure_importable(self):
        """Ensure denoise_baseline module is importable (may fail without Jittor)."""
        # Add project root to path
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

    def test_predict_function_signature(self):
        """predict() signature must not include GT/sigma/cd_score parameters."""
        tree = _parse_baseline_ast()
        predict_fns = _find_function_defs(tree, {"predict"})
        assert predict_fns, "Could not find predict function"

        for fn in predict_fns:
            params = _get_function_params(fn)
            forbidden_found = FORBIDDEN_PREDICT_PARAMS & params
            assert not forbidden_found, (
                f"predict() signature contains forbidden params: {forbidden_found}"
            )

    def test_predict_points_in_chunks_signature(self):
        """predict_points_in_chunks() must not accept GT parameters."""
        tree = _parse_baseline_ast()
        fns = _find_function_defs(tree, {"predict_points_in_chunks"})
        assert fns, "Could not find predict_points_in_chunks function"

        for fn in fns:
            params = _get_function_params(fn)
            forbidden_found = FORBIDDEN_PREDICT_PARAMS & params
            assert not forbidden_found, (
                f"predict_points_in_chunks() has forbidden params: {forbidden_found}"
            )


# ---------------------------------------------------------------------------
# Data Flow Analysis Tests
# ---------------------------------------------------------------------------


@pytest.mark.leak_check
class TestNoLabeledDataLoad:
    """Verify predict call chain does not load labeled (clean/GT) data."""

    def test_predict_does_not_load_clean_data(self):
        """predict() must not call ObjDenoiseDataset or load clean .obj files."""
        tree = _parse_baseline_ast()
        predict_fns = _find_function_defs(tree, {"predict"})

        for fn in predict_fns:
            calls = _get_all_call_names(fn)
            # ObjDenoiseDataset is training-only; predict should load noisy.npy directly
            assert "ObjDenoiseDataset" not in calls, (
                "predict() instantiates ObjDenoiseDataset which loads clean data"
            )
            assert "load_obj_vertices" not in calls, (
                "predict() calls load_obj_vertices which loads clean mesh data"
            )

    def test_predict_does_not_call_make_batch(self):
        """predict() must not call make_batch (which creates noisy/clean pairs)."""
        tree = _parse_baseline_ast()
        predict_fns = _find_function_defs(tree, {"predict"})

        for fn in predict_fns:
            calls = _get_all_call_names(fn)
            assert "make_batch" not in calls, (
                "predict() calls make_batch which creates training pairs"
            )
            assert "make_batch_prefetched" not in calls, (
                "predict() calls make_batch_prefetched which creates training pairs"
            )

    def test_predict_only_loads_noisy_npy(self):
        """predict() should load noisy.npy files (not clean.npy or gt.npy)."""
        tree = _parse_baseline_ast()
        predict_fns = _find_function_defs(tree, {"predict"})

        for fn in predict_fns:
            # Look for string constants that might be file paths
            for child in ast.walk(fn):
                if isinstance(child, ast.Constant) and isinstance(child.value, str):
                    val = child.value.lower()
                    # Ensure we're loading noisy data, not clean/gt
                    if "clean" in val and ".npy" in val:
                        pytest.fail(
                            f"predict() references 'clean' npy file: '{child.value}'"
                        )
                    if "gt" in val and ".npy" in val:
                        pytest.fail(
                            f"predict() references 'gt' npy file: '{child.value}'"
                        )

    def test_execute_does_not_compute_chamfer_in_forward(self):
        """ResidualDenoiser.execute() must not call chamfer_l2 or similar metrics."""
        tree = _parse_baseline_ast()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "ResidualDenoiser":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "execute":
                        calls = _get_all_call_names(item)
                        leaked = FORBIDDEN_METRIC_CALLS & calls
                        assert not leaked, (
                            f"ResidualDenoiser.execute() calls metrics: {leaked}. "
                            f"Model forward pass must not use evaluation metrics."
                        )

    def test_no_gt_file_glob_in_predict(self):
        """predict() must not glob for clean/gt files."""
        tree = _parse_baseline_ast()
        predict_fns = _find_function_defs(tree, {"predict"})

        for fn in predict_fns:
            for child in ast.walk(fn):
                if isinstance(child, ast.Constant) and isinstance(child.value, str):
                    val = child.value
                    # Check for glob patterns that would match GT data
                    if "clean.npy" in val or "gt.npy" in val or "ground_truth" in val:
                        pytest.fail(
                            f"predict() contains GT file pattern: '{val}'"
                        )
