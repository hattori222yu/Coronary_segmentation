# -*- coding: utf-8 -*-
"""
postprocess.py
==============
Post-processing module for coronary artery segmentation.

Retains the two largest connected components (left and right coronary arteries)
from a segmentation mask, and resolves any label-mixing within each component
by assigning the majority label to all voxels in that component.

Author : Hattori (2026)
License: MIT
"""

from __future__ import annotations

import os
import glob
import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import nibabel as nib
from scipy import ndimage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core processing function (single file)
# ---------------------------------------------------------------------------

def process_single_file(
    filepath: str | Path,
    output_dir: str | Path,
    top_n: int = 2,
) -> dict:
    """
    Process one segmentation NIfTI file.

    Parameters
    ----------
    filepath : str or Path
        Path to input ``*.nii.gz`` file.
    output_dir : str or Path
        Directory where the processed file will be saved.
    top_n : int, optional
        Number of largest connected components to retain (default: 2).

    Returns
    -------
    dict
        Processing result with keys:

        * ``fname``      – filename
        * ``status``     – ``"ok"``, ``"skip"`` (empty), or ``"error"``
        * ``n_before``   – connected-component count before filtering
        * ``n_after``    – connected-component count after filtering
        * ``log_parts``  – list of per-component detail strings
        * ``message``    – human-readable summary string
    """
    filepath = Path(filepath)
    output_dir = Path(output_dir)
    fname = filepath.name

    result = dict(fname=fname, status="ok", n_before=0, n_after=0,
                  log_parts=[], message="")

    try:
        img = nib.load(filepath)
        data = img.get_fdata().astype(np.uint8)

        binary = (data > 0).astype(np.uint8)
        labeled_array, num_features = ndimage.label(binary)
        result["n_before"] = num_features

        # ------------------------------------------------------------------
        # Empty mask → save as-is
        # ------------------------------------------------------------------
        if num_features == 0:
            result["status"] = "skip"
            result["message"] = "Segmentation is empty; saved unchanged."
            nib.save(img, output_dir / fname)
            return result

        # ------------------------------------------------------------------
        # Select top-N largest components
        # ------------------------------------------------------------------
        component_sizes = np.array(
            ndimage.sum(binary, labeled_array, range(1, num_features + 1))
        )
        actual_top = min(top_n, num_features)
        top_indices = np.argsort(component_sizes)[::-1][:actual_top]
        top_labels = top_indices + 1          # ndimage labels are 1-indexed
        result["n_after"] = actual_top

        # ------------------------------------------------------------------
        # Resolve label mixing: assign majority label inside each component
        # ------------------------------------------------------------------
        new_data = np.zeros_like(data, dtype=np.uint8)
        log_parts: list[str] = []

        for rank, comp_label in enumerate(top_labels, start=1):
            comp_mask = labeled_array == comp_label
            voxels_in_comp = data[comp_mask]
            unique, counts = np.unique(voxels_in_comp, return_counts=True)

            # Exclude background (label 0)
            valid = unique > 0
            unique, counts = unique[valid], counts[valid]
            if len(unique) == 0:
                continue

            majority_label = int(unique[np.argmax(counts)])
            total_vox = int(counts.sum())
            majority_vox = int(counts.max())
            minority_vox = total_vox - majority_vox
            minority_labels = unique[unique != majority_label]

            new_data[comp_mask] = majority_label

            if minority_vox > 0:
                minority_str = "+".join([f"label{lbl}" for lbl in minority_labels])
                log_parts.append(
                    f"  Component {rank} (size={int(component_sizes[top_indices[rank-1]])}): "
                    f"majority=label{majority_label} ({majority_vox} vox), "
                    f"corrected={minority_str} ({minority_vox} vox → label{majority_label})"
                )
            else:
                log_parts.append(
                    f"  Component {rank} (size={int(component_sizes[top_indices[rank-1]])}): "
                    f"label{majority_label} only (no mixing)"
                )

        result["log_parts"] = log_parts

        # ------------------------------------------------------------------
        # Save
        # ------------------------------------------------------------------
        new_img = nib.Nifti1Image(new_data, img.affine, img.header)
        nib.save(new_img, output_dir / fname)

        removed = num_features - actual_top
        result["message"] = (
            f"Components: {num_features} → {actual_top} retained, {removed} removed."
        )

    except Exception as exc:
        result["status"] = "error"
        result["message"] = str(exc)
        logger.exception("Error processing %s", fname)

    return result


# ---------------------------------------------------------------------------
# Batch processing function
# ---------------------------------------------------------------------------

def run_postprocess(
    input_dir: str | Path,
    output_dir: str | Path,
    top_n: int = 2,
    log_callback: Optional[Callable[[str], None]] = None,
) -> list[dict]:
    """
    Apply :func:`process_single_file` to all ``case_*.nii.gz`` files in
    *input_dir* and write results to *output_dir*.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing raw ``*.nii.gz`` predictions.
    output_dir : str or Path
        Directory for post-processed outputs (created if absent).
    top_n : int, optional
        Number of largest components to retain (default: 2).
    log_callback : callable, optional
        Function that accepts a single string argument; called for each log
        line so the caller can pipe messages to a GUI or logger.

    Returns
    -------
    list[dict]
        One result dict per file (see :func:`process_single_file`).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("case_*.nii.gz"))

    def emit(msg: str) -> None:
        logger.info(msg)
        if log_callback:
            log_callback(msg)

    emit(f"Post-processing: {len(files)} file(s) found in {input_dir}")

    results = []
    for filepath in files:
        res = process_single_file(filepath, output_dir, top_n=top_n)
        results.append(res)

        tag = {"ok": "[OK]", "skip": "[SKIP]", "error": "[ERROR]"}.get(
            res["status"], "[?]"
        )
        emit(f"{tag} {res['fname']} | {res['message']}")
        for part in res["log_parts"]:
            emit(part)

    n_ok = sum(r["status"] == "ok" for r in results)
    n_skip = sum(r["status"] == "skip" for r in results)
    n_err = sum(r["status"] == "error" for r in results)
    emit(
        f"\nDone — OK: {n_ok}, SKIP: {n_skip}, ERROR: {n_err}  "
        f"→ Output: {output_dir}"
    )
    return results
