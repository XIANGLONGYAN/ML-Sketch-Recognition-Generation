# ML Sketch Recognition & Generation (Unimodal Task)

This repository contains our implementation for the **Unimodal Task**, focusing on the **recognition** and **generation** of freehand sketches.

We developed:
- **Sketch Recognition**: a recognition model built upon **Axial Shift MLP (AS-MLP)**, trained and evaluated on the **QuickDraw-414k** dataset.
- **Sketch Generation**: a **multi-category controllable** sketch generation model capable of generating category-conditioned sketch images.

## Evaluation Summary

We analyze recognition and generation performance with a comprehensive suite of metrics:

### Recognition (Classification)
- **Top-1 / Top-5 Accuracy**
- **Confusion Matrix** (qualitative inspection and per-class behavior)

### Generation (Controllable Synthesis)
- **CLIP Score** (text-image alignment / semantic consistency)
- **LPIPS** (perceptual similarity / diversity)
- **FID** (distribution-level realism)

These metrics jointly reflect both recognition robustness and generation quality across multiple sketch categories.

---

## Code Execution Guide

This project is organized into two major components:

- `recognition/` — sketch classification (AS-MLP on QuickDraw-414k)
- `generation/` — multi-category controllable sketch generation

For detailed setup and command-line usage, **please refer to the component-level READMEs**:

- **Recognition instructions**: `recognition/README.md`
- **Generation instructions**: `generation/README.md`

Each submodule README contains:
- environment preparation (dependencies / datasets)
- training commands
- evaluation commands (including metric computation)
- output formats and checkpoint usage

---


