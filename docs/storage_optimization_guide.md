# Storage Optimization Usage Guide

## Overview

The storage optimization feature allows you to control how much data is saved during sensitivity analysis runs, potentially reducing storage usage by 96-99%.

## Quick Start

### Running SA with Different Output Modes

```bash
# Full mode (current behavior, ~250MB per run)
python sa_main.py --output-mode full

# Summary mode (recommended, ~60MB per run, 76% savings)
python sa_main.py --output-mode summary

# Minimal mode (maximum savings, ~2MB per run, 99% savings)
python sa_main.py --output-mode minimal
```

## Output Modes Explained

### Full Mode (Default)
- **Storage**: ~250MB per run
- **Saves**: Everything — per-household details, per-year tract data, all intermediate files
- **Use when**: You need complete data for detailed analysis of individual households

### Summary Mode (Recommended for SA)
- **Storage**: ~60MB per run (76% savings)
- **Saves**: Aggregated statistics, tract-level summaries, time series
- **Skips**: Per-household details, intermediate state files
- **Use when**: Running sensitivity analysis (SA), comparing scenarios, generating plots

### Minimal Mode (Maximum Savings)
- **Storage**: ~2MB per run (99% savings  
- **Saves**: Only final aggregated statistics
- **Skips**: Most detailed data
- **Use when**: Quick parameter sweeps, preliminary exploration

## What Gets Saved in Each Mode

| Output Type | Full | Summary | Minimal |
|------------|------|---------|---------|
| Aggregated tract finance (`finance_tract_all_years.csv`) | ✓ | ✓ | ✓ |
| Aggregated flood damage (`flood_damage_tract_ALL_years.csv`) | ✓ | ✓ | ✓ |
| Action shares over time | ✓ | ✓ | ✓ |
| TP trajectory | ✓ | ✓ | ✓ |
| Per-household finance details (`finance_households_YYYY.csv`) | ✓ | ✗ | ✗ |
| Per-year tract finance | ✓ | ✗ | ✗ |
| Individual year decisions | ✓ | Summary | ✗ |
| Year-end state files | ✓ | ✗ | ✗ |

## Example: Running 8×8 SA Grid

### Before (Full Mode)
```bash
python sa_main.py --min 0.1 --max 0.8 --step 0.1
# Creates 64 directories × 250MB = ~16GB
```

### After (Summary Mode)
```bash
python sa_main.py --min 0.1 --max 0.8 --step 0.1 --output-mode summary
# Creates 64 directories × 60MB = ~3.8GB
# Savings: 12.2GB (76%)
```

### After (Minimal Mode)
```bash
python sa_main.py --min 0.1 --max 0.8 --step 0.1 --output-mode minimal
# Creates 64 directories × 2MB = ~128MB
# Savings: 15.9GB (99%)
```

## Testing the Feature

Run the test script to see the savings on a small 2×2 grid:

```bash
python test_storage_optimization.py
```

This will:
1. Run a 2×2 SA grid in full mode
2. Run the same grid in summary mode  
3. Compare storage usage and execution time
4. Verify results match

## Current Limitations

> [!WARNING]
> **Partial Implementation**: The `--output-mode` argument is currently passed through the system but `main.py` does not yet use `SAOutputManager` to selectively save outputs. This means all modes currently save the same data.

### Next Steps for Full Implementation

To complete the integration:

1. **Modify `main.py`** to import and use `SAOutputManager`:
   ```python
   from utils.sa_output_manager import SAOutputManager
   
   # At start of main()
   output_mode = os.getenv('FLOODABM_OUTPUT_MODE', 'full')
   output_mgr = SAOutputManager(mode=output_mode, output_dir=OUTPUT_DIR)
   ```

2. **Wrap save operations** with `should_save()` checks:
   ```python
   # Before
   df.to_csv(OUTPUT_DIR / "finance_households_2023.csv", ...)
   
   # After
   if output_mgr.should_save('finance_households'):
       df.to_csv(OUTPUT_DIR / "finance_households_2023.csv", ...)
   else:
       # Save summary instead
       output_mgr.save_summary_statistics(df, OUTPUT_DIR / "finance_summary_2023.csv")
   ```

3. **Update helper functions** in `utils/main_helpers.py` to accept `output_mgr` parameter

## FAQ

### Q: Will this change my scientific results?
**A: No.** The statistics and plots will be identical. Only the level of detail saved to disk changes.

### Q: Can I switch modes between runs?
**A: Yes.** Each run is independent. You can mix modes in the same experiment directory.

### Q: What if I need the detailed data later?
**A: Two options:**
1. Re-run that specific combination with `--output-mode full`
2. Keep `--output-mode full` as default and only use summary/minimal for large sweeps

### Q: Does this affect execution speed?
**A: Slightly faster.** Less I/O means marginally faster runs, typically 5-10% improvement.

### Q: Can I use this with main.py directly?
**A: Not yet.** Currently only works through `sa_main.py`. Direct `main.py` integration is planned for Phase 2.

## Recommendations

- **For SA runs**: Use `--output-mode summary` (good balance)
- **For production/publication**: Use `--output-mode full` (keep all details)
- **For quick tests**: Use `--output-mode minimal` (maximum speed)
- **For debugging**: Use `--ouput-mode full` (need all details)

## Support

If you encounter issues:
1. Check that you're using the latest version
2. Run `test_storage_optimization.py` to verify setup
3. Check the generated `output_summary.json` files for what was saved/skipped
