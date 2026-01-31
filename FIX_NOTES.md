# Index Error Fix for Sequences > 500 Nucleotides

## Problem Description

The original submission notebooks had an index error when processing RNA sequences longer than 500 nucleotides. The model's `max_length` parameter is set to 500, so it only generates coordinates for the first 500 positions. However, the submission creation code attempted to access coordinates for all positions in the sequence, causing an `IndexError` when sequences exceeded 500 nucleotides.

## Root Cause

### Issue 1: Prediction Generation
In the prediction generation code:
```python
seq_len = len(row['sequence'])  # Could be > 500
predictions.append({
    'coordinates': pred_coords[j, :seq_len, :, :]  # IndexError if seq_len > 500
})
```

The code tried to slice `pred_coords` using `seq_len`, but `pred_coords` only has 500 positions (shape is `(batch, 500, 5, 3)`).

### Issue 2: Submission Creation
In the submission creation code:
```python
for i, nucleotide in enumerate(sequence):  # i can be 0 to len(sequence)-1
    x, y, z = coords[i, struct_idx, :]  # IndexError when i >= 500
```

When iterating through all nucleotides in a sequence longer than 500, the code tried to access `coords[i]` where `i >= 500`, but `coords` only has 500 positions (shape is `(500, 5, 3)`).

## Solution

### Fix 1: Limit seq_len in Prediction Generation
```python
seq_len = min(len(row['sequence']), max_length)  # Cap at max_length (500)
predictions.append({
    'coordinates': pred_coords[j, :seq_len, :, :]  # Safe indexing
})
```

### Fix 2: Clamp Index in Submission Creation
```python
for i, nucleotide in enumerate(sequence):
    # Clamp index to valid range - reuse last coordinate for positions > 500
    coord_idx = min(i, coords.shape[0] - 1)
    x, y, z = coords[coord_idx, struct_idx, :]  # Safe indexing
```

## Behavior

For sequences with ≤ 500 nucleotides:
- Works exactly as before
- Each nucleotide gets its own predicted coordinate

For sequences with > 500 nucleotides:
- First 500 nucleotides: Use their predicted coordinates
- Nucleotides 501 onwards: Reuse the coordinate from position 500
- All nucleotides are included in the submission file
- No IndexError occurs

## Files Modified

1. `standalone_submission_notebook.ipynb`
   - Cell 15: Fixed prediction generation
   - Cell 17: Fixed submission creation

2. `kaggle_submission_notebook.ipynb`
   - Cell 11: Fixed prediction generation
   - Cell 13: Fixed submission creation

## Testing

The fix was tested with sequences of different lengths:
- 500 nucleotides: Works correctly (baseline)
- 600 nucleotides: Successfully generates 600 rows, positions 501-600 reuse coordinate 500
- 1000 nucleotides: Successfully generates 1000 rows, positions 501-1000 reuse coordinate 500

## Impact

This fix ensures that:
1. No runtime errors occur for long sequences
2. All nucleotides in the sequence are included in the submission
3. The submission format remains compliant with competition requirements
4. Positions beyond max_length get reasonable (last available) coordinates rather than causing a crash
