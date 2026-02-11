# Best Practices

## Tips and General Guidelines

- **Keep everything on the same genome build.** Ensure your reference panel and test (query) files are in the **same build**. If they are not, perform a **liftover** before running GNomix.
- **Avoid missing data.** Make sure there is **no missing data** in your inputs. If missing data is present, perform **imputation**.
- **Watch overlap warnings.** If GNomix reports an overlap **smaller than 95%** of the variants, we highly encourage you to:
  - **Appropriately select** the variants used to train the model, and
  - Perform **adequate imputation** on the query data.

## Hyperparameter Tuning

All hyperparameters should be edited **through the config file**.

- **Control prediction smoothness via `gens`.**  
  Recommended default:
  - `gens: [0, 2, 4, 6, 8, 12, 16, 24]`

  For **highly smooth** predictions:
  - `gens: [0, 2, 4, 6]`

  To detect **very small segments**, train with:
  - `gens: [0, 2, 4, 6, 8, 12, 16, 24, 48, 64, 128]`

- **Trade training time for a performance boost with `r_admixed`.**  
  For extra performance (longer training time):
  - `r_admixed: 2` or `r_admixed: 3`

- **Prefer fast inference.**
  - We always recommend using `inference: fast`.

- **Try varying `context_ratio`.**
  - Changing `context_ratio: 0.5` to `context_ratio: 0` or to `context_ratio: 1` can sometimes lead to improvements.
  - `context_ratio: 1` can especially help with **array data**.

- **Adjust `window_size_cM` for segment resolution vs smoothness.**
  - Changing `window_size_cM: 0.2` to `window_size_cM: 0.1` can help detect **smaller segments**.
  - Changing `window_size_cM: 0.2` to `window_size_cM: 0.4` can help obtain **smoother predictions**.
  - `window_size_cM: 0.4` can help with **array data**.
