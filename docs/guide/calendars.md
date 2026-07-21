# Calendars (CMIP6 & climate-model output)

Climate-model output (CMIP6/CMIP5) often uses non-standard CF calendars —
`noleap`/`365_day` (never a Feb 29), `360_day` (every month 30 days, so a valid
"Feb 30"), etc. These can't be represented as NumPy `datetime64`, so xarray loads
them as `cftime` objects.

**aggfly supports these out of the box and preserves the model calendar.** Both
temporal engines (`"dask"` and `"numba"`) group cftime time axes correctly, and the
output panel carries the model-calendar timestamps (e.g. a `2000-02-30` label on a
360-day calendar). Loading via `dataset_from_path` preserves the calendar
automatically; `date`/`month`/`year` groupings all work.

```python
ds = af.dataset_from_path("cmip6_tas.zarr", var="tas", timecoord="time", lon_is_360=True,
                          preprocess=lambda x: x - 273.15)
df = af.aggregate_dataset(dataset=ds, weights=weights,
        tavg=[('aggregate', {'calc': 'mean', 'groupby': 'month'})])   # months are model months
```

Use `aggfly info <path>` to check which calendar a dataset uses — it flags
non-standard `cftime` calendars explicitly.

## Two things to know

### `groupby='week'` is not available on non-standard calendars

cftime has no weekly offset (a "week" is undefined on a 360-day calendar), so aggfly
raises a clear error for both engines. Use `date`/`month`/`year` instead.

### Comparing model calendars to real dates is a modeling decision

A 360-day "year" is 360 model-days, not 365.25, and model days don't map 1:1 to real
calendar dates — so joining a 360-day/noleap panel to real-world (Gregorian) data
needs care.

If you want a standard-calendar axis (lossy — drops Feb 29 / spreads 360→365),
convert *before* aggregating with xarray's `convert_calendar`, choosing the
`align_on` policy yourself:

```python
ds.da = ds.da.convert_calendar("standard", align_on="date")   # your choice; not done silently
```

aggfly will never do this conversion silently on your behalf.
