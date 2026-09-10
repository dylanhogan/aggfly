# Aggregation

`af.aggregate_dataset(dataset, weights, **named_specs)` runs temporal aggregation
first, then spatial aggregation, and returns a pandas DataFrame merged back onto
the region ids.

## The spec DSL

Each keyword argument to `aggregate_dataset` **names an output variable** and maps
to a **list of steps applied in order**. Each step is a `(step_type, params)` tuple.

```python
output_df = af.aggregate_dataset(
    dataset=dataset,
    weights=weights,
    tavg=[
        ('aggregate', {'calc': 'mean', 'groupby': 'date'}),
        ('transform', {'transform': 'power', 'exp': np.arange(1, 3)}),
        ('aggregate', {'calc': 'sum', 'groupby': 'year'})
    ],
    bins=[
        ('aggregate', {'calc': 'mean', 'groupby': 'date'}),
        ('aggregate', {'calc': 'bins', 'groupby': 'year',
                       'ddargs': [[25, 99, 0], [30, 99, 0]]})
    ],
    growing_dday=[
        ('aggregate', {'calc': 'dd', 'groupby': 'date', 'ddargs': [10, 30, 0]}),
        ('aggregate', {'calc': 'sum', 'groupby': 'year'}),
    ],
    heating_dday=[
        ('aggregate', {'calc': 'dd', 'groupby': 'date', 'ddargs': [-99, 20, 1]}),
        ('aggregate', {'calc': 'sum', 'groupby': 'year'}),
    ]
)
```

Read `tavg` above as: *average hourly values to daily means, raise to powers 1 and
2, then sum each over the year.*

## Step type: `aggregate`

```python
('aggregate', {'calc': ..., 'groupby': ..., 'ddargs': ...})
```

| Key | Meaning |
|---|---|
| `calc` | The reduction to apply — see the table below. |
| `groupby` | The time frequency to reduce over: `date`, `month`, `year`, … |
| `ddargs` | Thresholds; required for `dd` and `bins`. |

### Available calcs

| `calc` | Description |
|---|---|
| `mean` | Average value within the period given by `groupby`. |
| `min` | Minimum value within the period. |
| `max` | Maximum value within the period. |
| `sum` | Sum over the period. |
| `dd` | **Degree days** — sums the degrees by which temperature is above (cooling) or below (heating) a base temperature. |
| `bins` | Divides data into bins by threshold, counting occurrences in each. |
| `sine_dd` | Degree days computed with a sinusoidal within-day interpolation. |

### `ddargs`

`ddargs` gives thresholds as `[low, high, inc]`:

- **`dd`** takes a single triple: `[10, 30, 0]` accumulates degree days between
  10 and 30. `[-99, 20, 1]` gives heating degree days below 20.
- **`bins`** takes a *list* of triples: `[[25, 99, 0], [30, 99, 0]]` produces one
  output per bin.

A single `dd`/`bins` step with multiple `ddargs` ("multi-dd") fans one variable out
into several outputs keyed by threshold.

> **Constraint:** you cannot combine multi-`ddargs` with multiple upstream datasets
> — e.g. multiple polynomial exponents *and* multiple bins in the same chain. The
> CLI's `aggfly validate` catches this statically.

## Step type: `transform`

```python
('transform', {'transform': 'power', 'exp': np.arange(1, 3)})
```

| `transform` | Meaning |
|---|---|
| `power` | Raise the variable to the given `exp` powers, producing one output per exponent (keys suffixed `_1`, `_2`, …). |
| `inter` | Interact with another dataset. |
| `spline` | Spline basis expansion. |

## Daily temperature splines

A spline transform expands each grid-cell temperature into several basis columns.
Apply it **after daily averaging and before annual summation and spatial weighting**:

```python
df = af.aggregate_dataset(
    dataset=dataset,  # Celsius; knots use the same units as the data
    weights=weights,
    temperature=[
        ("aggregate", {"calc": "mean", "groupby": "date"}),
        ("transform", {
            "transform": "spline",
            "degree": 3,
            "restricted": True,
            "basis": "truncated_power",
            "knots": [0, 7.5, 12.5, 20],
        }),
        ("aggregate", {"calc": "sum", "groupby": "year"}),
    ],
)
```

| Option | Meaning |
|---|---|
| `degree` | Integer 1–4: linear, quadratic, cubic, quartic. Default 1. |
| `restricted` | `True` imposes linear tails outside the first/last knot; `False` continues the outer polynomial pieces. Default `False`. |
| `basis` | `"truncated_power"` (default) or `"bspline"`. Changes the representation of the spline space. |
| `knots` | Fixed, finite, strictly increasing breakpoints. Default `[20]` preserves the old transform. Restricted splines require at least `max(2, degree)` knots. |

Keep the same knots and options across all regions and years. No knots are
estimated from data. The curve is continuous through derivative `degree - 1`
at every knot, including the boundary knots. Linear splines permit slope jumps;
quadratic splines preserve slope; cubic splines also preserve curvature; quartic
splines preserve the third derivative. For restricted splines, first and last
knots mark the start of the linear tails. In the unrestricted case all supplied
knots are ordinary breakpoints.

**Truncated-power basis.** Unrestricted degree `p` returns powers `T` through
`T**p`, followed by `(T-knot).clip(min=0)**p` for each knot. Restricted splines
return `T` and corrected truncated-power terms whose higher-order tail powers
cancel. With `K` knots there are `K+p` unrestricted columns or `K-p+2` restricted
columns; neither includes an intercept. Explicit specifications produce names
such as `temperature_power_1` and `temperature_term_1`.

For `degree=3, restricted=True, basis="truncated_power"`, the formula and scaling
match **stagg**: `T` plus `K-2` unnormalized restricted cubic terms. The example
above returns `temperature_power_1`, `temperature_term_1`, and
`temperature_term_2`. Matching a whole stagg panel also requires matching weights,
time grouping, and missing-data policies.

**B-spline basis.** This evaluates B-splines lazily on the existing chunks,
with a fixed coefficient transformation enforcing the requested restrictions.
Columns are named `temperature_bspline_1`, etc. They are linear combinations of
B-splines, so constrained output columns need not retain local support or
nonnegativity. Basis construction depends only on the specification, not the
observed temperatures. Restricted tails are explicitly evaluated as linear
functions. No fitting or smoothing of the temperature data takes place.

The B-spline columns are anchored to zero at an auxiliary point one knot-span
below the first knot (a span of at least 1 in input units). The power basis uses
its original uncentered convention. **The bases span the same function space
when a constant is included**, but individual columns and coefficients differ.
For annual sums that constant becomes a valid-day count: include that count in
the regression if exposure duration varies and is not absorbed by fixed effects.
Penalties on coefficients are also basis-dependent. A B-spline representation
improves numerical scaling but cannot resolve a lack of independent exposure
variation after fixed effects.

NaNs are preserved by the transform; subsequent temporal and spatial reductions
retain their existing missing-data behavior. Use `mean` instead of `sum` for
average daily exposure. Both bases work with either temporal engine.

Calling `Dataset.spline()` or using `{"transform": "spline"}` without options
still returns the original `T` and `(T-20).clip(min=0)` expansion. The parameter-free
pipeline also keeps the historical `_spline1` and `_spline2` column names.

## Choosing a temporal engine

`aggregate_dataset` and `aggregate_time` accept `engine=`:

| Value | Behavior |
|---|---|
| `"auto"` *(default)* | Resolved per step from the spatial chunk size. |
| `"numba"` | Force the compiled kernel. |
| `"dask"` | Force the vectorized Dask path. |

The numba engine is **bit-equivalent** to the dask path — it changes speed, not
results. It wins dramatically on small/native spatial chunks and *loses* on large
rechunked blocks. `"auto"` picks correctly for you; see
[Execution & scaling](execution.md) for the details and the crossover rule.

## Output

The return value is a pandas DataFrame with one row per region per period and one
column per named output variable (fanned out by exponent or threshold where
applicable), merged back onto the region ids from your shapefile.
