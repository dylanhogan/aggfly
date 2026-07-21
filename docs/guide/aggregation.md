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
