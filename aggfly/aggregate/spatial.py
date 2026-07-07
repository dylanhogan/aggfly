"""
SpatialAggregator: aggregate (temporally-reduced) climate data onto regions using
grid weights.

The weighted regional average

    result[region, t] = Σ_cell weight[region, cell] · clim[cell, t]
                        ─────────────────────────────────────────────
                             Σ_cell weight[region, cell]           (over cells valid at t)

is a sparse (region × cell) operator applied to the (cell × time) climate matrix —
two scatter-add reductions (numerator and per-timestep valid-weight denominator).
It runs lazily through ``dask.array.map_blocks`` over the time chunks: each block is
independent, so there is no dataframe shuffle and the full space×time cube is never
materialized. NaN cells are excluded per timestep (the denominator only counts valid
weight), and a region/time with no valid cells is dropped — matching the previous
groupby implementation.

Attributes:
    dataset (list): list of Datasets holding the climate data (one per output name).
    grid: the weight grid (provides ``cell_id`` ordering).
    weights (DataFrame): region weights, columns ``cell_id``, ``index_right``, ``weight``.
    names (list): output variable names, aligned with ``dataset``.
"""
from typing import List, Union

import numpy as np
import pandas as pd
import xarray as xr
import dask
import dask.array as da

from ..dataset import Dataset
from ..weights import GridWeights


class SpatialAggregator:
    """Spatially aggregate climate data onto regions using grid weights."""

    def __init__(
        self,
        dataset: Union[list, Dataset],
        weights: GridWeights,
        names: Union[str, List[str]] = "climate",
    ) -> None:
        """
        Parameters
        ----------
        dataset : list or Dataset
            A Dataset (or list of them, one per output name) holding the
            temporally-aggregated climate data with dims (time, latitude, longitude).
        weights : GridWeights
            Grid weights; provides the grid cell ordering and the region weights table.
        names : str or list of str, optional
            Output variable name(s), aligned with ``dataset``.
        """
        self.dataset = dataset if isinstance(dataset, list) else [dataset]

        # Rescale longitude for datasets if necessary (match the grid convention)
        _ = [x.rescale_longitude() for x in self.dataset if x.lon_is_360]

        self.grid = weights.grid
        self.weights = weights.weights
        self.names = [names] if isinstance(names, str) else names

    def compute(self, npartitions: int = None) -> pd.DataFrame:
        """
        Compute the weighted average of the climate data over each region and time.

        Parameters
        ----------
        npartitions : int, optional
            Deprecated / unused. Parallelism now follows the climate array's existing
            time chunking (one ``map_blocks`` task per time chunk); kept for backward
            compatibility with callers that still pass it.

        Returns
        -------
        pandas.DataFrame
            Long-format frame with columns ``region_id``, ``time``, and one column per
            output name.
        """
        # Combine the (lazy) climate arrays and stack the spatial dims into a single
        # cell axis ordered to match grid.cell_id.
        clim_ds = xr.combine_by_coords(
            [x.da.to_dataset(name=self.names[i]) for i, x in enumerate(self.dataset)]
        )
        stacked = (
            clim_ds.stack(cell=["latitude", "longitude"])
            .drop_vars(["cell", "latitude", "longitude"])
            .assign_coords(cell=("cell", self.grid.cell_id))
        )
        time = stacked["time"].values
        cell_ids = stacked["cell"].values
        n_time = len(time)

        # Build the sparse weight operator as COO triplets (region_row, cell_col, w).
        region_idx, cell_idx, w_vals, region_ids = _weight_triplets(
            self.weights, cell_ids
        )
        n_regions = len(region_ids)

        # Per-name (cell, time) dask arrays with the cell axis in a single chunk (the
        # scatter reduces over all cells, so a block must see every cell).
        arrs = {
            nm: da.asarray(stacked[nm].transpose("cell", "time").data).rechunk({0: -1})
            for nm in self.names
        }
        # Shared validity mask (a cell/time is used only if every name is non-NaN there,
        # matching the previous dropna(subset=names) behaviour).
        valid = None
        for nm in self.names:
            v = ~da.isnan(arrs[nm])
            valid = v if valid is None else (valid & v)

        kw = dict(region_idx=region_idx, cell_idx=cell_idx, w_vals=w_vals, n_regions=n_regions)
        den = _scatter(valid.astype(float), **kw)
        nums = {nm: _scatter(da.where(valid, arrs[nm], 0.0), **kw) for nm in self.names}

        den_v, nums_v = dask.compute(den, nums)

        with np.errstate(invalid="ignore", divide="ignore"):
            res = {
                nm: np.divide(
                    nums_v[nm], den_v, out=np.full_like(den_v, np.nan), where=den_v != 0
                )
                for nm in self.names
            }

        # Assemble the long-format output; drop region/time cells with no valid weight.
        out = pd.DataFrame(
            {
                "region_id": np.repeat(region_ids, n_time),
                "time": np.tile(time, n_regions),
            }
        )
        for nm in self.names:
            out[nm] = res[nm].reshape(-1)
        out = out.dropna(subset=self.names).reset_index(drop=True)
        return out


def _weight_triplets(wdf: pd.DataFrame, cell_ids: np.ndarray):
    """
    Turn the region weights table into COO triplets aligned to the stacked cell order.

    Returns (region_idx, cell_idx, w_vals, region_ids) where region_idx/cell_idx are
    integer positions (into 0..n_regions and 0..n_cells) and region_ids maps a row
    position back to its original region id.
    """
    cellpos = {int(c): i for i, c in enumerate(cell_ids)}
    region_ids = np.sort(wdf["index_right"].unique())
    regionpos = {r: i for i, r in enumerate(region_ids)}

    rows = wdf["index_right"].map(regionpos).to_numpy()
    cols = wdf["cell_id"].map(cellpos).to_numpy()
    # Drop weight entries for cells absent from the climate grid.
    keep = ~pd.isna(cols)
    return (
        rows[keep].astype(np.intp),
        cols[keep].astype(np.intp),
        wdf["weight"].to_numpy(dtype=float)[keep],
        region_ids,
    )


def _scatter_block(block, region_idx, cell_idx, w_vals, n_regions):
    """Weighted region scatter-add for one (n_cells, n_time_block) block."""
    contrib = w_vals[:, None] * block[cell_idx, :]          # (n_entries, t_block)
    out = np.zeros((n_regions, block.shape[1]), dtype=float)
    np.add.at(out, region_idx, contrib)                     # sum entries into region rows
    return out


def _scatter(x: "da.Array", region_idx, cell_idx, w_vals, n_regions) -> "da.Array":
    """Apply the region scatter operator lazily across the time chunks of x."""
    return x.map_blocks(
        _scatter_block,
        region_idx=region_idx,
        cell_idx=cell_idx,
        w_vals=w_vals,
        n_regions=n_regions,
        dtype=float,
        chunks=((n_regions,), x.chunks[1]),
    )
