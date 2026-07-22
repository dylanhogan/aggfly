# Population weights are ordinary secondary weights with wtype="pop". Everything
# here is a thin wrapper over the generic implementation in secondary_weights.py,
# kept so the public names stay stable.

from .secondary_weights import SecondaryWeights, secondary_weights_from_path


class PopWeights(SecondaryWeights):
    """
    Secondary weights from a population raster (LandScan, WorldPop, ...).

    Equivalent to ``SecondaryWeights`` with ``wtype="pop"``. It subclasses rather
    than aliases so ``isinstance(w, PopWeights)`` and the existing type hints
    keep working.
    """

    def __init__(self, raster, name=None, path=None, project_dir=None,
                 cache_identifier=None):
        super().__init__(
            raster, name=name, path=path, project_dir=project_dir,
            wtype="pop", cache_identifier=cache_identifier,
        )


def pop_weights_from_path(path, name=None, project_dir=None, crs=None,
                          var=None, sel=None, cache_identifier=None,
                          preprocess=None, **kwargs):
    """
    Create PopWeights from a raster file.

    Parameters
    ----------
    path : str
        Path to the population raster (.tif, .zarr or .nc).
    name : str, optional
        A label for these weights.
    project_dir : str, optional
        Project directory; enables the cache.
    crs : str, optional
        Write this CRS onto the raster.
    var, sel, cache_identifier, preprocess
        See :func:`~aggfly.weights.secondary_weights.secondary_weights_from_path`.

    Returns
    -------
    PopWeights
    """
    w = secondary_weights_from_path(
        path, name=name, project_dir=project_dir, crs=crs, wtype="pop",
        var=var, sel=sel, cache_identifier=cache_identifier,
        preprocess=preprocess, **kwargs,
    )
    # Re-tag the instance rather than rebuild it: SecondaryWeights and
    # PopWeights differ only in wtype, which is already set.
    w.__class__ = PopWeights
    return w
