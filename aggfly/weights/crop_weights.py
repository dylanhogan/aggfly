# Crop weights are ordinary secondary weights: the crop is a coordinate
# selection on the raster, and the feed regime is a cache discriminator. Both are
# now expressed through the generic machinery in secondary_weights.py — `sel`
# and `cache_identifier` respectively — so everything here is a thin wrapper
# kept so the public names stay stable.

from .secondary_weights import SecondaryWeights, secondary_weights_from_path


class CropWeights(SecondaryWeights):
    """
    Secondary weights from a cropland raster.

    Equivalent to ``SecondaryWeights`` with ``wtype`` set to the crop and
    ``cache_identifier`` set to the feed regime. It subclasses rather than
    aliases so ``isinstance(w, CropWeights)`` and the existing type hints keep
    working.

    Attributes
    ----------
    feed : str or None
        The feed regime (rainfed / irrigated / total). Stored as the cache
        identifier, so two regimes never share a cache entry.
    """

    def __init__(self, raster, crop="corn", name=None, feed=None, path=None,
                 project_dir=None):
        super().__init__(
            raster, name=name, path=path, project_dir=project_dir,
            wtype=crop, cache_identifier=feed,
        )

    @property
    def feed(self):
        """The feed regime; an alias for the cache identifier."""
        return self.cache_identifier


def crop_weights_from_path(path, crop="corn", name=None, feed=None,
                           project_dir=None, crs=None, var="layer",
                           preprocess=None, **kwargs):
    """
    Create CropWeights from a cropland raster.

    A convenience wrapper: ``crop`` selects along the raster's ``crop``
    coordinate and ``feed`` becomes the cache discriminator. The equivalent
    generic call is::

        secondary_weights_from_path(path, var="layer", sel={"crop": crop},
                                    wtype=crop, cache_identifier=feed)

    Parameters
    ----------
    path : str
        Path to the cropland raster (.zarr, .nc or .tif).
    crop : str, optional
        Crop to select along the ``crop`` coordinate (default "corn").
    name : str, optional
        A label for these weights.
    feed : str, optional
        Feed regime (rainfed / irrigated / total). Kept out of the raster but
        folded into the cache key.
    project_dir : str, optional
        Project directory; enables the cache.
    crs : str, optional
        Write this CRS onto the raster.
    var : str, optional
        Data variable holding the cropland layer (default "layer"). Pass None
        for a file that is already a bare DataArray.

    Returns
    -------
    CropWeights
    """
    w = secondary_weights_from_path(
        path, name=name, project_dir=project_dir, crs=crs, wtype=crop,
        var=var, sel={"crop": crop}, cache_identifier=feed,
        preprocess=preprocess, **kwargs,
    )
    w.__class__ = CropWeights
    return w
