import numpy as np
import xarray as xr
from datasets.utils.file_utils import NonStreamableDatasetError

from src.constants import SCALING_LADDER


def init_default_coords():
  param_scales = list(SCALING_LADDER['models'].keys())
  token_budgets = list(SCALING_LADDER['batch_size_vs_token_budget_strategy']['staggered_grid'].keys())
  global_batch_sizes = SCALING_LADDER['batch_sizes']
  learning_rates = SCALING_LADDER['learning_rates']

  return {'n': param_scales, 'd': token_budgets, 'gbs': global_batch_sizes, 'lr': learning_rates}


class ScalingMetricTensor:
  def __init__(self, data, coords=None):
    """
    coords = {
        "n": [...],
        "d": [...],
        "gbs": [...],
        "lr": [...],
    }
    """
    if coords is None:
      coords = init_default_coords()
    else:
      pass

    self.stored_coords = coords

    dims = tuple(coords.keys())
    expected_shape = tuple(len(coords[d]) for d in dims)
    data = np.asarray(data)

    if data.shape != expected_shape:
      raise ValueError(f'Shape mismatch.\nExpected {expected_shape}\nGot      {data.shape}')

    self._da = xr.DataArray(
      data,
      dims=dims,
      coords=coords,
    )
    self._validate_unique_coords()

  @classmethod
  def empty(cls, coords, fill_value=np.nan):
    shape = tuple(len(coords[d]) for d in coords)

    return cls(
      np.full(shape, fill_value),
      coords,
    )

  @classmethod
  def from_tensor(cls, tensor, coords):
    return cls(tensor, coords)

  def _validate_unique_coords(self):
    for dim in self._da.dims:
      values = self._da.coords[dim].values

      if len(values) != len(set(values)):
        raise ValueError(f'{dim} contains duplicate coordinates')

  def _convert_key(self, key):
    if not isinstance(key, tuple):
      key = (key,)
    if len(key) > self._da.ndim:
      raise IndexError(f'Too many indices ({len(key)})')
    key = list(key)

    while len(key) < self._da.ndim:
      key.append(slice(None))
    result = {}

    for dim, item in zip(self._da.dims, key):
      if item is Ellipsis:
        result[dim] = slice(None)
      elif isinstance(item, slice):
        result[dim] = item
      elif isinstance(item, (list, tuple)):
        result[dim] = list(item)
      else:
        result[dim] = item

    return result

  def __getitem__(self, key):
    selection = self._convert_key(key)
    out = self._da.sel(selection)

    if out.ndim == 0:
      return out.item()

    return ScalingMetricTensor(
      out.values,
      {d: out.coords[d].values.tolist() for d in out.dims},
    )

  def __setitem__(self, key, value):
    selection = self._convert_key(key)
    self._da.loc[selection] = value

  def sel(self, **kwargs):
    out = self._da.sel(kwargs)
    if out.ndim == 0:
      return out.item()
    return self._wrap(out)

  def isel(self, **kwargs):
    out = self._da.isel(kwargs)
    return self._wrap(out)

  def mean(self, axis):
    return self._wrap(self._da.mean(dim=axis))

  def min(self, axis):
    return self._wrap(self._da.min(dim=axis))

  def max(self, axis):
    return self._wrap(self._da.max(dim=axis))

  def std(self, axis):
    return self._wrap(self._da.std(dim=axis))

  def argmin(self, axis):
    return self._wrap(self._da.argmin(dim=axis))

  def argmax(self, axis):
    return self._wrap(self._da.argmax(dim=axis))

  def argmin_full(self):
    flat_idx = np.nanargmin(self._da.values)
    unravel = np.unravel_index(flat_idx, self._da.shape)
    return {
      dim: self._da.coords[dim].values[idx]
      for dim, idx in zip(
        self._da.dims,
        unravel,
      )
    }

  def argmax_full(self):
    flat_idx = np.nanargmax(self._da.values)
    unravel = np.unravel_index(flat_idx, self._da.shape)
    return {
      dim: self._da.coords[dim].values[idx]
      for dim, idx in zip(
        self._da.dims,
        unravel,
      )
    }

  def coord(self, axis):
    return self._da.coords[axis].values

  @property
  def shape(self):
    return self._da.shape

  @property
  def numel(self):
    return self._da.size

  @property
  def dims(self):
    return self._da.dims

  @property
  def values(self):
    return self._da.values

  def to_xarray(self):
    return self._da

  def save(self, path):
    self._da.to_netcdf(path)

  @classmethod
  def load(cls, path):
    da = xr.load_dataarray(path)

    return cls(
      da.values,
      {d: da.coords[d].values.tolist() for d in da.dims},
    )

  def _wrap(self, da):
    if da.ndim == 0:
      return da.item()
    return ScalingMetricTensor(
      da.values,
      {d: da.coords[d].values.tolist() for d in da.dims},
    )

  def __repr__(self):
    return repr(self._da)

  def at(self, **coords):
    indexers = {}
    for k, v in coords.items():
      if isinstance(v, (list, tuple)):
        indexers[k] = list(v)
      else:
        indexers[k] = v
    out = self._da.sel(indexers)
    if out.ndim == 0:
      return out.item()
    return self._wrap(out)

  def set(self, value, **coords):
    self._da.loc[coords] = value
