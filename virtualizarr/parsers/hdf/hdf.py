from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Iterable,
    Literal,
    Mapping,
)

import numcodecs
import numpy as np
from obspec_utils.protocols import ReadableFile
from obspec_utils.readers import BlockStoreReader
from obspec_utils.registry import ObjectStoreRegistry

from virtualizarr.codecs import zarr_codec_config_to_v3
from virtualizarr.manifests import (
    ChunkEntry,
    ChunkManifest,
    ManifestArray,
    ManifestGroup,
    ManifestStore,
)
from virtualizarr.manifests.utils import create_v3_array_metadata
from virtualizarr.parsers.hdf.filters import codecs_from_dataset
from virtualizarr.parsers.typing import ReaderFactory
from virtualizarr.parsers.utils import encode_cf_fill_value
from virtualizarr.types import ChunkKey
from virtualizarr.utils import soft_import

h5py = soft_import("h5py", "reading hdf files", strict=False)


if TYPE_CHECKING:
    from h5py import Dataset as H5Dataset
    from h5py import File as H5File
    from h5py import Group as H5Group
    from h5py import Reference as H5Reference


def _get_fill_value(dataset: H5Dataset):
    """
    Extract the fill value from an h5py dataset, handling string/bytes dtypes
    that don't return numpy scalars from dataset.fillvalue.
    """
    try:
        raw = dataset.fillvalue
    except RuntimeError:
        return np.ma.default_fill_value(dataset.dtype)
    if h5py.check_vlen_dtype(dataset.dtype) in (str, bytes):
        # Variable-length string fill values come back as raw bytes; the array
        # is virtualized as VariableLengthUTF8, so the fill value must be str.
        # Decode strictly: a fill value that is not valid UTF-8 cannot be
        # represented in that dtype, and silently replacing bytes would corrupt it.
        if isinstance(raw, bytes):
            return raw.decode("utf-8")
        return raw
    elif isinstance(raw, np.generic):
        return raw.item()
    else:
        return raw


NonVirtualizableAction = Literal["load", "drop"]


@dataclass(frozen=True)
class NonVirtualizableKind:
    description: str
    loadable: bool


def _non_virtualizable_kind(dtype: np.dtype) -> NonVirtualizableKind | None:
    """
    Describe why an HDF5 dtype can't be virtualized, or return None if it can.

    These dtypes store something other than the data in their chunks: variable-length
    values are pointers into the HDF5 global heap, and references are file addresses
    only an HDF5 library can resolve.
    """
    if dtype.names:
        members = [_non_virtualizable_kind(dtype[name]) for name in dtype.names]
        member_kinds = [kind for kind in members if kind is not None]
        if not member_kinds:
            return None
        descriptions = sorted({kind.description for kind in member_kinds})
        return NonVirtualizableKind(
            description=f"compound with {' and '.join(descriptions)} member",
            loadable=all(kind.loadable for kind in member_kinds),
        )
    if h5py.check_vlen_dtype(dtype) in (str, bytes):
        return NonVirtualizableKind("variable-length string", loadable=True)
    ref = h5py.check_ref_dtype(dtype)
    if ref is h5py.RegionReference:
        return NonVirtualizableKind("region reference", loadable=False)
    if ref is h5py.Reference:
        return NonVirtualizableKind("object reference", loadable=True)
    if h5py.check_vlen_dtype(dtype) is not None:
        return NonVirtualizableKind("variable-length sequence", loadable=False)
    if dtype.kind == "O":
        return NonVirtualizableKind("object", loadable=False)
    return None


@dataclass
class NonVirtualizablePolicy:
    """
    Decide what to do with each non-virtualizable dataset, and collect the datasets
    the user hasn't made a valid choice for so they can be reported together.
    """

    choice: NonVirtualizableAction | Mapping[str, NonVirtualizableAction] | None = None
    unresolved: list[str] = field(default_factory=list)
    unloadable: list[str] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.choice, str) or self.choice is None:
            actions = [] if self.choice is None else [self.choice]
        else:
            actions = list(self.choice.values())
            if relative := [
                p for p in self.choice if p != "*" and not p.startswith("/")
            ]:
                raise ValueError(
                    "non_virtualizable keys must be dataset paths from the file root, "
                    f'such as "/group/labels", or "*". Got {relative}.'
                )
            if self.choice.get("*", "drop") != "drop":
                raise ValueError(
                    'The "*" entry of non_virtualizable only accepts "drop". Loading '
                    "copies data into the store, so each dataset to load must be "
                    "listed by path."
                )
        for action in actions:
            if action not in ("load", "drop"):
                raise ValueError(
                    f'non_virtualizable actions must be "load" or "drop", got {action!r}'
                )

    def action(
        self, dataset: H5Dataset, kind: NonVirtualizableKind
    ) -> NonVirtualizableAction | None:
        action: NonVirtualizableAction | None
        if isinstance(self.choice, str):
            action = self.choice
        elif self.choice is not None:
            action = self.choice.get(dataset.name) or self.choice.get("*")
        else:
            action = None
        description = f"  {dataset.name}: {kind.description}, shape {dataset.shape}"
        if action is None:
            self.unresolved.append(description)
        elif action == "load" and not kind.loadable:
            self.unloadable.append(description)
            return None
        return action

    def raise_if_unresolved(self):
        if self.unloadable:
            raise ValueError(
                "HDFParser can't load these datasets, because their values have no "
                "Zarr representation:\n" + "\n".join(self.unloadable) + "\n"
                'Map their paths to "drop" in non_virtualizable to leave them out.'
            )
        if self.unresolved:
            raise ValueError(
                f"{len(self.unresolved)} dataset(s) in this file can't be virtualized, "
                "because their chunks store pointers into the HDF5 file rather than "
                "their values:\n" + "\n".join(self.unresolved) + "\n"
                "Choose what HDFParser does with them using `non_virtualizable`:\n"
                '  "load" reads the values now and copies them into the store\n'
                '  "drop" leaves the datasets out\n'
                '  a dict maps each dataset path to "load" or "drop", with an '
                'optional "*": "drop" entry for every dataset not listed.'
            )


class ReferencePaths:
    """
    Resolve HDF5 object references to the path of the object they point to.

    h5py's ``file[ref].name`` makes HDF5 search the file for a path on every call,
    which cost about 10 ms per reference on a real NWB file. Indexing every object's
    address once makes each lookup a dict access. Addresses are stable across
    handles, so one index serves every time the file is reopened during a parse.
    """

    def __init__(self):
        self._paths: dict[int, str] | None = None

    def __call__(self, file: H5File, ref: H5Reference) -> str:
        if not ref:
            return ""
        if self._paths is None:
            paths = {h5py.h5o.get_info(file.id).addr: "/"}

            def index(name, obj):
                paths.setdefault(h5py.h5o.get_info(obj.id).addr, f"/{name}")

            file.visititems(index)
            self._paths = paths
        try:
            target = h5py.h5r.dereference(ref, file.id)
            return self._paths[h5py.h5o.get_info(target).addr]
        except KeyError as err:
            raise ValueError(
                f"An object reference in {file.filename!r} points to an object that "
                "has no path in the file."
            ) from err


def _load_values(
    values: np.ndarray,
    dtype: np.dtype,
    file: H5File,
    references: ReferencePaths,
) -> np.ndarray:
    """
    Convert values read by h5py into a dtype Zarr can store: strings for variable-length
    strings, the referenced object's path for object references, and fixed-width strings
    for those same members inside a compound dtype.
    """
    if dtype.names:
        members = {
            name: _load_values(values[name], dtype[name], file, references)
            for name in dtype.names
        }
        # numpy structured dtypes can't hold StringDType fields
        fields = [
            (
                name,
                f"U{max(int(np.strings.str_len(member).max(initial=0)), 1)}"
                if member.dtype == np.dtypes.StringDType()
                else member.dtype,
            )
            for name, member in members.items()
        ]
        loaded = np.empty(values.shape, dtype=fields)
        for name, member in members.items():
            loaded[name] = member
        return loaded
    if h5py.check_ref_dtype(dtype) is h5py.Reference:
        values = np.asarray(values, dtype=object)
        strings = [references(file, ref) for ref in values.ravel()]
    elif h5py.check_vlen_dtype(dtype) in (str, bytes):
        values = np.asarray(values, dtype=object)
        strings = [
            v.decode("utf-8") if isinstance(v, bytes) else v for v in values.ravel()
        ]
    else:
        return values
    return np.array(strings, dtype=np.dtypes.StringDType()).reshape(values.shape)


def _inlined_manifest_array(
    dataset: H5Dataset,
    group: str,
    references: ReferencePaths,
) -> ManifestArray:
    """
    Construct a ManifestArray whose chunks hold the dataset's values, read with h5py,
    for datasets whose chunk bytes aren't their data.

    The HDF5 chunk grid is kept so each chunk is read and encoded separately, trimmed
    to the array shape since the chunks are rewritten anyway.
    """
    if dataset.chunks is None:
        chunks = tuple(max(s, 1) for s in dataset.shape)
    else:
        chunks = tuple(min(c, max(s, 1)) for c, s in zip(dataset.chunks, dataset.shape))

    try:
        if dataset.dtype.names:
            # member widths depend on every value, so load the whole dataset at once
            values = _load_values(dataset[()], dataset.dtype, dataset.file, references)
            dtype = values.dtype
            fill_value = None
            padding = np.zeros((), dtype=dtype)

            def read(selection):
                return values[selection]

            def encode(chunk):
                return chunk.tobytes()
        else:
            dtype = np.dtypes.StringDType()
            fill_value = (
                _get_fill_value(dataset)
                if h5py.check_vlen_dtype(dataset.dtype) in (str, bytes)
                else ""
            )
            padding = np.array(fill_value, dtype=dtype)

            def read(selection):
                return _load_values(
                    dataset[selection], dataset.dtype, dataset.file, references
                )

            def encode(chunk):
                return numcodecs.VLenUTF8().encode(chunk.astype(object))

        grid = tuple(math.ceil(s / c) for s, c in zip(dataset.shape, chunks))
        entries = {}
        for index in np.ndindex(grid):
            selection = tuple(
                slice(i * c, min((i + 1) * c, s))
                for i, c, s in zip(index, chunks, dataset.shape)
            )
            chunk_values = read(selection)
            # Zarr decodes every chunk at the full chunk shape, including edge chunks
            chunk = np.full(chunks, padding, dtype=dtype)
            chunk[tuple(slice(0, n) for n in chunk_values.shape)] = chunk_values
            data = encode(chunk)
            key = ".".join(map(str, index)) or "0"
            entries[key] = {"path": "", "offset": 0, "length": len(data), "data": data}
    except UnicodeDecodeError as err:
        raise ValueError(
            f"Dataset {dataset.name!r} holds strings that aren't valid UTF-8, so "
            'HDFParser can\'t load them. Map its path to "drop" in non_virtualizable '
            "to leave it out."
        ) from err

    metadata = create_v3_array_metadata(
        shape=dataset.shape,
        data_type=dtype,
        chunk_shape=chunks,
        fill_value=fill_value,
        dimension_names=tuple(_dataset_dims(dataset, group=group)),
        attributes=_extract_attrs(dataset, references),
    )
    manifest = ChunkManifest(entries, shape=grid)
    return ManifestArray(metadata=metadata, chunkmanifest=manifest)


def _construct_manifest_array(
    filepath: str,
    dataset: H5Dataset,
    group: str,
    references: ReferencePaths,
) -> ManifestArray:
    """
    Construct a ManifestArray from an h5py dataset

    Parameters
    ----------
    filepath
        The path of the hdf5 file.
    dataset
        An h5py dataset.
    group
        Name of the group containing this h5py.Dataset.
    references
        Resolves object references in the dataset's attributes to paths.

    Returns
    -------
    ManifestArray
    """
    chunks = _chunk_shape(dataset)
    codecs = codecs_from_dataset(dataset)
    attrs = _extract_attrs(dataset, references)
    dtype = dataset.dtype

    # Temporarily disable use CF->Codecs - TODO re-enable in subsequent PR.
    # cfcodec = cfcodec_from_dataset(dataset)
    # if cfcodec:
    # codecs.insert(0, cfcodec["codec"])
    # dtype = cfcodec["target_dtype"]
    # attrs.pop("scale_factor", None)
    # attrs.pop("add_offset", None)
    # else:
    # dtype = dataset.dtype

    if "_FillValue" in attrs and dtype.kind not in ("S", "U", "O", "T"):
        encoded_cf_fill_value = encode_cf_fill_value(attrs["_FillValue"], dtype)
        attrs["_FillValue"] = encoded_cf_fill_value

    codec_configs = [zarr_codec_config_to_v3(codec.get_config()) for codec in codecs]

    fill_value = _get_fill_value(dataset)
    dims = tuple(_dataset_dims(dataset, group=group))
    metadata = create_v3_array_metadata(
        shape=dataset.shape,
        data_type=dtype,
        chunk_shape=chunks,
        fill_value=fill_value,
        codecs=codec_configs,
        dimension_names=dims,
        attributes=attrs,
    )
    manifest = _dataset_chunk_manifest(filepath, dataset, chunks=chunks)
    return ManifestArray(metadata=metadata, chunkmanifest=manifest)


def _chunk_shape(dataset: H5Dataset) -> tuple[int, ...]:
    """
    Determine the chunk shape to report for an h5py dataset.

    For a dataset along an unlimited (extendable) dimension, h5py reports the
    chunk shape allocated for the full maxshape, which can exceed the actual
    array shape - e.g. a coordinate holding 5 values along an unlimited
    dimension reports ``chunks=(512,)``. An oversized chunk inhibits
    concatenation of the resulting virtual dataset, so trim it down to the array
    shape where it is safe to do so.

    Trimming the chunk shrinks the in-bounds region the chunk covers, so the
    manifest must point at fewer bytes than the full stored chunk. That region
    is only a contiguous byte range - and so expressible as a single manifest
    entry - when the chunk is unfiltered (uncompressed) and only the leading
    (slowest-varying) dimension is trimmed. When an oversized chunk can't be
    trimmed safely (e.g. it is compressed) the original chunk shape is kept: the
    variable still reads correctly (zarr crops the oversized edge chunk) and can
    be written as virtual references, but it can't be concatenated with other
    virtual datasets (the oversized chunk prevents a regular chunk grid). That
    case is surfaced to the user as a warning at
    ``ManifestStore.to_virtual_dataset`` time, suggesting they load the variable
    instead.

    This relies on the same invariant as the sub-chunk slicing in
    ``virtualizarr.manifests.indexing`` (a contiguous sub-range of an
    uncompressed, fixed-order chunk is addressable as a single byte range);
    trimming here is the special case of taking the leading prefix along axis 0.
    """
    shape = dataset.shape
    # Clamp each dim to >= 1: zarr v3 allows shape=(0,) but forbids zero-length
    # chunk dimensions (enforced by zarr-python >= 3.2.0). See
    # https://github.com/zarr-developers/zarr-python/issues/3711.
    if dataset.chunks is None:
        return tuple(max(s, 1) for s in shape)

    chunks = tuple(min(c, max(s, 1)) for c, s in zip(dataset.chunks, shape))
    if chunks == dataset.chunks:
        return chunks

    unfiltered = dataset.id.get_create_plist().get_nfilters() == 0
    leading_dim_only = chunks[1:] == dataset.chunks[1:]
    if unfiltered and leading_dim_only:
        return chunks
    return dataset.chunks


def _construct_manifest_group(
    filepath: str,
    reader: ReadableFile,
    *,
    group: str | None = None,
    drop_variables: Iterable[str] | None = None,
    non_virtualizable: NonVirtualizablePolicy,
    references: ReferencePaths,
) -> ManifestGroup:
    """
    Construct a virtual Group from a HDF dataset.
    """
    import h5py

    with h5py.File(reader, mode="r") as f:
        if not isinstance(g := f.get(group or "/"), h5py.Group):
            raise ValueError(f"Group {group!r} is not an HDF Group")

        # Several of our test fixtures which use xr.tutorial data have
        # non coord dimensions serialized using big endian dtypes which are not
        # yet supported in zarr-python v3.  We'll drop these variables for the
        # moment until big endian support is included upstream.

        non_coordinate_dimension_vars = _find_non_coord_dimension_vars(group=g)
        drop_variables = set(drop_variables or ()) | set(non_coordinate_dimension_vars)
        group_name = str(g.name)  # NOTE: this will always include leading "/"
        arrays = {}
        for key in g.keys():
            if key in drop_variables or not isinstance(dataset := g[key], h5py.Dataset):
                continue
            kind = _non_virtualizable_kind(dataset.dtype)
            if kind is None:
                arrays[key] = _construct_manifest_array(
                    filepath, dataset, group_name, references
                )
            elif non_virtualizable.action(dataset, kind) == "load":
                arrays[key] = _inlined_manifest_array(dataset, group_name, references)
        groups = {
            key: _construct_manifest_group(
                filepath,
                reader,
                group=str(Path(group) / key) if group is not None else key,
                non_virtualizable=non_virtualizable,
                references=references,
            )
            for key in g.keys()
            if key not in drop_variables
            if isinstance(g[key], h5py.Group)
        }
        attributes = _extract_attrs(g, references)

    return ManifestGroup(arrays=arrays, groups=groups, attributes=attributes)


class HDFParser:
    """Create a [ManifestStore][virtualizarr.manifests.ManifestStore] from an HDF5/NetCDF4 file.

    Parameters
    ----------
    group
        Name of the group within the HDF5 file to virtualize.
    drop_variables
        Variables in the file that will be ignored when creating the ManifestStore
        (default: `None`, do not ignore any variables).
    reader_factory
        A callable that creates a file-like reader from a store and path.
        Must return an object implementing the
        [ReadableFile][obspec_utils.protocols.ReadableFile] protocol.
        Default is [BlockStoreReader][obspec_utils.readers.BlockStoreReader].
    non_virtualizable
        What to do with datasets whose chunks store pointers into the HDF5 file
        rather than their values: variable-length strings, object references, and
        compound datasets with either as a member. These can't be read through
        virtual references.

        - `"load"` reads their values with h5py and stores them as inlined chunks.
          Object references become the path of the object they point to, and
          compound members become fixed-width strings. The values are copied into
          any store the result is written to.
        - `"drop"` leaves them out.
        - A dict maps dataset paths (from the file root, e.g. `"/group/labels"`) to
          `"load"` or `"drop"`. A `"*": "drop"` entry drops every dataset not listed.

        The default, `None`, raises an error listing every such dataset, so that
        copying data into the store is always a deliberate choice. Region
        references and variable-length sequences have no Zarr representation, so
        they can only be dropped.
    """

    def __init__(
        self,
        group: str | None = None,
        drop_variables: Iterable[str] | None = None,
        reader_factory: ReaderFactory = BlockStoreReader,
        non_virtualizable: NonVirtualizableAction
        | Mapping[str, NonVirtualizableAction]
        | None = None,
    ):
        NonVirtualizablePolicy(non_virtualizable)
        self.group = group
        self.drop_variables = drop_variables
        self.reader_factory = reader_factory
        self.non_virtualizable = non_virtualizable

    def __call__(
        self,
        url: str,
        registry: ObjectStoreRegistry,
    ) -> ManifestStore:
        """
        Parse the metadata and byte offsets from a given HDF5/NetCDF4 file to produce a VirtualiZarr
        [ManifestStore][virtualizarr.manifests.ManifestStore].

        Parameters
        ----------
        url
            The URL of the input HDF5/NetCDF4 file (e.g., `"s3://bucket/store.zarr"`).
        registry
            An [ObjectStoreRegistry][obspec_utils.registry.ObjectStoreRegistry] for resolving urls and reading data.

        Returns
        -------
        ManifestStore
            A [ManifestStore][virtualizarr.manifests.ManifestStore] which provides a Zarr representation of the parsed file.
        """
        store, path_in_store = registry.resolve(url)
        reader = self.reader_factory(store, path_in_store)
        non_virtualizable = NonVirtualizablePolicy(self.non_virtualizable)
        manifest_group = _construct_manifest_group(
            filepath=url,
            reader=reader,
            group=self.group,
            drop_variables=self.drop_variables,
            non_virtualizable=non_virtualizable,
            references=ReferencePaths(),
        )
        non_virtualizable.raise_if_unresolved()
        # Convert to a manifest store
        return ManifestStore(registry=registry, group=manifest_group)


def _dataset_chunk_manifest(
    filepath: str,
    dataset: H5Dataset,
    *,
    chunks: tuple[int, ...],
) -> ChunkManifest:
    """
    Generate ChunkManifest for HDF5 dataset.

    Parameters
    ----------
    filepath
        The path of the HDF5 file
    dataset
        h5py dataset for which to create a ChunkManifest
    chunks
        The chunk shape to use, as returned by ``_chunk_shape``. This may be
        smaller than ``dataset.chunks`` when an oversized chunk has been trimmed
        to the array shape (see ``_chunk_shape``), in which case each chunk's
        byte length is recomputed for the trimmed, in-bounds region.

    Returns
    -------
    ChunkManifest
        A Virtualizarr ChunkManifest
    """
    dsid = dataset.id
    if dataset.chunks is None:
        if dsid.get_offset() is None:
            chunk_manifest = ChunkManifest(entries={}, shape=dataset.shape)
        elif dataset.shape == ():
            chunk_manifest = ChunkManifest.from_arrays(
                paths=np.array(filepath, dtype=np.dtypes.StringDType),  # type: ignore
                offsets=np.array(dsid.get_offset(), dtype=np.uint64),
                lengths=np.array(dsid.get_storage_size(), dtype=np.uint64),
            )
        else:
            key_list = [0] * (len(dataset.shape) or 1)
            key = ".".join(map(str, key_list))

            chunk_entry: ChunkEntry = ChunkEntry.with_validation(  # type: ignore[attr-defined]
                path=filepath, offset=dsid.get_offset(), length=dsid.get_storage_size()
            )
            chunk_key = ChunkKey(key)
            chunk_entries = {chunk_key: chunk_entry}
            chunk_manifest = ChunkManifest(entries=chunk_entries)
    else:
        num_chunks = dsid.get_num_chunks()
        if num_chunks == 0:
            chunk_manifest = ChunkManifest(entries={}, shape=dataset.shape)
        else:
            grid_shape = tuple(math.ceil(a / b) for a, b in zip(dataset.shape, chunks))
            paths = np.empty(grid_shape, dtype=np.dtypes.StringDType)
            offsets = np.empty(grid_shape, dtype=np.uint64)
            lengths = np.empty(grid_shape, dtype=np.uint64)

            # When an oversized chunk has been trimmed the stored chunk holds
            # more bytes than the in-bounds region, so use the trimmed chunk's
            # byte size (valid because the trimmed region is a contiguous prefix
            # of an unfiltered chunk - see _chunk_shape) rather than blob.size.
            trimmed = chunks != dataset.chunks
            trimmed_length = math.prod(chunks) * dataset.dtype.itemsize

            def get_key(blob):
                return tuple(a // b for a, b in zip(blob.chunk_offset, chunks))

            def add_chunk_info(blob):
                key = get_key(blob)
                paths[key] = filepath
                offsets[key] = blob.byte_offset
                lengths[key] = trimmed_length if trimmed else blob.size

            has_chunk_iter = callable(getattr(dsid, "chunk_iter", None))
            if has_chunk_iter:
                dsid.chunk_iter(add_chunk_info)
            else:
                for index in range(num_chunks):
                    add_chunk_info(dsid.get_chunk_info(index))

            chunk_manifest = ChunkManifest.from_arrays(
                paths=paths,  # type: ignore
                offsets=offsets,
                lengths=lengths,
            )
    return chunk_manifest


def _dataset_dims(dataset: H5Dataset, group: str = "/") -> list[str]:
    """
    Get a list of dimension scale names attached to input HDF5 dataset.

    This is required by the xarray package to work with Zarr arrays. Only
    one dimension scale per dataset dimension is allowed. If dataset is
    dimension scale, it will be considered as the dimension to itself.

    Parameters
    ----------
    dataset
        An h5py dataset.
    group
        Name of the group we are pulling these dimensions from (default: the root
        group "/"). Required for removing subgroup prefixes.

    Returns
    -------
    list[str]
        List with HDF5 path names of dimension scales attached to input
        dataset.
    """
    import h5py

    dims: list[str] = []

    for n in range(len(dataset.shape)):
        if (num_scales := len(dataset.dims[n])) == 1:
            dims.append(str(dataset.dims[n][0].name))
        elif h5py.h5ds.is_scale(dataset.id):
            dims.append(str(dataset.name))
        elif num_scales > 1:
            raise ValueError(
                f"{dataset.name} has {num_scales} dimension scales attached to "
                f"dimension #{n}; require exactly 1"
            )
        elif num_scales == 0:
            # Some HDF5 files do not have dimension scales.
            # If this is the case, `num_scales` will be 0.
            # In this case, we mimic netCDF4 and assign phony dimension names.
            # See https://github.com/fsspec/kerchunk/issues/41
            dims.append(f"phony_dim_{n}")

    return [dim.removeprefix(group).removeprefix("/") for dim in dims]


def _extract_attrs(h5obj: H5Dataset | H5Group, references: ReferencePaths):
    """
    Extract attributes from an HDF5 group or dataset.

    Parameters
    ----------
    h5obj
        An h5py group or dataset.
    references
        Resolves object-reference attribute values to the path of the object they
        point to.
    """
    _HIDDEN_ATTRS = {
        "REFERENCE_LIST",
        "CLASS",
        "DIMENSION_LIST",
        "NAME",
        "_Netcdf4Dimid",
        "_Netcdf4Coordinates",
        "_nc3_strict",
        "_NCProperties",
    }
    attrs = {}
    for n, v in h5obj.attrs.items():
        if n in _HIDDEN_ATTRS:
            continue
        if n == "_FillValue":
            v = v
        if type(v) is h5py.Reference:
            v = references(h5obj.file, v)
        elif (
            isinstance(v, np.ndarray)
            and h5py.check_ref_dtype(v.dtype) is h5py.Reference
        ):
            v = np.array(
                [references(h5obj.file, ref) for ref in v.ravel()], dtype=str
            ).reshape(v.shape)
        # Fix some attribute values to avoid JSON encoding exceptions...
        if isinstance(v, bytes):
            v = v.decode("utf-8") or " "
        elif isinstance(v, (np.ndarray, np.number, np.bool_)):
            if v.dtype.kind == "S":
                v = v.astype(str)
            elif v.size == 1:
                v = v.flatten()[0]
                if isinstance(v, (np.ndarray, np.number, np.bool_)):
                    v = v.tolist()
            else:
                v = v.tolist()
        elif isinstance(v, h5py._hl.base.Empty):
            v = ""
        if v == "DIMENSION_SCALE":
            continue
        attrs[n] = v
    return attrs


def _find_non_coord_dimension_vars(group: H5Group) -> list[str]:
    import h5py

    dimension_names = []
    non_coordinate_dimension_variables = []
    for name, obj in group.items():
        if "_Netcdf4Dimid" in obj.attrs:
            dimension_names.append(name)
    for name, obj in group.items():
        if type(obj) is h5py.Dataset:
            if obj.id.get_storage_size() == 0 and name in dimension_names:
                non_coordinate_dimension_variables.append(name)

    return non_coordinate_dimension_variables
