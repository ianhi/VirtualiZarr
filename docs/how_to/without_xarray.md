# Using VirtualiZarr without xarray

A parser returns a [ManifestStore][virtualizarr.manifests.ManifestStore], which you can write to Icechunk directly with [ManifestStore.to_icechunk][virtualizarr.manifests.ManifestStore.to_icechunk], without converting it to a virtual dataset first.
Every array and group is written with exactly the dimension names, attributes, codecs and fill values the parser produced.

This is how to write files whose structure is valid Zarr but doesn't fit [xarray's data model](../explanation/data_structures.md#how-the-zarr-and-xarray-data-models-differ), such as:

- arrays with no dimension names;
- sibling arrays that share a dimension name at different lengths, like the levels of a multiscale image pyramid;
- a subgroup that reuses one of its parent's dimension names at a different length.

The examples on this page use one file from the public NEX-GDDP-CMIP6 dataset on S3, and an in-memory Icechunk repository that can read from that bucket, as in the [usage guide](usage.md):

```python exec="on" session="without_xarray" source="material-block"
import icechunk
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import from_url

from virtualizarr.parsers import HDFParser

bucket = "s3://nex-gddp-cmip6"
path = "NEX-GDDP-CMIP6/ACCESS-CM2/ssp126/r1i1p1f1/tasmax/tasmax_day_ACCESS-CM2_ssp126_r1i1p1f1_gn_2015_v2.0.nc"
url = f"{bucket}/{path}"
store = from_url(bucket, region="us-west-2", skip_signature=True)
registry = ObjectStoreRegistry({bucket: store})

config = icechunk.RepositoryConfig.default()
config.set_virtual_chunk_container(
    icechunk.VirtualChunkContainer(
        url_prefix="s3://nex-gddp-cmip6/",
        store=icechunk.s3_store(region="us-west-2", anonymous=True),
    ),
)
repo = icechunk.Repository.create(icechunk.in_memory_storage(), config)
```

## Writing to Icechunk

```python exec="on" session="without_xarray" source="material-block" result="code"
manifest_store = HDFParser()(url=url, registry=registry)

session = repo.writable_session("main")
manifest_store.to_icechunk(session.store, group="without_xarray")
snapshot_id = session.commit("Wrote the file without going via xarray")
print(snapshot_id)
```

## Loading arrays

Loading an array copies its data into Icechunk, so reading it no longer touches the archival file.
This is helpful for small arrays that are read often, such as coordinates, especially when they are split into one chunk per archival file and would be better stored as a few larger chunks.
Chunks the parser inlined, such as small chunks from Kerchunk references, are already written to Icechunk as native chunks, so reading them never touches the archival file.

To load an array, copy it out of the `ManifestStore` with [zarr.from_array][], passing `overwrite=True` to replace the virtual array.
The copy has the same chunks, codecs and dimension names as the source:

```python exec="on" session="without_xarray" source="material-block" result="code"
import zarr

session = repo.writable_session("main")

lat_source = zarr.open_array(manifest_store, path="lat", mode="r", zarr_format=3)
lat = zarr.from_array(
    session.store, name="without_xarray/lat", data=lat_source, overwrite=True
)
print("source:", lat_source.chunks, lat_source.compressors)
print("copy:  ", lat.chunks, lat.compressors)
```

To update the chunking, codecs or any other metadata, pass new values to `from_array`, such as `chunks=` or `compressors=`.
This copy of `lon` splits the file's single chunk of 1440 values into four chunks of 360, and uses Zstandard instead of the file's shuffle and zlib:

```python exec="on" session="without_xarray" source="material-block" result="code"
lon_source = zarr.open_array(manifest_store, path="lon", mode="r", zarr_format=3)
lon = zarr.from_array(
    session.store,
    name="without_xarray/lon",
    data=lon_source,
    chunks=(360,),
    compressors=zarr.codecs.ZstdCodec(),
    overwrite=True,
)
print("source:", lon_source.chunks, lon_source.compressors)
print("copy:  ", lon.chunks, lon.compressors)

snapshot_id = session.commit("Loaded lat and lon")
```

Merging chunks works the same way: to load a coordinate stored as one chunk per file into a single chunk, pass its full length as `chunks=`.

!!! important
    `zarr.from_array` copies the fill value and attributes only from zarr 3.4 onwards.
    With older versions, pass them yourself, as in `fill_value=lat_source.fill_value, attributes=lat_source.attrs.asdict()`.
