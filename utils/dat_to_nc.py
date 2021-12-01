## dat to netcdf, based on https://gis.stackexchange.com/a/414698/156215

from pathlib import Path
import pandas as pd
import xarray as xr

in_file = Path(
    r"C:/Users/Luke/Downloads/133023_11_0/2019_National_Gravity_Grids_Ground_Gravity_PLD/2019_National_Gravity_Grids_Ground_Gravity_PLD.dat"
)

dat_file = in_file.with_suffix(".dat")
ddf_file = in_file.with_suffix(".ddf")
csv_file = in_file.with_suffix(".csv")
nc_file = in_file.with_suffix(".nc")

column_headers = []
column_specs = []

with open(ddf_file, "r") as ddf:
    next(ddf)  # skip first line
    for row in ddf:
        name, cols, *_ = row.split(" ")  # get column name and column specs
        column_headers.append(name)
        f, t = [int(c) - 1 for c in cols.split("-")]
        column_specs.append((f, t))

pd_dset = pd.read_fwf(dat_file, colspecs=column_specs)
pd_dset.columns = column_headers

xr_dset = xr.Dataset(pd_dset)
xr_dset.to_netcdf(path=nc_file, mode="w", encoding={"compression": "gzip", "compression_opts": 9})

print(f"Wrote out {nc_file}")

# first_chunk = next(reader)
# first_chunk.columns = column_headers
# first_chunk.to_csv(csv_file)

# for chunk in reader:
#     chunk.to_csv(csv_file, mode="a", header=False)
