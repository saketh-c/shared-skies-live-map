"""Stream one AQS hourly zip and keep only Texas + PurpleAir-bbox rows.

usage: python -I filter_aqs.py <in.zip> <out.parquet>
"""
import sys
import zipfile

import pandas as pd

LAT_MIN, LAT_MAX = 25.84, 36.5
LON_MIN, LON_MAX = -106.65, -93.5

USECOLS = ["State Code", "County Code", "Site Num", "Parameter Code", "POC",
           "Latitude", "Longitude", "Date Local", "Time Local", "Date GMT", "Time GMT",
           "Sample Measurement", "Qualifier", "Method Type", "Method Code", "State Name"]


def main(src, dst):
    zf = zipfile.ZipFile(src)
    name = zf.namelist()[0]
    keep = []
    n_in = 0
    with zf.open(name) as fh:
        for ch in pd.read_csv(fh, usecols=USECOLS, dtype={
                "State Code": str, "County Code": str, "Site Num": str,
                "Parameter Code": str, "Qualifier": str, "Method Code": str},
                chunksize=2_000_000, low_memory=False):
            n_in += len(ch)
            m = (ch["State Name"] == "Texas") | (
                ch["Latitude"].between(LAT_MIN, LAT_MAX)
                & ch["Longitude"].between(LON_MIN, LON_MAX))
            if m.any():
                keep.append(ch[m])
    out = pd.concat(keep, ignore_index=True)
    out.to_parquet(dst, index=False)
    print(f"{src}: {n_in:,} rows in, {len(out):,} kept, "
          f"{out.groupby(['State Code','County Code','Site Num']).ngroups} sites")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
