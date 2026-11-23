Place the input daily multivariate CSV here (e.g. `data/ERA.csv`).

The archived merged CSV was **not included** in the public-code package because
its underlying ERA5 / Copernicus Marine redistribution terms and scientific
provenance need to be checked before public release. Obtain the source data
under the applicable provider licenses and cite the providers.

Required columns include `date`, `chl`, and the seven primary drivers listed in
the repository README. Daily intervals must be checked. The original attached
`ERA.csv` has missing records for 1999-12-31 and 2024-12-31, and 2635 missing
entries in `cloud_base_height`. The code marks missing days as NaN and excludes
incomplete sample windows rather than silently interpolating them.
