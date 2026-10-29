# vtools agent guidance

`vtools` is the foundational general-purpose time-series processing package in this ecosystem.

It should remain independent of SCHISM, BayDeltaSCHISM, and repository-specific workflow policy.

## Package role

`vtools` owns reusable time-series algorithms and conventions.

Downstream projects should call these implementations rather than recreating equivalent operations.

Important established functionality includes:

- prioritized time-series merging and tiling
- tidal filtering
- interval comparison and division
- conservative interpolation and resampling

Keep algorithms general and reusable.

## Dependency policy

Keep `vtools` foundational.

Do not introduce:

- SCHISM dependencies
- BayDeltaSCHISM assumptions
- time-series repository policy belonging to `dms_datastore`
- unnecessary spatial dependencies
- unnecessary plotting dependencies
- indirect dependencies on HEC-DSS

Assume the established scientific Python/tool stack is available.

## Coding practice

- Plan before coding.
- Keep functions single-purpose.
- Keep functions testable.
- Do not refactor outside the scope of the requested work. Alert the user if broader refactoring appears warranted.
- Do not contract existing documentation.
- Preserve NumPy-style documentation; repair it when interfaces change.
- Prefer explicit errors such as `ValueError` over elaborate recovery from invalid arguments.
- Avoid making inference from surrounding files the only way to use an API. Inference may be a convenience, but important inputs should also be supplyable explicitly.
- Assume the established tool stack exists; do not add elaborate defensive discovery for expected dependencies.
- Preserve the mathematical and metadata semantics of established algorithms rather than tailoring them to one downstream application.

## Time-series conventions

Use lower-case frequency codes such as:

`min`, `h`, `d`, `s`

For prioritized merging or tiling, use the established merge functionality. Understand the `names` argument rather than reimplementing merge logic.

For tidal filtering, use the established cosine-Lanczos implementation, for example:

```python
from vtools import cosine_lanczos
tsave = cosine_lanczos(ts, "40h")
```

For interval comparison and division, use the established interval helper functions.

For interval creation, prefer helpers such as `days(1)`, `hours(3)`, or `pd.to_timedelta(...)` rather than hard-wiring interval implementation details.

For interpolating daily or monthly averages, use the established histogram-conserving functionality. Use its lower-bound and shape controls when positivity preservation is required.

Do not add SCHISM- or repository-specific conventions to generic algorithms.

## Testing

- Use `pytest`.
- Mark network-dependent tests as `integration`.
- Normal CI should exclude integration tests where appropriate.
