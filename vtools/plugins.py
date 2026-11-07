"""Discover ordinary callables advertised by installed packages."""

from importlib.metadata import entry_points


def load_plugins(group, *, overrides=None):
    """Load named callables from an entry-point group.

    Parameters
    ----------
    group : str
        Entry-point group, such as ``timeseries.transforms`` or
        ``timeseries.readers``.
    overrides : mapping of str to callable, optional
        Explicit local implementations. These replace advertised entries
        with the same name without loading those entries.

    Returns
    -------
    dict of str to callable
        A new dictionary owned by the caller. Discovery is not cached.

    Raises
    ------
    ValueError
        Multiple providers advertise the same name without an override.
    TypeError
        An implementation is not callable.
    RuntimeError
        An advertised implementation cannot be imported.
    """
    overrides = dict(overrides or {})
    for name, implementation in overrides.items():
        if not callable(implementation):
            raise TypeError(f"Plug-in {group}:{name} is not callable")
    advertised = entry_points()
    if hasattr(advertised, "select"):
        advertised = advertised.select(group=group)
    else:
        advertised = advertised.get(group, ())
    plugins = {}
    providers = {}
    for entry in advertised:
        if entry.name in overrides:
            continue
        if entry.name in plugins:
            raise ValueError(
                f"Duplicate plug-in {group}:{entry.name}: "
                f"{providers[entry.name]} and {entry.value}; supply an explicit override"
            )
        try:
            implementation = entry.load()
        except Exception as exc:
            raise RuntimeError(
                f"Cannot load plug-in {group}:{entry.name} from {entry.value}"
            ) from exc
        if not callable(implementation):
            raise TypeError(f"Plug-in {group}:{entry.name} is not callable")
        plugins[entry.name] = implementation
        providers[entry.name] = entry.value
    plugins.update(overrides)
    return plugins