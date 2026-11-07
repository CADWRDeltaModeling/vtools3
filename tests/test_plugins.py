"""Discovery tests use synthetic metadata, not installed provider packages."""

from importlib.metadata import EntryPoint, EntryPoints

import pytest

from vtools.plugins import load_plugins


def test_loads_only_requested_group(monkeypatch):
    entries = EntryPoints([
        EntryPoint("absolute", "builtins:abs", "timeseries.transforms"),
        EntryPoint("reader", "builtins:str", "timeseries.readers"),
    ])
    monkeypatch.setattr("vtools.plugins.entry_points", lambda: entries)
    assert load_plugins("timeseries.transforms") == {"absolute": abs}
    assert load_plugins("missing") == {}


def test_rejects_duplicate_names_unless_explicitly_overridden(monkeypatch):
    entries = EntryPoints([
        EntryPoint("operation", "builtins:abs", "timeseries.transforms"),
        EntryPoint("operation", "builtins:round", "timeseries.transforms"),
    ])
    monkeypatch.setattr("vtools.plugins.entry_points", lambda: entries)
    with pytest.raises(ValueError, match="Duplicate.*operation"):
        load_plugins("timeseries.transforms")
    assert load_plugins("timeseries.transforms", overrides={"operation": str}) == {
        "operation": str
    }


def test_reports_loading_failure_and_preserves_cause(monkeypatch):
    entries = EntryPoints([
        EntryPoint("broken", "missing_plugin_package:operation", "timeseries.transforms")
    ])
    monkeypatch.setattr("vtools.plugins.entry_points", lambda: entries)
    with pytest.raises(RuntimeError, match="timeseries.transforms:broken") as failure:
        load_plugins("timeseries.transforms")
    assert isinstance(failure.value.__cause__, ModuleNotFoundError)
    assert load_plugins("timeseries.transforms", overrides={"broken": abs}) == {
        "broken": abs
    }


def test_rejects_noncallable_entries_and_overrides(monkeypatch):
    entries = EntryPoints([
        EntryPoint("constant", "math:pi", "timeseries.transforms")
    ])
    monkeypatch.setattr("vtools.plugins.entry_points", lambda: entries)
    with pytest.raises(TypeError, match="constant.*not callable"):
        load_plugins("timeseries.transforms")
    with pytest.raises(TypeError, match="local.*not callable"):
        load_plugins("timeseries.transforms", overrides={"local": 1})


def test_returns_independent_dictionaries(monkeypatch):
    monkeypatch.setattr("vtools.plugins.entry_points", lambda: EntryPoints([]))
    first = load_plugins("timeseries.transforms", overrides={"local": abs})
    first.clear()
    assert load_plugins("timeseries.transforms", overrides={"local": abs}) == {
        "local": abs
    }


def test_supports_python39_metadata_mapping(monkeypatch):
    entries = {"timeseries.transforms": [
        EntryPoint("absolute", "builtins:abs", "timeseries.transforms")
    ]}
    monkeypatch.setattr("vtools.plugins.entry_points", lambda: entries)
    assert load_plugins("timeseries.transforms") == {"absolute": abs}