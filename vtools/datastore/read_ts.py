import warnings

all = ["read_ts"]

def read_ts(fpath, start=None, end=None, force_regular=True,nrows=None, selector = None,hint=None):
    warnings.warn("Moved to the dms_datastore package." , PendingDeprecationWarning)
    import dms_datastore.read_ts
    ts = dms_datastore.read_ts.read_ts(fpath=fpath,start=start,end=end,force_regular=force_regular,nrows=nrows,selector=selector,hint=hint)

    return ts
