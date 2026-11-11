"""Read bounded live JSON snapshots with explicit Windows sharing."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path


@contextmanager
def open_shared(path):
    path=Path(path)
    if os.name!='nt':
        with path.open('rb') as stream:yield stream
        return
    import ctypes
    from ctypes import wintypes
    import msvcrt
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,
        ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    kernel.CreateFileW.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.CloseHandle.restype=wintypes.BOOL
    # GENERIC_READ; share READ | WRITE | DELETE; OPEN_EXISTING; NORMAL attributes.
    handle=kernel.CreateFileW(str(path.resolve()),0x80000000,7,None,3,0x80,None)
    if handle in (None,wintypes.HANDLE(-1).value):raise ctypes.WinError(ctypes.get_last_error())
    try:descriptor=msvcrt.open_osfhandle(handle,os.O_RDONLY|os.O_BINARY|os.O_NOINHERIT)
    except BaseException:
        kernel.CloseHandle(handle);raise
    # The CRT descriptor owns the native handle and closes it with the stream.
    with os.fdopen(descriptor,'rb') as stream:yield stream


def read_shared_json(path,max_bytes=4*1024**2):
    if type(max_bytes) is not int or not 1<=max_bytes<=4*1024**2:
        raise ValueError('live JSON byte bound must be an integer up to 4 MiB')
    with open_shared(path) as stream:data=stream.read(max_bytes+1)
    if len(data)>max_bytes:raise MemoryError('live JSON exceeds its byte bound')
    value=json.loads(data)
    if not isinstance(value,dict):raise ValueError('live progress needs a JSON object')
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('paths',type=Path,nargs='+')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]/'outputs';failed=False
    for path in args.paths:
        try:
            path=path.resolve()
            if not path.is_relative_to(root):raise ValueError('live observation is restricted to project outputs')
            record=read_shared_json(path);history=record.get('stages') or record.get('samples') or []
            print(json.dumps(dict(path=str(path),completed=record.get('completed'),passed=record.get('passed'),
                steps_completed=record.get('steps_completed'),planned_steps=record.get('planned_steps'),
                pause_seconds=record.get('pause_seconds'),event_count=len(record.get('events',[])),
                last_observation=history[-1] if history else None)))
        except (OSError,ValueError,MemoryError) as error:
            failed=True;print(json.dumps(dict(path=str(path),observation_failed=True,error=f'{type(error).__name__}: {error}')))
    # An observation failure says nothing about whether a worker is still live.
    return 1 if failed else 0


if __name__=='__main__':raise SystemExit(main())
