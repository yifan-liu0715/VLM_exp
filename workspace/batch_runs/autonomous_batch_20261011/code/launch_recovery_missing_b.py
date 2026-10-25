#!/usr/bin/env python3
import importlib.util
import pathlib
import sys
CODE=pathlib.Path('/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_20261011/code')
sys.path.insert(0,str(CODE))
source=CODE/'auto_semantic_batch_full.py'
spec=importlib.util.spec_from_file_location('auto_semantic_recovery',source)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.INV=pathlib.Path('/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_20261011/plans/recovery_missing_b.json')
sys.argv=[str(source),'--workers','2','--offset','0','--limit','7']
mod.main()