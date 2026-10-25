#!/usr/bin/env python3
import importlib.util, pathlib, sys
CODE=pathlib.Path('/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_20261011/code')
sys.path.insert(0,str(CODE))
source=CODE/'auto_semantic_batch_full.py'
spec=importlib.util.spec_from_file_location('auto_semantic_test_0132',source)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.INV=pathlib.Path('/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_20261011/plans/recovery_test_0132.json')
sys.argv=[str(source),'--workers','1','--offset','0','--limit','1','--force']
mod.main()