# Decord wheel metadata repair

The official SCAIL-2 requirement `decord>=0.6.0` resolved to the PyPI wheel
`decord-0.6.0-py3-none-manylinux2010_x86_64.whl`.

The wheel filename claims the generic `py3-none` tag, but its internal
`decord-0.6.0.dist-info/WHEEL` metadata incorrectly contains:

```text
Tag: cp36-cp36m-manylinux2010_x86_64
```

The original wheel also fails `wheel unpack` because one original RECORD hash
does not match. Despite these packaging defects, the installed native library
imports successfully under Python 3.11.15.

`scripts/repair_decord_wheel.py` extracts the original PyPI wheel, changes only
the internal compatibility tag to match its published filename, and repacks it
with `wheel pack`, which regenerates RECORD. No package source or shared-library
payload is changed.

Repaired wheel:

```text
decord-0.6.0-1-py3-none-manylinux2010_x86_64.whl
SHA256 3ba940c9cd6cd84e99f81638a264333b8e17f7984a6652c698b0ccf020e49609
```

After installing the repaired wheel, `import decord` succeeds and
`python -m pip check` reports `No broken requirements found.`
