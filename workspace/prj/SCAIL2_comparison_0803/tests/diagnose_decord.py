import importlib.metadata as metadata
import sys

print("python", sys.version)
distribution = metadata.distribution("decord")
print("decord_version", distribution.version)
print("location", distribution.locate_file(""))
for file in distribution.files or []:
    if str(file).endswith(".dist-info/WHEEL"):
        print("wheel_metadata")
        print(distribution.locate_file(file).read_text())

try:
    import decord
except Exception as error:
    print("import_failed", type(error).__name__, repr(error))
    raise
else:
    print("import_ok", decord.__version__, decord.__file__)
