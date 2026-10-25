import json

from modelscope.hub.api import HubApi


api = HubApi()
files = api.get_model_files(
    model_id="ZhipuAI/SCAIL-2",
    revision="master",
    recursive=True,
)
print(json.dumps(files, ensure_ascii=False, indent=2))
