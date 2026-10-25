# Large-file upload status

The bulk Release upload is running on the source server. The JSON status is a point-in-time snapshot; it is not a live website. Only entries in `release_receipts/` with `state=uploaded_verified` are confirmed uploaded. The original full transfer plan remains pending overall; `payload_safety/` records checks completed per job. The source workspace is unchanged. Credentials never appear in these files.

Download Release assets and use `tools/restore_release.py` with the original transfer plan and verified receipts to recover source-relative paths. A `.bin` asset may be one raw range of an original video or model; do not rename and treat every range as a full file.
