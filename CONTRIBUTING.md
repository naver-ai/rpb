# Contributing

Add the following comment to the top of every new original Python, shell,
YAML, or TOML file. Keep an executable file's shebang on the first line and
place the notice immediately after it.

```text
# Router Prior Bias (RPB)
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0
```

For adapted code, retain the original copyright and license comments below
the NAVER notice. Record the source revision, adapted portions, and applicable
license in `THIRD_PARTY_NOTICES.md`. Files copied entirely from upstream and
automatically generated templates retain their original notices; document
reviewed exceptions in `scripts/check_release_content.py`.

Edit `patches/source/rpb.py` and regenerate the distributable patch with
`uv run python scripts/sync_patch.py`. Do not edit the generated patch directly.

Before publication, run `bash scripts/validate_release.sh` and review all files
for confidential data. The automated scan flags common private paths,
hostnames, cluster identifiers, and credentials; it cannot establish that
every kind of confidential information is absent.

Publish from a repository whose history starts with the fully attributed
release. Do not import earlier development commits, branches, tags, or backups.
