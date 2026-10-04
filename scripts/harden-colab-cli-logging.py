#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import glob

OLD = """    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    requests_log = logging.getLogger("urllib3")
    requests_log.setLevel(logging.DEBUG)
    requests_log.propagate = True
"""

NEW = """    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    requests_log = logging.getLogger("urllib3")
    requests_log.setLevel(logging.WARNING)
    requests_log.propagate = True
"""

paths = [Path(p) for p in glob.glob("/opt/colab-cli/lib/python*/site-packages/colab_cli/common.py")]
if len(paths) != 1:
    raise SystemExit(f"Expected one colab_cli/common.py, found {len(paths)}")

path = paths[0]
text = path.read_text()
if NEW in text:
    print(f"already hardened: {path}")
elif OLD in text:
    path.write_text(text.replace(OLD, NEW, 1))
    print(f"hardened: {path}")
else:
    raise SystemExit("Colab CLI logging implementation changed; inspect before patching")
