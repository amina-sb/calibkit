"""The README's quick-start block must run as written."""

import re
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_python_blocks_execute() -> None:
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(), flags=re.S)
    assert blocks, "README has no python example"
    for block in blocks:
        exec(compile(block, str(README), "exec"), {})
