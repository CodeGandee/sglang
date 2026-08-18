"""Resolution does not read the config bags, because they do not exist yet.

The bags are projected from what resolution decides, so anything the pipeline
calls has to read the resolving state instead — `resolved_view(server_args)`,
or the view a handler already holds. A bag read reached from resolution raises
`config namespace ... not published`, and only on the branch that reaches it:
the diffusion-LM page-size pass needed one model family, the Marlin LoRA
validation needed one MoE runner backend. Both were written, merged into a
branch, and stayed green for everything except the configuration that triggers
them.

`test_publish_precedes_bag_reads.py` is the same worry from the other side, but
it walks the *process entries* — it cannot see a helper the pipeline calls, and
neither of the two above appeared in it.

What this cannot see: a call through a value (a callable parameter, an
attribute off something other than a module), and a bag read behind an import
this does not follow. The walk goes through direct calls to module-level
functions in modules the pipeline imports, one hop out, which is what the two
known cases looked like.
"""

import ast
import pathlib
import unittest

import sglang
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_SRT = pathlib.Path(sglang.__file__).resolve().parent / "srt"

_BAG_ACCESSORS = frozenset(
    {
        "get_exec",
        "get_memory",
        "get_schedule",
        "get_device",
        "get_model",
        "get_spec",
        "get_lora",
        "get_mm",
        "get_disagg",
        "get_serving",
        "get_observability",
        "get_parallel",
        "get_server_args",
        "configured_tp_size",
        "configured_pp_size",
        "configured_moe_dp_size",
        "configured_attn_cp_size",
        "configured_dcp_size",
    }
)

# The pipeline itself and the mechanism it publishes through: `runtime_context`
# defines the accessors, and `arg_groups` is the pipeline's own code.
_OWN = ("server_args.py", "runtime_context.py")


def _module_of(name):
    """`sglang.srt.a.b` -> the file, if it is one of ours."""
    if not name or not name.startswith("sglang.srt."):
        return None
    rel = name[len("sglang.srt.") :].replace(".", "/")
    for candidate in (_SRT / f"{rel}.py", _SRT / rel / "__init__.py"):
        if candidate.exists():
            return candidate
    return None


def _imported_symbols(paths):
    """{module file: {symbol names imported from it}} across the given sources."""
    out = {}
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if not isinstance(node, ast.ImportFrom):
                continue
            target = _module_of(node.module)
            if target is None or target.name in _OWN:
                continue
            out.setdefault(target, set()).update(alias.name for alias in node.names)
    return out


def _reaches_a_bag(path, entry):
    """Does `entry` in `path` reach a bag accessor, following calls in-module?"""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    seen = set()

    def walk(name):
        if name in seen or name not in functions:
            return None
        seen.add(name)
        for node in ast.walk(functions[name]):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            if node.func.id in _BAG_ACCESSORS:
                return node.lineno
            found = walk(node.func.id)
            if found is not None:
                return found
        return None

    return walk(entry)


class TestResolutionReadsNoBag(CustomTestCase):
    def test_the_walk_finds_something_to_walk(self):
        """A collapsed import map would make the pin vacuous."""
        imported = _imported_symbols(
            [_SRT / "server_args.py", _SRT / "arg_groups" / "overrides.py"]
        )
        self.assertGreater(
            len(imported),
            20,
            f"the pipeline only imports from {len(imported)} of our modules; "
            "the scan broke",
        )

    def test_nothing_the_pipeline_calls_reads_a_bag(self):
        imported = _imported_symbols(
            [_SRT / "server_args.py", _SRT / "arg_groups" / "overrides.py"]
        )
        found = []
        for path, symbols in sorted(imported.items()):
            for symbol in sorted(symbols):
                line = _reaches_a_bag(path, symbol)
                if line is not None:
                    found.append(
                        f"{path.relative_to(_SRT)}:{line} reached from "
                        f"{symbol}(), which the resolution pipeline imports"
                    )
        self.assertEqual(
            found,
            [],
            "resolution reaches a config-bag read, which raises on whichever "
            "branch gets there first; read the resolving state instead:\n  "
            + "\n  ".join(found),
        )


if __name__ == "__main__":
    unittest.main()
