"""Nobody reaches the startup record through another object for a resolved value.

The supplied-instance census counts three spellings, all of which start from a
`server_args` parameter -- the caller chose the object, which is the contract
that makes those reads defensible. This pins the fourth: `model_runner.
server_args.field`, `self.scheduler.server_args.field`, `tokenizer_manager.
server_args.field`. A reference lifted off whatever object happened to hold the
record carries no contract at all, and it was invisible to every census, which
is how it grew to ninety-five reads across thirty-five files unnoticed.

They are gone, and this is what keeps them gone. Only fields resolution writes
are pinned: reading `model_runner.server_args.host` off the record answers with
what the caller asked for, which is what the record is for. The written set is
derived from the declaration sites rather than listed, so a field that stops
being resolution-written drops out on its own -- and a field that stops being
*declared* cannot slip out that way, because bare assignment during resolution
is refused by `server_args/test_resolution_declarations.py`.
"""

import ast
import pathlib
import unittest

import sglang
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_SRT = pathlib.Path(sglang.__file__).resolve().parent / "srt"

# The pipeline and its extension points: reading the in-flight record is their
# job, and they run before anything is published.
_OWNERS = ("server_args.py", "runtime_context.py", "arg_groups/")

_DECLARERS = ("_declare", "declare_resolution", "declare_late_resolution")


def _declared_by_keyword():
    """Fields named as a keyword at a declaration site."""
    written = set()
    for path in sorted(_SRT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except SyntaxError:
            raise AssertionError(f"unparsable module in the census: {path}")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute):
                name = node.func.attr
            elif isinstance(node.func, ast.Name):
                name = node.func.id
            else:
                continue
            if name in _DECLARERS:
                written |= {kw.arg for kw in node.keywords if kw.arg}
    return written


def _declared_by_registry():
    """Fields the model-override registry writes.

    `MODEL_OVERRIDES` maps arch -> {field: value} and the provider functions
    build the same shape, so these field names are *data*, not keywords -- a
    keyword scan misses every one of them (`dtype`, the hybrid-SWA switch, the
    multi-layer EAGLE flag). They are resolution writes all the same, and a
    borrowed-record read of one answers with the CLI default just as loudly.
    """
    tree = ast.parse((_SRT / "arg_groups/overrides.py").read_text(encoding="utf-8-sig"))
    fields = set()
    for node in tree.body:
        target = None
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
        if target != "MODEL_OVERRIDES" or node.value is None:
            continue
        for inner in ast.walk(node.value):
            if not isinstance(inner, ast.Dict):
                continue
            for key, value in zip(inner.keys, inner.values):
                if isinstance(value, ast.Dict):
                    continue  # the arch -> {…} outer layer
                if not isinstance(key, ast.Constant):
                    raise AssertionError("non-literal override key")
                fields.add(key.value)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not any(
            isinstance(dec, ast.Call)
            and isinstance(dec.func, ast.Name)
            and dec.func.id.startswith("register_model_override")
            for dec in node.decorator_list
        ):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Assign) and isinstance(
                inner.targets[0], ast.Subscript
            ):
                key = inner.targets[0].slice
                if not isinstance(key, ast.Constant):
                    raise AssertionError(f"non-literal override key in {node.name}")
                fields.add(key.value)
            if isinstance(inner, ast.Dict):
                for key in inner.keys:
                    if not isinstance(key, ast.Constant):
                        raise AssertionError(f"non-literal override key in {node.name}")
                    fields.add(key.value)
    return fields


def _resolution_written():
    """Every field resolution can write, by either mechanism."""
    return _declared_by_keyword() | _declared_by_registry()


def _chain_reads(written):
    """`<expression>.server_args.<field>` where the field is resolution-written."""
    found = []
    for path in sorted(_SRT.rglob("*.py")):
        rel = path.relative_to(_SRT).as_posix()
        if path.name in _OWNERS or rel.startswith(_OWNERS[-1]):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except SyntaxError:
            raise AssertionError(f"unparsable module in the census: {rel}")
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Attribute)
                and isinstance(node.ctx, ast.Load)
                and node.attr in written
            ):
                continue
            base = node.value
            # `self.server_args.field` is the parked spelling the
            # supplied-instance census already counts; this is about a record
            # reached through some *other* object.
            if not (
                isinstance(base, ast.Attribute)
                and base.attr == "server_args"
                and not (isinstance(base.value, ast.Name) and base.value.id == "self")
            ):
                continue
            found.append(f"{rel}:{node.lineno} {ast.unparse(base)}.{node.attr}")
    return sorted(found)


class TestNoChainReadsOfResolvedConfig(CustomTestCase):
    def test_the_census_has_something_to_count(self):
        """A written set that collapsed would make the pin vacuous.

        Both mechanisms are checked separately: the registry's field names are
        data rather than keywords, so a keyword-only scan would look healthy
        while missing every override the registry applies.
        """
        by_keyword = _declared_by_keyword()
        by_registry = _declared_by_registry()
        self.assertGreater(
            len(by_keyword),
            100,
            f"only {len(by_keyword)} fields are declared by keyword; the scan broke",
        )
        # The registry's surface is small enough to name, and it should say
        # something when it grows: a new override needs its borrowed-record
        # readers checked, which is the whole point of scanning it here.
        self.assertEqual(
            sorted(by_registry),
            [
                "attention_backend",
                "disable_hybrid_swa_memory",
                "dtype",
                "enable_multi_layer_eagle",
                "swa_full_tokens_ratio",
            ],
            "the model-override registry's written fields changed; check the "
            "borrowed-record readers of any new one",
        )

    def test_nothing_reads_a_resolved_field_off_a_borrowed_record(self):
        found = _chain_reads(_resolution_written())
        self.assertEqual(
            found,
            [],
            "these reach the startup record through another object for a value "
            "resolution decides, so they answer with the CLI default once the "
            "record stays raw; read the config bag instead:\n  " + "\n  ".join(found),
        )


if __name__ == "__main__":
    unittest.main()
