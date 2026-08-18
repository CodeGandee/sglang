"""`ModelConfig` is built from values resolution has already decided.

Resolution builds a `ModelConfig` partway through and keys later decisions off
it, so the pipeline reads its own output through that object. The loop is only
benign while every field `ModelConfig.from_server_args` reads has been resolved
by the time it is built -- otherwise the model configuration describes a
half-resolved input, and every handler downstream of it inherits that.

Nothing enforces the ordering today; it holds because the path and quantization
handlers happen to run early. So this derives both sides from the source -- the
fields the constructor reads, and the step each is declared at -- and pins the
one field that is deliberately read before resolution touches it.
"""

import ast
import pathlib
import unittest

import sglang
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_SRT = pathlib.Path(sglang.__file__).resolve().parent / "srt"

# `is_embedding` is read for what the *caller asked for*: the constructor passes
# it as `is_embedding_requested` and to `is_generation_model`, and never stores
# it. Resolution later overwrites the field with the value the architecture
# implies -- a different quantity sharing one name -- so the constructor wanting
# the earlier one is the point, not an ordering bug.
_READ_BEFORE_RESOLUTION = frozenset({"is_embedding"})


def _server_args_names(tree, path):
    names = {"self"} if path.name == "server_args.py" else {"server_args"}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        for arg in args.posonlyargs + args.args + args.kwonlyargs:
            annotation = arg.annotation
            if isinstance(annotation, ast.Constant):
                text = annotation.value
            elif isinstance(annotation, ast.Name):
                text = annotation.id
            elif isinstance(annotation, ast.Attribute):
                text = annotation.attr
            else:
                continue
            if text == "ServerArgs":
                names.add(arg.arg)
    return names


def _constructor_reads():
    """Fields `ModelConfig.from_server_args` takes off the record."""
    path = _SRT / "configs/model_config.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    constructor = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "from_server_args"
    )
    names = _server_args_names(tree, path)
    return {
        node.attr
        for node in ast.walk(constructor)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in names
        and isinstance(node.ctx, ast.Load)
    }


def _pipeline():
    """(ordered steps, {step: methods it reaches}) for the resolution dispatch."""
    source = (_SRT / "server_args.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    record = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ServerArgs"
    )
    methods = {
        node.name: node for node in record.body if isinstance(node, ast.FunctionDef)
    }
    dispatch = methods["_run_resolution_pipeline"]
    steps = [
        name
        for _line, name in sorted(
            (node.lineno, node.func.attr)
            for node in ast.walk(dispatch)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        )
    ]

    def reaches(name, seen=None):
        seen = seen if seen is not None else set()
        if name in seen or name not in methods:
            return seen
        seen.add(name)
        for node in ast.walk(methods[name]):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
                and node.func.attr in methods
            ):
                reaches(node.func.attr, seen)
        return seen

    return steps, methods, {name: reaches(name) for name in steps}


class TestModelConfigReadsResolvedInput(CustomTestCase):
    def test_every_field_it_reads_is_resolved_before_it_is_built(self):
        steps, methods, reached = _pipeline()
        wanted = _constructor_reads()

        first_build = None
        declared_at = {}
        for index, step in enumerate(steps):
            for method in reached[step]:
                body = methods[method]
                for node in ast.walk(body):
                    if not isinstance(node, ast.Call):
                        continue
                    if (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr == "get_model_config"
                        and first_build is None
                    ):
                        first_build = (index, step)
                    if (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr == "_declare"
                    ):
                        for keyword in node.keywords:
                            if keyword.arg in wanted:
                                # The *last* declaration is the one that has to
                                # precede the build: a later one moves the value
                                # after the model configuration was derived from
                                # it, which is exactly the staleness at issue.
                                declared_at[keyword.arg] = max(
                                    declared_at.get(keyword.arg, index), index
                                )
        self.assertIsNotNone(
            first_build, "no handler builds a ModelConfig; the scan broke"
        )

        late = sorted(
            field
            for field, index in declared_at.items()
            if index >= first_build[0] and field not in _READ_BEFORE_RESOLUTION
        )
        self.assertEqual(
            late,
            [],
            "resolution decides these after it builds the ModelConfig that reads "
            f"them, so the model configuration describes a half-resolved input "
            f"(first build: step {first_build[0]}, {first_build[1]}): {late}",
        )

    def test_the_documented_exception_is_still_the_only_one(self):
        """A field pinned as read-before-resolution has to still be both."""
        steps, methods, reached = _pipeline()
        wanted = _constructor_reads()
        declared = set()
        for step in steps:
            for method in reached[step]:
                for node in ast.walk(methods[method]):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "_declare"
                    ):
                        declared |= {kw.arg for kw in node.keywords if kw.arg}
        stale = sorted(
            field
            for field in _READ_BEFORE_RESOLUTION
            if field not in wanted or field not in declared
        )
        self.assertEqual(
            stale,
            [],
            "these are pinned as read-before-resolution but are no longer both "
            f"read by the constructor and written by resolution: {stale}",
        )


if __name__ == "__main__":
    unittest.main()
