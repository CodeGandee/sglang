"""Resolution writes are recorded, not just applied.

The projection that replaces field materialization reads the declaration stash,
so a resolution write that only assigns the field is invisible to it. Every
resolver declares now -- the record's handlers through `self._declare`, the
hooks and hardware defaults through `declare_resolution` -- and that is pinned
two ways: no bare assignment to a field survives anywhere a ServerArgs instance
is in reach, and after resolution every declared field agrees with what the
stash says. The second check is the one that keeps the transition honest --
while a declaration still writes the field immediately, a stash entry and a
field can only disagree if something assigned the field behind the stash's
back. A third check runs the other way: every field resolution moved has to
be explained by the stash, which covers the spellings a source scan cannot
see.
"""

import ast
import dataclasses
import json
import os
import pathlib
import shutil
import tempfile
import unittest

import sglang
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_SRT = pathlib.Path(sglang.__file__).resolve().parent / "srt"

# Every field of the record. Resolution has no bare-assignment writer left, so
# the scan below states that as a whole rather than tracking a list of the
# fields converted so far: a new resolution write that assigns instead of
# declaring is a failure the day it is written.
_RESOLVED_FIELDS = frozenset(field.name for field in dataclasses.fields(ServerArgs))

# Resolution shapes the agreement check runs on. Each needs a real
# config.json: `model_path="dummy"` takes the pipeline's early return, so most
# handlers never run and the agreement check would pass over a nearly empty
# stash.
_MINI_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "vocab_size": 128,
    "max_position_embeddings": 2048,
}

_SHAPES = (
    {"tp_size": 2, "dwdp_size": 2},
    {"random_seed": None},
    {"enable_deterministic_inference": True},
    {"enable_return_hidden_states": True},
    {
        "speculative_algorithm": "EAGLE",
        "speculative_num_steps": 3,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 4,
    },
    {"dp_size": 2, "tp_size": 2, "enable_dp_attention": True},
    {"enable_hierarchical_cache": True},
    {"disaggregation_mode": "prefill"},
    {"enable_lora": True, "max_lora_rank": 16},
    {"kv_cache_dtype": "fp8_e4m3", "page_size": 64},
)

# Which converted fields the shapes above actually reach. The rest declare only
# under conditions no CPU fixture reproduces (an env var, a device, a specific
# architecture), so the agreement check cannot see them -- the source scan is
# what covers those. Pinned so a shape that stops reaching a field is a
# failure rather than a silently narrower check; add to it when a shape is
# added.
_REACHED_BY_SHAPES = frozenset(
    {
        "_speculative_draft_quantization_explicitly_set",
        "allowed_media_domains",
        "attention_backend",
        "chunked_prefill_size",
        "cuda_graph_config",
        "custom_weight_loader",
        "device",
        "disable_cuda_graph",
        "disaggregation_ib_device",
        "dp_size",
        "enable_dp_attention",
        "enable_dp_attention_local_control_broadcast",
        "enable_dp_lm_head",
        "enable_flashinfer_allreduce_fusion",
        "encoder_transfer_backend",
        "enforce_disable_flashinfer_allreduce_fusion",
        "ep_size",
        "expert_distribution_recorder_buffer_size",
        "flashinfer_allreduce_fusion_backend",
        "grammar_backend",
        "hicache_ratio",
        "keep_mm_feature_on_device",
        "load_balance_method",
        "max_running_requests",
        "mem_fraction_static",
        "mm_feature_transport",
        "mm_process_config",
        "moe_a2a_backend",
        "moe_dense_tp_size",
        "moe_dp_size",
        "page_size",
        "random_seed",
        "return_hidden_states_mode",
        "sampling_backend",
        "schedule_conservativeness",
        "served_model_name",
        "speculative_algorithm",
        "speculative_draft_model_quantization",
        "tokenizer_path",
        "uses_mamba_radix_cache",
    }
)


def _server_args_writers(tree, path):
    """Assignment targets that land on a ServerArgs instance.

    Two mechanisms reach the same instance during resolution: a handler writing
    `self.<field>`, and a helper elsewhere in the tree writing through a
    `ServerArgs`-annotated parameter -- `set_default_server_args(args)` is
    called from the pipeline and writes `args.<field>`. Both bypass the
    declaration stash, so both have to be scanned; scanning only the handlers
    would let a field look converted while a second writer still assigns it.
    """
    names = {"self"} if path.name == "server_args.py" else set()
    # A parameter *named* `server_args` counts whether or not it carries the
    # annotation, so an unannotated writer cannot slip past the scan.
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
        names |= {
            arg.arg for arg in args.posonlyargs + args.args if arg.arg == "server_args"
        }
    return names


def _bare_assignments():
    """Assignments to a converted field that never reach the stash."""
    found = []
    for path in sorted(_SRT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except SyntaxError:
            continue
        names = _server_args_writers(tree, path)
        if not names:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            else:
                continue
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id in names
                    and target.attr in _RESOLVED_FIELDS
                ):
                    found.append(
                        f"{path.relative_to(_SRT)}:{node.lineno} "
                        f"{target.value.id}.{target.attr}"
                    )
    return sorted(found)


def _stash_overlay(server_args):
    """What the declarations say, last writer wins -- the projection's input."""
    overlay = {}
    for _source, declared in getattr(server_args, "_resolved_overrides", None) or ():
        overlay.update(declared)
    return overlay


class TestResolutionDeclarations(CustomTestCase):
    def setUp(self):
        # Resolution writes environment variables (the deprecated CUDA-IPC
        # transport flag among them), and those outlive the object that set
        # them -- a later test reading one would see this file's leftovers.
        super().setUp()
        environment = dict(os.environ)

        def restore():
            os.environ.clear()
            os.environ.update(environment)

        self.addCleanup(restore)

    def _resolve(self, extra):
        """A fully-resolved config: a real config.json, so the pipeline runs
        past its dummy-model early return."""
        path = tempfile.mkdtemp(prefix="declarations_")
        self.addCleanup(shutil.rmtree, path, ignore_errors=True)
        with open(os.path.join(path, "config.json"), "w") as handle:
            json.dump(_MINI_CONFIG, handle)
        fields = {"random_seed": 42}
        fields.update(extra)
        server_args = ServerArgs(model_path=path, device="cuda", **fields)
        # Construction is inert; this suite is about what resolution declares.
        server_args.resolve_once()
        return server_args

    def test_converted_fields_are_not_assigned_bare(self):
        bare = _bare_assignments()
        self.assertEqual(
            bare,
            [],
            "a converted field is assigned directly, so the projection would "
            "not see this write:\n  " + "\n  ".join(bare),
        )

    def test_the_stash_accounts_for_every_change_resolution_made(self):
        """The other direction: a field resolution moved is in the stash.

        The source scan states that no *assignment* escapes, which leaves the
        spellings a source scan cannot see -- a computed name, a write through
        a helper the scan does not recognize as holding the record. This
        compares the resolved value against what the caller supplied (or the
        field's default) and asks the stash to explain every difference, which
        is what the projection has to be able to do.
        """
        unexplained = []
        for shape in _SHAPES:
            supplied = {"random_seed": 42, **shape}
            server_args = self._resolve(shape)
            overlay = _stash_overlay(server_args)
            for field in dataclasses.fields(server_args):
                if field.name in ("model_path", "device") or field.name in overlay:
                    continue
                if field.name in supplied:
                    before = supplied[field.name]
                elif field.default is not dataclasses.MISSING:
                    before = field.default
                elif field.default_factory is not dataclasses.MISSING:
                    before = field.default_factory()
                else:
                    continue
                after = getattr(server_args, field.name, None)
                if after != before:
                    unexplained.append(
                        f"{shape} -> {field.name}: {before!r} -> {after!r}"
                    )
        self.assertEqual(
            unexplained,
            [],
            "resolution moved these fields without declaring them, so the "
            "projection would answer with the unresolved value:\n  "
            + "\n  ".join(unexplained),
        )

    def test_the_projection_input_is_the_resolved_configuration(self):
        """What the bags are built from equals what the record ends up holding.

        The projection reads `raw input + declarations` rather than the
        fields, so that it keeps working when the declarations stop
        materializing. While they still do, the two have to agree leaf for
        leaf -- a difference means the projection would publish something the
        record does not say, which is the failure this whole transition is
        meant to avoid.
        """
        from sglang.srt.arg_groups.arg_utils import namespace_of
        from sglang.srt.arg_groups.overrides import resolution_result

        differences = []
        for shape in _SHAPES:
            server_args = self._resolve(shape)
            for field in namespace_of(type(server_args)):
                projected = resolution_result(server_args, field)
                on_record = getattr(server_args, field)
                if projected != on_record:
                    differences.append(
                        f"{shape} -> {field}: projection={projected!r} "
                        f"record={on_record!r}"
                    )
        self.assertEqual(
            differences,
            [],
            "the projection and the record disagree about a config leaf:\n  "
            + "\n  ".join(differences),
        )

    def test_the_stash_agrees_with_the_fields_it_declared(self):
        mismatches = []
        for shape in _SHAPES:
            server_args = self._resolve(shape)
            overlay = _stash_overlay(server_args)
            for field, declared in overlay.items():
                if field not in _RESOLVED_FIELDS:
                    continue
                actual = getattr(server_args, field)
                if actual != declared:
                    mismatches.append(
                        f"{shape} -> {field}: field={actual!r} stash={declared!r}"
                    )
        self.assertEqual(
            mismatches,
            [],
            "a declared field and its stash entry disagree, so something "
            "assigned the field behind the declaration:\n  " + "\n  ".join(mismatches),
        )

    def test_the_shapes_reach_the_fields_they_are_meant_to(self):
        """A green agreement check over an empty stash would prove nothing."""
        declared = set()
        for shape in _SHAPES:
            declared |= set(_stash_overlay(self._resolve(shape))) & _RESOLVED_FIELDS
        missing = sorted(_REACHED_BY_SHAPES - declared)
        self.assertEqual(
            missing,
            [],
            "the shapes no longer reach these converted fields, so the "
            "agreement check silently stopped covering them:\n  "
            + "\n  ".join(missing),
        )


if __name__ == "__main__":
    unittest.main()
