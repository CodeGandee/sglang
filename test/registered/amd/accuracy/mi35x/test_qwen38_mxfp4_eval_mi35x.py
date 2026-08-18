"""MI35x Qwen3.8-2.4T-A95B MXFP4 GSM8K Evaluation Test (8-GPU)

Tests amd/Qwen3.8-2.4T-A95B-Quark-MXFP4, AMD's day-0 Quark quantization of
Qwen/Qwen3.8-2.4T-A95B-FP8, on a single 8-GPU MI35x node.

Qwen3.8 is a 2.4T-parameter / 95B-active hybrid MoE: 23 repeats of 3 x Gated
DeltaNet -> MoE then 1 x Gated Attention -> MoE, 512 experts with 10 routed + 1
shared active. It reuses the Qwen3.5 architecture -- the checkpoint reports
``Qwen3_5MoeForCausalLM`` -- so no model code is added here. What is missing,
and what this test supplies, is nightly evidence that the ROCm kernels behind
that path keep producing correct tokens.

MXFP4 rather than FP8: at 2.4T parameters FP8 is ~2.4 TB against 8 x 288 GB =
2.30 TB per MI355X node, so the FP8 checkpoint has no single-node AMD recipe
(the cookbook serves it as MI300X TP8 x PP2 over two nodes) and single-node
means FP4. Only the routed experts are quantized; attention, the shared expert,
the MoE gate and ``lm_head`` stay at source precision, which is why AMD
measures the same 97.49 GSM8K as the FP8 baseline (100% recovery) and why this
test gates the FP8 checkpoint's quality even though it serves the MXFP4 one.

Server arguments and the eval invocation reproduce the recipe published on the
model card, so a regression here is a regression against a number AMD has
already measured. Two of those flags are load-bearing rather than restatements
of a default:

  * ``--page-size 1`` -- ``_page_size_default`` bumps the default to 64 on HIP
    when the container sets SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d, so the
    measured geometry only holds if the page size is pinned.
  * ``--attention-backend aiter`` -- no arg override picks a backend for
    ``Qwen3_5MoeForCausalLM`` on ROCm, so the AITER path has to be named.

The scorer extracts the last number in the reply and the server runs with no
``--reasoning-parser``, so a ``<think>`` block still scores: the reasoning
stays in ``message.content`` rather than being split into ``reasoning_content``,
which would leave ``content`` empty and score 0.

MXFP4 needs gfx95x, so this is MI35x-only and ROCm 7.2-only; it does not
register on gfx942 (MI300/MI325).

Registry: nightly-amd-accuracy-8-gpu-mi35x-qwen38-mxfp4 suite
"""

import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

# Register for AMD CI - Qwen3.8 MXFP4 accuracy test on MI35x (~150 min: the
# 1.2 TB checkpoint dominates startup, then the GSM8K test split at TP8)
register_amd_ci(
    est_time=9000, suite="nightly-amd-accuracy-8-gpu-mi35x-qwen38-mxfp4", nightly=True
)

QWEN38_MXFP4_MODEL_PATH = os.environ.get(
    "QWEN38_MXFP4_MODEL_PATH", "amd/Qwen3.8-2.4T-A95B-Quark-MXFP4"
)
SERVER_LAUNCH_TIMEOUT = 9000
TP_SIZE = 8
# AMD measures 0.9749 on this checkpoint. The gate sits ~5% below it, matching
# the relative tolerance the sibling Qwen3.5 MI35x evals allow.
ACCURACY_THRESHOLD = 0.93


class TestQwen38Mxfp4EvalMI35x(CustomTestCase):
    """Qwen3.8-2.4T-A95B MXFP4 GSM8K Evaluation Test for AMD MI35x."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_examples = int(os.environ.get("GSM8K_NUM_QUESTIONS", "1319"))
        cls.max_tokens = int(os.environ.get("GSM8K_MAX_NEW_TOKENS", "2048"))

    def test_qwen38_mxfp4_gsm8k_accuracy(self):
        """Test Qwen3.8 MXFP4 with the GSM8K few-shot benchmark."""
        other_args = [
            "--tp",
            str(TP_SIZE),
            "--attention-backend",
            "aiter",
            "--page-size",
            "1",
            "--chunked-prefill-size",
            "16384",
            "--mem-fraction-static",
            "0.9",
            "--trust-remote-code",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
            "--watchdog-timeout",
            "1200",
        ]
        env = os.environ.copy()
        # Gates the AITER MXFP4-MoE / GEMM / norm / rope kernels. The ROCm image
        # sets it; a bare-pip host does not.
        env["SGLANG_USE_AITER"] = "1"

        process = popen_launch_server(
            QWEN38_MXFP4_MODEL_PATH,
            self.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            env=env,
        )

        try:
            requests.get(self.base_url + "/flush_cache")

            args = SimpleNamespace(
                base_url=self.base_url,
                model=QWEN38_MXFP4_MODEL_PATH,
                eval_name="gsm8k",
                num_examples=self.num_examples,
                num_threads=512,
                max_tokens=self.max_tokens,
                chat_template_kwargs={"enable_thinking": False},
            )
            metrics = run_eval(args)
            acc = metrics["score"]

            passed = acc >= ACCURACY_THRESHOLD
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  accuracy={acc:.3f} threshold={ACCURACY_THRESHOLD} {status}")

            if is_in_ci():
                summary = "### Qwen3.8-2.4T-A95B MXFP4 (MI35x)\n\n"
                summary += "| Model | TP | Accuracy | Threshold | Status |\n"
                summary += "| ----- | -- | -------- | --------- | ------ |\n"
                summary += (
                    f"| {QWEN38_MXFP4_MODEL_PATH} | {TP_SIZE} | {acc:.3f} | "
                    f"{ACCURACY_THRESHOLD} | {status} |\n"
                )
                write_github_step_summary(summary)

            self.assertGreaterEqual(
                acc,
                ACCURACY_THRESHOLD,
                f"Qwen3.8 MXFP4 accuracy {acc:.3f} below threshold {ACCURACY_THRESHOLD}",
            )
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
