"""The Jibo C runtime (ports/jibo) must answer as the torch engine does, from the same checkpoint.

A tiny random-weight Qwen3.5 text model is saved as a base checkpoint, a decider with a LoRA
adapter over every projection is built on it, and its tokenizer is a small byte-level BPE with
Qwen's split, trained here on the test prompts. The exporter writes model.jdw and tokenizer.jdt,
the C runtime is compiled with the host compiler, and both engines answer the same requests:
one question (the whole-prompt path) and several (the shared-prefix path). Skips without a C
compiler. Nothing is downloaded.

To run the same tests on emulated ARMv7, as the robot runs the code (NEON kernels included):

    JIBO_CC=arm-linux-gnueabihf-gcc JIBO_CFLAGS="-O2 -march=armv7-a -mfpu=neon -mfloat-abi=hard" \
    JIBO_RUNNER="qemu-arm -L /usr/arm-linux-gnueabihf" pytest tests/test_jibo_runtime.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3_5TextConfig"):
    pytest.skip("this transformers has no Qwen3.5", allow_module_level=True)
CC = shutil.which(os.environ.get("JIBO_CC", os.environ.get("CC", "cc"))) or shutil.which("gcc")
if CC is None or shutil.which("make") is None:
    pytest.skip("no C compiler", allow_module_level=True)
RUNNER = os.environ.get("JIBO_RUNNER", "").split()

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PORT = os.path.join(REPO, "ports", "jibo")
sys.path.insert(0, os.path.join(PORT, "tools"))

from export_jdw import load_merged, meta_for  # noqa: E402
from jdt import build_jdt  # noqa: E402
from jdw import dequantise, quantise, tensors_from_model, write_jdw  # noqa: E402

from strands_decider.infer import EngineConfig, SystemOneEngine  # noqa: E402
from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel  # noqa: E402
from strands_decider.prompting import render_question, render_state  # noqa: E402
from strands_decider.schema import (  # noqa: E402
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOneRequest,
)

QWEN2_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
               r"|\s*[\r\n]+|\s+(?!\S)|\s+")
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]
POLICY = ("Loans run 21 days and renew twice online unless another patron holds the item. "
          "Accounts with fines above ten dollars can't renew until the fine is paid. ")
QUESTIONS = {
    "team": ChoiceQuestion(instructions="Which team should handle this?",
                           criteria={"circulation": "renewals and holds", "billing": "fines",
                                     "reference": None}),
    "urgent": NoulQuestion(instructions="  Does the patron convey urgency?\n",
                           criteria={"true": "the patron needs it soon"}),
    "mood": ScoreQuestion(instructions="How frustrated is the patron?",
                          criteria=["calm", "  annoyed\tbut polite ", "angry"]),
}
REQUESTS = [
    SystemOneRequest(state=POLICY + "Can I renew the cookbook?", questions={"team": QUESTIONS["team"]}),
    SystemOneRequest(state=POLICY * 6 + "My renewal failed and I leave Friday! Café ☕ naïve 日本",
                     questions=QUESTIONS),
]


def _tokenizer():
    from tokenizers import (
        Regex,
        Tokenizer,
        decoders,
        models,
        normalizers,
        pre_tokenizers,
        processors,
        trainers,
    )
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN2_SPLIT), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    tok.decoder = decoders.ByteLevel()
    texts = [render_state(r.state) for r in REQUESTS] + [render_question(q).text for q in QUESTIONS.values()]
    tok.train_from_iterator(texts * 3, trainers.BpeTrainer(
        vocab_size=600, special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|endoftext|>", pad_token="<|endoftext|>")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("jibo")
    out = tmp / "build"
    make = ["make", "-s", "-C", PORT, f"OUT={out}", f"CC={CC}"]
    if os.environ.get("JIBO_CFLAGS"):
        make.append(f"CFLAGS={os.environ['JIBO_CFLAGS']}")
    subprocess.run(make, check=True)
    tok = _tokenizer()
    torch.manual_seed(0)
    base_cfg = transformers.Qwen3_5TextConfig(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=16,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"])
    base = tmp / "base"
    transformers.Qwen3_5ForCausalLM(base_cfg).save_pretrained(base)
    cfg = StrandsDeciderConfig(base_model=str(base), head_type="pointer", pointer_dim=16,
                               max_length=512, torch_dtype="float32", lora_r=4, lora_alpha=8,
                               lora_targets=LORA_TARGETS, temperature_by_kind={"noul": 0.8, "score": 1.3},
                               ordinal_smoothing=0.1)
    torso = transformers.Qwen3_5ForCausalLM.from_pretrained(base).model
    model = StrandsDeciderModel(cfg, torso, tok)
    model.attach_lora()
    with torch.no_grad():
        for name, p in model.torso.named_parameters():
            if "lora_B" in name:
                p.normal_(std=0.1)
        for p in model.head.parameters():
            p.normal_(std=0.3)
    ckpt = tmp / "ckpt"
    model.save_pretrained(str(ckpt))

    merged = load_merged(str(ckpt))
    exports = {}
    for kind in ("f32", "q8"):
        d = tmp / kind
        d.mkdir()
        with torch.no_grad():
            write_jdw(str(d / "model.jdw"), meta_for(merged, str(ckpt), kind),
                      tensors_from_model(merged.torso, merged.head, merged.config, kind, "f32"))
        (d / "tokenizer.jdt").write_bytes(build_jdt(merged.tokenizer.backend_tokenizer.to_str()))
        exports[kind] = d
    engine = SystemOneEngine(StrandsDeciderModel.load(str(ckpt)), EngineConfig(device="cpu"))
    return out, exports, engine


def _ask(binary, export, request, *extra):
    res = subprocess.run([*RUNNER, str(binary), "ask", str(export / "model.jdw"), str(export / "tokenizer.jdt"),
                          "--raw", *extra], input=request.model_dump_json().encode(), capture_output=True,
                         check=True)
    lines = res.stdout.decode().strip().split("\n")
    raw = {d["name"]: dict(zip(d["labels"], d["probs"], strict=True)) for d in map(json.loads, lines[1:])}
    return json.loads(lines[0]), raw


def _probabilities(answer):
    if answer["type"] == "noul":
        return {"true": answer["noul"], "false": 1 - answer["noul"]}
    return answer["probabilities"]


@pytest.mark.parametrize("request_index", range(len(REQUESTS)))
@pytest.mark.parametrize("prefix_cache", [True, False])
def test_c_runtime_answers_as_torch_does(built, request_index, prefix_cache):
    binary, exports, engine = built
    request = REQUESTS[request_index]
    engine.cfg.use_prefix_cache = prefix_cache
    want = json.loads(engine.evaluate(request).model_dump_json())
    got, raw = _ask(binary / "jibo-decider", exports["f32"], request,
                    *([] if prefix_cache else ["--no-prefix-cache"]))
    assert got["usage"] == want["usage"]
    assert got["answers"].keys() == want["answers"].keys()
    for name, w in want["answers"].items():
        g = got["answers"][name]
        assert g["type"] == w["type"]
        for label, p in _probabilities(w).items():
            assert raw[name][label] == pytest.approx(p, abs=6e-5), (name, label)  # want is rounded to 4 places
        if w["type"] == "score":
            assert g["legend"] == w["legend"]
            assert g["score"] == pytest.approx(w["score"], abs=2e-4)
        if w["type"] == "choice":
            assert g["choice"] == w["choice"]


def test_quantised_weights_stay_close(built):
    binary, exports, _ = built
    _, f32 = _ask(binary / "jibo-decider", exports["f32"], REQUESTS[1])
    _, q8 = _ask(binary / "jibo-decider", exports["q8"], REQUESTS[1])
    worst = max(abs(q8[q][k] - f32[q][k]) for q in f32 for k in f32[q])
    assert worst < 0.02


def test_quantisation_round_trips():
    rng = torch.Generator().manual_seed(1)
    w = torch.randn(8, 64, generator=rng).numpy()
    for dtype, tol in ((3, 0.01), (4, 0.2)):
        back = dequantise(quantise(w, dtype), dtype, 8, 64)
        assert abs(back - w).max() <= tol * abs(w).max()


def test_c_tokenizer_matches(built):
    binary, exports, engine = built
    import struct
    texts = ["Hello  world\n\n x", "don't I'LL", "naïve café 日本 ☕", "<|endoftext|>a", "1. x — y\n2. z",
             "   ", "\t\n\r\n", "a" * 300]
    payload = b"".join(struct.pack("<I", len(t.encode())) + t.encode() for t in texts)
    res = subprocess.run([*RUNNER, str(binary / "tok_dump"), str(exports["f32"] / "tokenizer.jdt")], input=payload,
                         capture_output=True, check=True)
    for text, line in zip(texts, res.stdout.decode().split("\n"), strict=False):
        ids = [int(x.split(":")[0]) for x in line.split()]
        assert ids == engine.tok(text, add_special_tokens=False)["input_ids"], text


@pytest.mark.parametrize("body", [
    '{"state": {"b": [1, 2.50, {"x": null}], "a": "q\\"\\\\\\t\\u00e9\\ud83d\\ude00", "n": -0, "f": 1e16,'
    ' "g": 1.5e-05, "h": 12345678901234567890, "i": 1E2, "j": 0.1, "k": 123456789.125, "e": {}, "l": []},'
    ' "questions": {"q": {"type": "choice", "instructions": ["do", {"it": true}],'
    ' "criteria": {"a": {"k": 1.0, "k": 2}, "b": null, "c": "  spaced\\n  out  "}}}}',
    '{"state": "  Caf\\u0065\\u0301 \\u3000\\u000b", "questions": {"n": {"type": "noul", "instructions": "x",'
    ' "criteria": {"true": {"why": [3.0, -2.5e-07]}}}, "s": {"type": "score", "instructions": {"rate": 1},'
    ' "criteria": [" low ", "\\thigh\\u2028"]}}}',
])
def test_c_renders_prompts_as_python_does(built, body):
    import unicodedata

    binary, _, _ = built
    req = SystemOneRequest.model_validate_json(body)
    want = [render_state(req.state)] + [render_question(q).text for q in req.questions.values()]
    res = subprocess.run([*RUNNER, str(binary / "jibo-decider"), "render"], input=body.encode(),
                         capture_output=True, check=True)
    assert json.loads(res.stdout) == [unicodedata.normalize("NFC", w) for w in want]
