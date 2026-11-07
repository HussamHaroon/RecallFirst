"""Contrastive NAMI scorer.

Ranks code chunks by the drop in token-level cross-entropy of a frozen
autoregressive LM when the chunk is conditioned on a real bug report vs a
length-matched null control issue.

Core identity (Research.md section 8.1):
    NAMI_contrast(C; I) = dH(C; I_real) - dH(C; I_null)
                        = [H(C|I_real) - H(C|I_null)] * (-1)
                        = H_null - H_real          (bits per token)
The unconditional pass H(C) cancels algebraically; we keep it optional for
diagnostics only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

LN2 = math.log(2.0)


@dataclass
class ScoreResult:
    h_real: float            # bits per token, conditioned on real issue
    h_null: float            # bits per token, conditioned on null issue
    h_uncond: float | None   # bits per token, no issue (diagnostics only)
    n_tokens: int            # chunk token count under full tokenization
    contrast_bpt: float      # H_null - H_real, bits per token  (the score)
    contrast_bits: float     # contrast_bpt * n_tokens (total bits)
    boundary_shifted: bool   # True if BPE merged across the prefix boundary


def resolve_boundary(tokenizer, prefix: str, chunk: str) -> tuple[list[int], int, bool]:
    """Tokenize prefix+chunk and find where chunk tokens start.

    Verifies that tokenizer(prefix) is an exact left-prefix of
    tokenizer(prefix+chunk). If BPE merged across the boundary, the boundary
    moves back to the longest common prefix and boundary_shifted=True.
    Returns (full_ids, chunk_start_index, boundary_shifted).
    """
    prefix_ids = tokenizer(prefix).input_ids
    full_ids = tokenizer(prefix + chunk).input_ids
    k = 0
    maxk = min(len(prefix_ids), len(full_ids))
    while k < maxk and prefix_ids[k] == full_ids[k]:
        k += 1
    return full_ids, k, k != len(prefix_ids)


class NAMIScorer:
    """Batched contrastive entropy scorer over a frozen causal LM.

    All entropies are computed with teacher forcing; model.generate() is never
    called, so scoring is deterministic given a fixed batch composition.
    """

    def __init__(self, model, tokenizer, device=None, dtype=torch.bfloat16):
        # Never move models that accelerate dispatched (device_map) or that are
        # quantized (bitsandbytes swaps module classes; torchao swaps weights).
        dispatched = getattr(model, "hf_device_map", None) is not None
        quantized = any(type(m).__name__ in ("Linear8bitLt", "Linear4bit")
                        for m in model.modules())
        if not dispatched and not quantized and dtype is not None:
            model = model.to(device=device or next(model.parameters()).device,
                             dtype=dtype)
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = self.model.device  # handles device_map="auto" shardings
        self._resolve_decoder_path()

    def _resolve_decoder_path(self):
        """Find backbone (pre-lm_head) and lm_head for memory-safe scoring.

        Gemma's ~262k vocab means a full [B, L, V] logits tensor costs ~2GB per
        (batch x 1k tokens) in fp32 — infeasible on consumer GPUs. Scoring via
        the decoder path materializes only hidden states [B, L, d] and computes
        CE in position slices through the lm_head: identical math, tiny memory.
        """
        self.decoder = None
        self.lm_head = None
        m = self.model
        candidates = [m] + [getattr(m, a, None) for a in
                            ("language_model", "text_model", "model")]
        for c in candidates:
            if c is None:
                continue
            dec = None
            try:
                dec = c.get_decoder()
            except Exception:
                dec = getattr(c, "model", None) or getattr(c, "text_model", None)
            head = None
            try:
                head = c.get_output_embeddings()
            except Exception:
                head = getattr(c, "lm_head", None)
            if dec is not None and head is not None:
                self.decoder, self.lm_head = dec, head
                break
        cfg = getattr(m, "config", None)
        for c in (cfg, getattr(cfg, "text_config", None)):
            sc = getattr(c, "final_logit_softcapping", None)
            if sc is not None:
                self.logit_softcapping = sc
                break
        else:
            self.logit_softcapping = None

    # ---- internals -------------------------------------------------------

    def _build_batch(self, prefixes, chunks):
        """Tokenize prefix+chunk pairs; return padded batch + chunk starts."""
        seqs, starts, shifted = [], [], []
        for pre, chunk in zip(prefixes, chunks):
            ids, start, sh = resolve_boundary(self.tokenizer, pre, chunk)
            if start >= len(ids):  # degenerate: chunk produced no new tokens
                ids, start = ids + [self.tokenizer.eos_token_id or 0], start
            seqs.append(ids)
            starts.append(start)
            shifted.append(sh)
        maxlen = max(len(s) for s in seqs)
        pad = self.tokenizer.pad_token_id or 0
        input_ids = torch.full((len(seqs), maxlen), pad, dtype=torch.long)
        attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, : len(s)] = torch.tensor(s)
            attn[i, : len(s)] = 1
        n_tok = [len(s) - st for s, st in zip(seqs, starts)]
        return input_ids.to(self.device), attn.to(self.device), starts, n_tok, shifted

    @torch.no_grad()
    def _entropy_nats(self, input_ids, attn, starts, slice_len: int = 128):
        """Mean NLL (nats/token) over chunk-token positions, per batch row.

        Memory-safe for huge vocabs: if the decoder path is available, run the
        backbone only (hidden states [B,L,d]) and compute CE in position slices
        through the lm_head — the [B,L,V] logits tensor never exists.
        """
        B = input_ids.shape[0]
        tgt = input_ids[:, 1:]  # [B, L-1]
        Lm1 = tgt.shape[1]
        _dbg = not getattr(type(self), "_dbg_once", False)
        type(self)._dbg_once = True
        if _dbg and torch.cuda.is_available():
            print(f"  [dbg] B={B} L={input_ids.shape[1]} mem_pre="
                  f"{torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
        nll = torch.zeros(B, Lm1, device=input_ids.device, dtype=torch.float32)
        if self.decoder is not None and self.lm_head is not None:
            hs = self.decoder(input_ids=input_ids, attention_mask=attn,
                              use_cache=False).last_hidden_state  # [B,L,d]
            if _dbg and torch.cuda.is_available():
                print(f"  [dbg] after decoder mem="
                      f"{torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
            for s in range(0, Lm1, slice_len):
                e = min(s + slice_len, Lm1)
                logits = self.lm_head(hs[:, s:e, :])  # [B, sl, V] transient
                if self.logit_softcapping:
                    cap = self.logit_softcapping
                    logits = cap * torch.tanh(logits.float() / cap)
                nll[:, s:e] = torch.nn.functional.cross_entropy(
                    logits.float().reshape(-1, logits.size(-1)),
                    tgt[:, s:e].reshape(-1), reduction="none",
                ).view(B, e - s)
                del logits
            del hs
            if _dbg and torch.cuda.is_available():
                print(f"  [dbg] after CE mem="
                      f"{torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
        else:
            logits = self.model(input_ids=input_ids, attention_mask=attn,
                                use_cache=False).logits  # [B, L, V]
            for s in range(0, Lm1, slice_len):
                e = min(s + slice_len, Lm1)
                lg = logits[:, s:e, :].float()
                nll[:, s:e] = torch.nn.functional.cross_entropy(
                    lg.reshape(-1, lg.size(-1)), tgt[:, s:e].reshape(-1),
                    reduction="none",
                ).view(B, e - s)
        mask = torch.zeros_like(nll)
        for b, st in enumerate(starts):
            mask[b, st - 1 :] = 1.0
        if not getattr(self, "count_padding", False):
            # Bug fix 2026-11-01: exclude right-padding targets. The original
            # code averaged pad-token NLL into shorter rows of each batch.
            # Set scorer.count_padding = True to reproduce pre-fix numbers.
            mask = mask * attn[:, 1:].to(mask.dtype)
        lengths = mask.sum(-1).clamp(min=1)
        return (nll * mask).sum(-1) / lengths  # [B] nats per chunk token

    # ---- public API ------------------------------------------------------

    def score_chunks(
        self,
        issue_text: str,
        null_issue_text: str,
        chunks: list[str],
        batch_size: int = 8,
        with_uncond: bool = False,
        prefix_template: str = "### Issue Description:\n{issue}\n\n### Code Snippet:\n",
        base_prefix: str = "### Code Snippet:\n",
    ) -> list[ScoreResult]:
        """Score every chunk: contrast_bpt = H_null - H_real (bits/token).

        Prefixes for real/null use the SAME template so their only difference
        is issue content (length-matching is the null selector's job).
        """
        pre_real = prefix_template.format(issue=issue_text)
        pre_null = prefix_template.format(issue=null_issue_text)
        pre_base = base_prefix

        h_real = self._pass(pre_real, chunks, batch_size, label="real")
        h_null = self._pass(pre_null, chunks, batch_size, label="null")
        h_uncond = (self._pass(pre_base, chunks, batch_size, label="uncond")
                    if with_uncond else None)
        # one extra call to reuse boundary info
        _, _, _, n_tok, shifted = self._build_batch([pre_real] * len(chunks), chunks)

        out = []
        for i in range(len(chunks)):
            contrast = (h_null[i] - h_real[i]) / LN2
            out.append(
                ScoreResult(
                    h_real=h_real[i] / LN2,
                    h_null=h_null[i] / LN2,
                    h_uncond=None if h_uncond is None else h_uncond[i] / LN2,
                    n_tokens=n_tok[i],
                    contrast_bpt=contrast,
                    contrast_bits=contrast * n_tok[i],
                    boundary_shifted=shifted[i],
                )
            )
        return out

    @torch.no_grad()
    def score_qlm(
        self,
        issue_text: str,
        chunks: list[str],
        prefix_template: str = "### Code Snippet:\n{chunk}\n\n### Issue Description:\n",
        max_issue_chars: int = 4000,
        label: str = "qlm",
        progress_every: int = 10,
    ) -> list[float]:
        """Query-likelihood baseline (Zhuang et al., EMNLP-F 2023, arXiv:2310.13243).

        Scores each chunk by the mean log2-probability of the ISSUE tokens given
        the chunk: QLM(C; I) = (1/M) sum_j log2 P(s_j | C, s_<j). Higher = better.
        This is the reversed-direction likelihood estimate of the same
        chunk-issue association NAMI measures; since log P(I) is constant per
        instance, ranking by QLM equals ranking by PMI(I; C) under the model.
        The issue text is fixed across chunks, so mean vs sum gives the same
        ordering up to BPE boundary effects. Uses the same memory-safe decoder
        path and adaptive batching as the NAMI passes (sequence = chunk + issue).
        """
        issue = issue_text[:max_issue_chars]
        prefixes = [prefix_template.format(chunk=c) for c in chunks]
        res = []
        batches = self._adaptive_batches(chunks)
        pos = 0
        n_batches = len(batches)
        for bi, sub in enumerate(batches, 1):
            pre = prefixes[pos:pos + len(sub)]
            pos += len(sub)
            ids, attn, starts, _, _ = self._build_batch(pre, [issue] * len(sub))
            nats = self._entropy_nats(ids, attn, starts).tolist()
            res.extend(-n / LN2 for n in nats)  # mean log2 P(issue | chunk)
            if label and (bi % progress_every == 0 or bi == n_batches):
                print(f"  [{label}] batch {bi}/{n_batches}", flush=True)
        return res

    def _adaptive_batches(self, chunks: list[str], token_budget: int = 12000,
                          min_bs: int = 2, max_bs: int = 8,
                          prefix_tokens: int = 450) -> list[list[str]]:
        """Group length-sorted chunks so batch_size x padded_len stays within a
        token budget. Every sequence carries the issue prefix (~450 tokens) on
        top of the chunk itself — the estimate must include it (the B=190/L=465
        OOM cascade was exactly this omission). max_bs is a hard seatbelt."""
        batches, cur, cur_max = [], [], 0
        for ch in chunks:
            est = prefix_tokens + max(8, int(len(ch) / 3.5))
            new_max = max(cur_max, est)
            would_split = ((len(cur) + 1) * new_max > token_budget
                           or len(cur) + 1 > max_bs)
            if cur and would_split and len(cur) >= min_bs:
                batches.append(cur)
                cur, cur_max = [ch], est
            else:
                cur.append(ch)
                cur_max = new_max
        if cur:
            batches.append(cur)
        return batches

    def _pass(self, prefix: str, chunks: list[str], batch_size: int,
              label: str = "", progress_every: int = 10,
              adaptive: bool = True) -> list[float]:
        res = []
        batches = (self._adaptive_batches(chunks) if adaptive
                   else [chunks[i:i + batch_size]
                         for i in range(0, len(chunks), batch_size)])
        n_batches = len(batches)
        for bi, sub in enumerate(batches, 1):
            ids, attn, starts, _, _ = self._build_batch([prefix] * len(sub), sub)
            res.extend(self._entropy_nats(ids, attn, starts).tolist())
            if label and (bi % progress_every == 0 or bi == n_batches):
                print(f"  [{label}] batch {bi}/{n_batches}", flush=True)
        return res
