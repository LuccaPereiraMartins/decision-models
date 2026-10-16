"""Raw-document model definitions: windows, frozen features, attention heads and LoRA."""

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel, AutoTokenizer
from schemas import BINARY_LABELS, LAYA_MODEL, LAYA_REVISION, task_labels, task_questions


MODELS = {
    "finbert": ("ProsusAI/finbert", "4556d13015211d73dccd3fdd39d39232506f3e43"),
}


def checkpoint_config(saved, requested=None):
    """Reject a checkpoint from another reference task before loading pretrained weights."""
    config = saved["config"] if saved else (requested or {})
    labels = task_labels(config)
    if labels != BINARY_LABELS:
        raise ValueError("Historical three-class models require their archived source snapshot")
    if saved and requested is not None and task_labels(requested) != labels:
        raise ValueError("Checkpoint task differs from the requested task")
    return config


def window_checkpoint(function, *args):
    # Forward without a graph also releases masks captured by nested model checkpoints.
    # Token IDs cannot require gradients; the empty input enables reentrant backward.
    dummy = torch.empty(0, device=args[0].device, requires_grad=True)
    return checkpoint(lambda _, *values: function(*values), dummy, *args, use_reentrant=True)


class EncoderModel(nn.Module):
    def __init__(self, device="cuda", kind="finbert", adaptation="head", lora_rank=8, checkpoint=None, config=None):
        super().__init__()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True) if checkpoint else None
        config = checkpoint_config(saved, config)
        self.labels = task_labels(config)
        self.task_id = config.get("task_id", "binary-v2")
        self.training_config = config or None
        self.selected_epoch = saved["state"]["epoch"] if saved else None
        self.device = torch.device(device)
        self.kind = config.get("kind", kind)
        self.adaptation = config.get("adaptation", adaptation)
        self.lora_rank = config.get("lora_rank", lora_rank)
        self.model_id, self.revision = MODELS[self.kind]
        self.model_id = config.get("model", self.model_id)
        self.revision = config.get("revision", self.revision)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        self.encoder = AutoModel.from_pretrained(self.model_id, revision=self.revision)
        self.encoder.requires_grad_(False)
        if self.adaptation == "lora":
            from peft import LoraConfig, get_peft_model

            self.encoder = get_peft_model(
                self.encoder,
                LoraConfig(
                    r=self.lora_rank,
                    lora_alpha=2 * self.lora_rank,
                    lora_dropout=0.05,
                    target_modules=["query", "value"],
                    bias="none",
                ),
            )
        if self.adaptation != "head":
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        width = self.encoder.config.hidden_size
        self.pool = nn.Sequential(nn.Linear(width, 32), nn.Tanh(), nn.Linear(32, 1, bias=False))
        nn.init.zeros_(self.pool[-1].weight)
        self.classifier = nn.Sequential(nn.LayerNorm(width), nn.Dropout(0.1), nn.Linear(width, len(self.labels)))
        self.cfg = {"max_len": config.get("max_len", 512), "head_max_len": 0}
        if self.cfg["max_len"] > self.encoder.config.max_position_embeddings:
            raise ValueError("Context exceeds the pinned encoder's positional capacity")
        self.uniform_pooling = config.get("pooling_method") == "uniform"
        if self.uniform_pooling:
            self.pool.requires_grad_(False)
        self.window_batch = config.get("window_batch", 8)
        self.to(self.device)
        if saved:
            expected = {n for n, p in self.named_parameters() if p.requires_grad}
            if set(saved["weights"]) != expected:
                raise ValueError("Checkpoint trainable parameters do not match the encoder adapter")
            self.load_state_dict(saved["weights"], strict=False)

    def train(self, mode=True):
        super().train(mode)
        if self.adaptation == "head":
            self.encoder.eval()
        return self

    def windows(self, body):
        ids = self.tokenizer.encode(body, add_special_tokens=False, truncation=False, verbose=False)
        # Context windows bound GPU memory; retain every token, regardless of document type.
        budget = self.cfg["max_len"] - 2
        return [
            [self.tokenizer.cls_token_id, *ids[i : i + budget], self.tokenizer.sep_token_id]
            for i in range(0, max(1, len(ids)), budget)
        ]

    def features(self, items):
        """One FinBERT vector per window, averaged over unpadded tokens."""
        vectors = []
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            for start in range(0, len(items), self.window_batch):
                rows = [torch.tensor(ids, device=self.device) for ids in items[start : start + self.window_batch]]
                ids = pad_sequence(rows, batch_first=True, padding_value=self.tokenizer.pad_token_id)
                mask = ids != self.tokenizer.pad_token_id
                # Recompute whole window batches so long documents retain inputs, not every hidden activation.
                pooled = (
                    window_checkpoint(self.window_features, ids, mask)
                    if self.adaptation == "lora" and torch.is_grad_enabled()
                    else self.window_features(ids, mask)
                )
                vectors.append(pooled)
        return torch.cat(vectors)

    def window_features(self, ids, mask):
        hidden = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        return (hidden.float() * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]

    def embed(self, body):
        """Frozen linear baselines use the normalized mean of every raw window vector."""
        vector = self.features(self.windows(body)).mean(0)
        return torch.nn.functional.normalize(vector, dim=0)

    def forward(self, items):
        vectors = items.to(self.device) if isinstance(items, torch.Tensor) else self.features(items)
        weights = (
            torch.ones((len(vectors), 1), device=self.device) / len(vectors)
            if self.uniform_pooling
            else self.pool(vectors).softmax(0)
        )
        return self.classifier((weights * vectors).sum(0)), weights[:, 0]


class DocumentModel(nn.Module):
    def __init__(self, device="cuda", checkpoint=None, lora_rank=0, config=None):
        import warnings

        super().__init__()
        self.device = torch.device(device)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True) if checkpoint else None
        config = checkpoint_config(saved, config)
        self.labels = task_labels(config)
        self.task_id = config.get("task_id", "binary-v2")
        self.training_config = saved["config"] if saved else None
        self.selected_epoch = saved["state"]["epoch"] if saved else None
        config = {"model": LAYA_MODEL, "revision": LAYA_REVISION, **config}
        self.model_id, self.revision = config["model"], config["revision"]
        self.kind = "laya"
        self.lora_rank = config.get("lora_rank", lora_rank)
        warnings.filterwarnings("ignore", message=r"laya:.*invalid temperatures")
        self.questions = config.get("questions", task_questions(self.labels))
        if tuple(self.questions["triage"]["criteria"]) != self.labels:
            raise ValueError("Question options must follow the task's label order")
        import laya

        agent = laya.load(config["model"], device=device, revision=config["revision"], fast=False)
        self.base, self.tokenizer, self.cfg = agent.model, agent.tok, agent.cfg
        self.window_batch = config.get("window_batch", 8)
        if self.training_config:
            self.cfg.update(max_len=config["max_len"], head_max_len=config["head_max_len"])
        elif "max_len" in config:
            self.cfg["max_len"] = config["max_len"]
        if self.cfg["max_len"] > self.base.encoder.config.max_position_embeddings:
            raise ValueError("Context exceeds the pinned encoder's positional capacity")
        for name, parameter in self.base.named_parameters():
            parameter.requires_grad_(not name.startswith(("encoder.", "act_head.")))
        if self.lora_rank:
            from peft import LoraConfig, get_peft_model

            self.base.encoder = get_peft_model(
                self.base.encoder,
                LoraConfig(
                    r=self.lora_rank,
                    lora_alpha=2 * self.lora_rank,
                    lora_dropout=0.05,
                    target_modules=["Wqkv"],
                    bias="none",
                ),
            )
            self.base.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.base.head_checkpointing = True
        self.pool = nn.Sequential(nn.Linear(len(self.labels), 16), nn.Tanh(), nn.Linear(16, 1, bias=False))
        nn.init.zeros_(self.pool[-1].weight)  # Begin with uniform document pooling.
        self.uniform_pooling = config.get("pooling_method") == "uniform"
        if self.uniform_pooling:
            self.pool.requires_grad_(False)
        self.to(self.device)
        if saved:
            expected = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
            if set(saved["weights"]) != expected:
                raise ValueError("Checkpoint does not match the trainable head/pooling parameters")
            self.load_state_dict(saved["weights"], strict=False)

    def train(self, mode=True):
        super().train(mode)
        if not self.lora_rank:
            self.base.encoder.eval()  # Keep frozen encoder features deterministic.
        self.base.act_head.eval()
        return self

    def windows(self, body):
        from laya.common import build_sequence

        question = self.questions["triage"]
        q = {"t": "choice", "ins": question["instructions"], "crit": question["criteria"]}
        tokens = self.tokenizer.encode(
            body.replace(self.tokenizer.mask_token, " "), add_special_tokens=False, truncation=False, verbose=False
        )
        budget = self.cfg["max_len"] - self.cfg["head_max_len"] - 8
        items = []
        for start in range(0, max(1, len(tokens)), budget // 2):
            chunk = tokens[start : start + budget]
            ids, markers = build_sequence(
                self.tokenizer, "", q, self.cfg["max_len"], self.cfg["head_max_len"], state_ids=chunk
            )
            if len(markers) != len(self.labels) or ids[-len(chunk) - 1 : -1] != chunk:
                raise ValueError("Window formatting dropped input tokens or label markers")
            items.append((ids, markers))
            if start + budget >= len(tokens):
                break
        return items

    def forward(self, items):
        scores = []
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            for start in range(0, len(items), self.window_batch):
                chunk = items[start : start + self.window_batch]
                sequences = [torch.tensor(ids, device=self.device) for ids, _ in chunk]
                ids = pad_sequence(sequences, batch_first=True, padding_value=self.tokenizer.pad_token_id)
                lengths = torch.tensor([len(seq) for seq in sequences], device=self.device)
                mask = torch.arange(ids.shape[1], device=self.device)[None, :] < lengths[:, None]
                markers = torch.tensor([positions for _, positions in chunk], device=self.device)
                logits = (
                    window_checkpoint(self.window_scores, ids, mask, markers)
                    if torch.is_grad_enabled()
                    else self.window_scores(ids, mask, markers)
                )
                scores.append(logits)
            scores = torch.cat(scores)
            if self.uniform_pooling:
                return scores.mean(0), torch.ones(len(scores), device=self.device) / len(scores)
            # Gate on relative scores so arbitrary common logit offsets cannot affect attention.
            attention = self.pool(scores.log_softmax(-1)).float().softmax(0)
            return (attention * scores).sum(0), attention[:, 0]

    def window_scores(self, ids, mask, markers):
        logits, _ = self.base(
            ids,
            mask,
            markers,
            torch.ones_like(markers, dtype=torch.bool),
            torch.zeros(len(ids), dtype=torch.long, device=self.device),
            detach_encoder=not self.lora_rank,
        )
        if logits.shape != (len(ids), len(self.labels)):
            raise ValueError("Laya logits do not match the task's ordered labels")
        return logits


def load_model(device="cuda", checkpoint=None, kind="laya", adaptation="head", lora_rank=0, config=None):
    if checkpoint:
        config = checkpoint_config(torch.load(checkpoint, map_location="cpu", weights_only=True), config)
        kind = config.get("kind", "laya")
    if kind == "laya":
        return DocumentModel(device, checkpoint, lora_rank, config)
    return EncoderModel(device, kind, adaptation, lora_rank, checkpoint, config)
