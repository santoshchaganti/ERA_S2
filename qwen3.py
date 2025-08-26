from importlib.metadata import version
import json
import os
from pathlib import Path
from safetensors.torch import load_file
from huggingface_hub import snapshot_download
import torch
import torch.nn as nn
import re
from tokenizers import Tokenizer

pkgs = [
    "huggingface_hub",
    "tokenizers",
    "torch",
]

for p in pkgs:
    print(f"{p} version: {version(p)}")
    

# Architecture
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)
        
    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)
    
class MoEFeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_experts_per_tok = cfg["num_experts_per_tok"]
        self.num_experts = cfg["num_experts"]
        self.gate = nn.Linear(cfg["emb_dim"], cfg["num_experts"], dtype=cfg["dtype"], bias=False)
        
        meta_device = torch.device("meta")
        self.fc1 = nn.ModuleList([
            nn.Linear(
                cfg["emb_dim"], cfg["moe_intermediate_size"],
                dtype=cfg["dtype"], bias=False, device=meta_device)
            for _ in range(cfg["num_experts"])
        ])
        self.fc2 = nn.ModuleList([
            nn.Linear(cfg["emb_dim"], cfg["moe_intermediate_size"],
                      dtype=cfg["dtype"], bias=False, device=meta_device)
            for _ in range(cfg["num_experts"])
        ])
        self.fc3 = nn.ModuleList([
            nn.Linear(cfg["moe_intermediate_size"], cfg["emb_dim"],
                      dtype=cfg["dtype"], bias=False)
            for _ in range(cfg["num_experts"])
        ])
        
    def forward(self, x):
        b, seq_len, embed_dim = x.shape
        scores = self.gate(x)
        topk_scores, topk_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)
        topk_probs = torch.softmax(topk_scores, dim=-1)
        
        expert_outputs = []
        for e in range(self.num_experts):
            hidden = torch.nn.functional.silu(self.fc1[e](x)) * self.fc2[e](x)
            out = self.fc3[e](hidden)
            expert_outputs.append(out.unsqueeze(-2))
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        
        gating_probs = torch.zeros_like(scores)
        for i in range(self.num_experts_per_tok):
            indices = topk_indices[...,i:i+1]
            prob = topk_probs[...,i:i+1]
            gating_probs.scatter_(dim=-1, index=indices, src=prob)
        gating_probs = gating_probs.unsqueeze(-1)
            
        y = (gating_probs*expert_outputs).sum(dim=-2)
        return y
    
class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype

        if self.qwen3_compatible:
            x = x.to(torch.float32)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_dtype)
    
# ROPE: rope's goal is to inject information about the absolute position of a token in a sequence into the attention mechanism.
# It does this by rotating the query and key vectors using sine and cosine functions.
# The angle of rotation depends on the position and the specific dimension of the vector.
def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"
    
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))
    positions = torch.arange(context_length, dtype=dtype)
    angles = positions[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=1)
    
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    
    return cos, sin

def apply_rope(x, cos, sin, offset=0):
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"
    
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2:]
    
    # offset is used for inference when we process sequences longer than the precomputed context_length or for key caching
    cos = cos[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    
    rotated = torch.cat((-x2, x1),dim=-1)
    x_rotated = (x * cos) + (rotated * sin)
    
    return x_rotated.to(dtype=x.dtype)

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"
        
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups
        
        if head_dim is None:
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads
            
        self.head_dim = head_dim
        self.d_out = num_heads * head_dim
        
        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)
        
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None
            
    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        b, num_tokens, _ = x.shape
        
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1,2)
        keys_new = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1,2)
        values_new = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1,2)
        
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys_new = self.k_norm(keys_new)

        queries = apply_rope(queries, cos, sin, offset=start_pos)
        keys_new = apply_rope(keys_new, cos, sin, offset=start_pos)
        
        if cache is not None:
            prev_k, prev_v = cache
            keys = torch.cat([prev_k, keys_new], dim=2)
            values = torch.cat([prev_v, values_new], dim=2)
            next_cache = (keys, values)
        else:
            start_pos = 0
            keys, values = keys_new, values_new
            next_cache = (keys, values)
            
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)
        
        attn_scores = queries @ keys.transpose(2,3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        
        context = (attn_weights @ values).transpose(1,2).reshape(b,num_tokens, self.d_out)
        return self.out_proj(context), next_cache
    
class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in = cfg["emb_dim"],
            num_heads = cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"]
        )
        if cfg["num_experts"] > 0:
            self.ff = MoEFeedForward(cfg)
        else:
            self.ff = FeedForward(cfg)
        
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        
    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        shortcut = x
        x = self.norm1(x)
        x, next_cache = self.att(x, mask, cos, sin, start_pos=start_pos, cache=cache)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut
        
        return x, next_cache
    
class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])
        
        if cfg["head_dim"] is None:
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
        else:
            head_dim = cfg["head_dim"]
            cos, sin = compute_rope_params(
                head_dim = head_dim,
                theta_base=cfg["rope_base"],
                context_length=cfg["context_length"]
            )        
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg
        self.current_pos = 0
    
    def forward(self, in_idx, cache=None):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds
        
        num_tokens = x.shape[1]
        if cache is not None:
            pos_start = self.current_pos
            pos_end = pos_start + num_tokens
            self.current_pos = pos_end
            mask = torch.triu(
                torch.ones(pos_end, pos_end, device=x.device, dtype=torch.bool), diagonal=1
            )[pos_start:pos_end, :pos_end]
        else:
            pos_start=0
            mask = torch.triu(
                torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1
            )
            mask = mask[None, None, :, :]
            
        next_cache = []
        for i, block in enumerate(self.trf_blocks):
            blk_cache = cache.get(i) if cache else None
            x, new_blk_cache = block(x, mask, self.cos, self.sin,
                                     start_pos=pos_start,
                                     cache=blk_cache)
            if cache is not None:
                cache.update(i, new_blk_cache)
            next_cache.append(new_blk_cache)
            
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits

    def reset_kv_cache(self):
        self.current_pos=0
        
class KVCache:
    def __init__(self, n_layers):
        self.cache = [None] * n_layers
    
    def get(self, layer_idx):
        return self.cache[layer_idx]
    
    def update(self, layer_idx, value):
        self.cache[layer_idx] = value
        
    def get_all(self):
        return self.cache
    
    def reset(self):
        for i in range(len(self.cache)):
            self.cache[i]=None
            
            
QWEN3_CONFIG = {
    "vocab_size": 151_936,
    "context_length": 262_144,
    "emb_dim": 2048,
    "n_heads": 32,
    "n_layers": 48,
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 4,
    "rope_base": 10_000_000.0,
    "dtype": torch.bfloat16,
    "num_experts": 128,
    "num_experts_per_tok": 8,
        "moe_intermediate_size": 768,
}


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"\n>$ Device : {device}")

torch.manual_seed(123)
with device:
    model = Qwen3Model(QWEN3_CONFIG)

model(torch.tensor([1, 2, 3]).unsqueeze(0).to(device))

total_params = sum(p.numel() for p in model.parameters())
print(f"\n>$ Total number of parameters: {total_params:,}")

total_params_normalized = total_params - model.tok_emb.weight.numel()
print(f"\n>$ Total number of unique parameters: {total_params_normalized:,}")

def model_memory_size(model, input_dtype = torch.float32):
    total_params = 0
    total_grads = 0
    for param in model.parameters():
        param_size = param.numel()
        total_params += param_size
        if param.requires_grad:
            total_grads += param_size
            
    total_buffers = sum(buf.numel() for buf in model.buffers())
    element_size = torch.tensor(0, dtype=input_dtype).element_size()
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size
    
    total_memory_gb = total_memory_bytes /(1024**3)
    
    return total_memory_gb

print(f"float32 (PyTorch default): {model_memory_size(model, input_dtype=torch.float32):.2f} GB")
print(f"bfloat16: {model_memory_size(model, input_dtype=torch.bfloat16):.2f} GB")

def load_weights_into_qwen(model, param_config, params):
    def assign(left, right, tensor_name="unknown"):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")
        return torch.nn.Parameter(right.clone().detach() if isinstance(right, torch.Tensor) else torch.tensor(right))
    
    model.tok_emb.weight = assign(model.tok_emb.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")
    
    for l in range(param_config["n_layers"]):
        block = model.trf_blocks[l]
        att = block.att
        
        att.W_query.weight = assign(
            att.W_query.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight"
        )
        
        att.W_key.weight = assign(
            att.W_key.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight"
        )
        
        att.W_value.weight = assign(
            att.W_value.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight"
        )
        
        att.out_proj.weight = assign(
            att.out_proj.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"models.layers.{l}.self_attn.o_proj.weight"
        )
        
        if hasattr(att, "q_norm") and att.q_norm is not None:
            att.q_norm.scale = assign(
                att.q_norm.scale,
                params[f"model.layers.{l}.self_attn.q_norm.weight"],
                f"model.layers.{l}.self_attn.q_norm.weight"
            )
        if hasattr(att, "k_norm") and att.k_norm is not None:
            att.k_norm.scale = assign(
                att.k_norm.scale,
                params[f"model.layers.{l}.self_attn.k_norm.weight"],
                f"model.layers.{l}.self_attn.k_norm.weight"
            )

        # Attention layernorm
        block.norm1.scale = assign(
            block.norm1.scale,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight"
        )

        # Feedforward weights
        if "num_experts" in param_config:
            # Load router (gating) weights
            block.ff.gate.weight = assign(
                block.ff.gate.weight,
                params[f"model.layers.{l}.mlp.gate.weight"],
                f"model.layers.{l}.mlp.gate.weight"
            )
            # Load expert weights
            for e in range(param_config["num_experts"]):
                prefix = f"model.layers.{l}.mlp.experts.{e}"
                block.ff.fc1[e].weight = assign(
                    block.ff.fc1[e].weight,
                    params[f"{prefix}.gate_proj.weight"],
                    f"{prefix}.gate_proj.weight"
                )
                block.ff.fc2[e].weight = assign(
                    block.ff.fc2[e].weight,
                    params[f"{prefix}.up_proj.weight"],
                    f"{prefix}.up_proj.weight"
                )
                block.ff.fc3[e].weight = assign(
                    block.ff.fc3[e].weight,
                    params[f"{prefix}.down_proj.weight"],
                    f"{prefix}.down_proj.weight"
                )
                # After assigning weights, move the expert layers from meta to CPU
                block.ff.fc1[e] = block.ff.fc1[e].to("cpu")
                block.ff.fc2[e] = block.ff.fc2[e].to("cpu")
                block.ff.fc3[e] = block.ff.fc3[e].to("cpu")

        else:
            block.ff.fc1.weight = assign(
                block.ff.fc1.weight,
                params[f"model.layers.{l}.mlp.gate_proj.weight"],
                f"model.layers.{l}.mlp.gate_proj.weight"
            )
            block.ff.fc2.weight = assign(
                block.ff.fc2.weight,
                params[f"model.layers.{l}.mlp.up_proj.weight"],
                f"model.layers.{l}.mlp.up_proj.weight"
            )
            block.ff.fc3.weight = assign(
                block.ff.fc3.weight,
                params[f"model.layers.{l}.mlp.down_proj.weight"],
                f"model.layers.{l}.mlp.down_proj.weight"
            )

        block.norm2.scale = assign(
            block.norm2.scale,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight"
        )

    # Final normalization and output head
    model.final_norm.scale = assign(model.final_norm.scale, params["model.norm.weight"], "model.norm.weight")

    if "lm_head.weight" in params:
        model.out_head.weight = assign(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
    else:
        # Model uses weight tying, hence we reuse the embedding layer weights here
        print("Model uses weight tying.")
        model.out_head.weight = assign(model.out_head.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")

repo_id = "Qwen/Qwen3-30B-A3B"  # Original Instruct/Thinking hybrind model
# repo_id = "Qwen/Qwen3-235B-A22B-Instruct-2507"  # New instruct model
# repo_id = "Qwen/Qwen3-30B-A3B-Thinking-2507"  # New thinking model
# repo_id = "Qwen/Qwen3-Coder-30B-A3B-Instruct"  # (Qwen3 Coder Flash)

local_dir = Path(repo_id).parts[-1]
print("start")
repo_dir = snapshot_download(repo_id=repo_id, local_dir=local_dir)
print("hello")
index_path = os.path.join(repo_dir, "model.safetensors.index.json")
with open(index_path, "r") as f:
    index = json.load(f)

weights_dict = {}
for filename in set(index["weight_map"].values()):
    shard_path = os.path.join(repo_dir, filename)
    shard = load_file(shard_path)
    weights_dict.update(shard)

load_weights_into_qwen(model, QWEN3_CONFIG, weights_dict)
model.to(device);

class Qwen3Tokenizer:
    _SPECIALS = [
        "<|endoftext|>",
        "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>",
        "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>",
        "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
    ]
    _SPLIT_RE = re.compile(r"(<\|[^>]+?\|>)")

    def __init__(self, tokenizer_file_path="tokenizer.json", repo_id=None,
                 apply_chat_template=True, add_generation_prompt=False, add_thinking=False):

        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking

        tok_file = Path(tokenizer_file_path)
        self._tok = Tokenizer.from_file(str(tok_file))
        self._special_to_id = {t: self._tok.token_to_id(t) for t in self._SPECIALS}

        self.pad_token_id = self._special_to_id.get("<|endoftext|>")
        self.eos_token_id = self.pad_token_id

        if repo_id and "Base" not in repo_id:
            eos_token = "<|im_end|>"
        else:
            eos_token = "<|endoftext|>"
        if eos_token in self._special_to_id:
            self.eos_token_id = self._special_to_id[eos_token]

    def encode(self, text, chat_wrapped=None):
        if chat_wrapped is None:
            chat_wrapped = self.apply_chat_template

        stripped = text.strip()
        if stripped in self._special_to_id and "\n" not in stripped:
            return [self._special_to_id[stripped]]

        if chat_wrapped:
            text = self._wrap_chat(text)

        ids = []
        for part in filter(None, self._SPLIT_RE.split(text)):
            if part in self._special_to_id:
                ids.append(self._special_to_id[part])
            else:
                ids.extend(self._tok.encode(part).ids)
        return ids

    def decode(self, ids):
        return self._tok.decode(ids, skip_special_tokens=False)

    def _wrap_chat(self, user_msg):
        s = f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        if self.add_generation_prompt:
            s += "<|im_start|>assistant"
            if self.add_thinking:
                s += "\n"
            else:
                s += "\n<think>\n\n</think>\n\n"
        return s

tokenizer_file_path = f"{Path(repo_id).parts[-1]}/tokenizer.json"
tokenizer = Qwen3Tokenizer(
    tokenizer_file_path=tokenizer_file_path,
    repo_id=repo_id,
    add_generation_prompt=True,
    add_thinking=True
)

prompt = "Implement a binary search function in Python"

input_token_ids = tokenizer.encode(prompt)
text = tokenizer.decode(input_token_ids)
text

def generate_text_basic_stream(model, token_ids, max_new_tokens, eos_token_id=None, context_size=None):
    model.eval()

    with torch.no_grad():
        cache = KVCache(n_layers=model.cfg["n_layers"])
        model.reset_kv_cache()

        # Prime the cache with the initial context
        logits = model(token_ids, cache=cache)

        for _ in range(max_new_tokens):
            next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)

            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

            yield next_token

            token_ids = torch.cat([token_ids, next_token], dim=1)

            # Feed only the new token to the model; cache handles history
            logits = model(next_token, cache=cache)

input_token_ids_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)


for token in generate_text_basic_stream(
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=200,
    eos_token_id=tokenizer.eos_token_id
):
    token_id = token.squeeze(0).tolist()
    print(
        tokenizer.decode(token_id),
        end="",
        flush=True
    )