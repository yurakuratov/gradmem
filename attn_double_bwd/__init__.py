from transformers import AttentionInterface, AttentionMaskInterface
from transformers.masking_utils import sdpa_mask

from attn_double_bwd.hvp_manual import hvp_manual
from attn_double_bwd.hvp_semi_manual import hvp_semi_manual
from attn_double_bwd.flashbob_attention import flashbob_attention_forward


AttentionInterface.register("hvp_manual", hvp_manual)
AttentionMaskInterface.register("hvp_manual", sdpa_mask)

AttentionInterface.register("hvp_semi_manual", hvp_semi_manual)
AttentionMaskInterface.register("hvp_semi_manual", sdpa_mask)

# Register the interface without importing Triton kernels until selected.
for name in ("flashbob", "bob"):
    AttentionInterface.register(name, flashbob_attention_forward)
    AttentionMaskInterface.register(name, sdpa_mask)
