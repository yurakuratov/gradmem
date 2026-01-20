from transformers import AttentionInterface, AttentionMaskInterface
from transformers.masking_utils import sdpa_mask

from attn_double_bwd.jvp_attention import jvp_flash
from attn_double_bwd.hvp_manual import hvp_manual
from attn_double_bwd.hvp_semi_manual import hvp_semi_manual


AttentionInterface.register("jvp_flash", jvp_flash)
AttentionMaskInterface.register("jvp_flash", sdpa_mask)

AttentionInterface.register("hvp_manual", hvp_manual)
AttentionMaskInterface.register("hvp_manual", sdpa_mask)

AttentionInterface.register("hvp_semi_manual", hvp_semi_manual)
AttentionMaskInterface.register("hvp_semi_manual", sdpa_mask)
