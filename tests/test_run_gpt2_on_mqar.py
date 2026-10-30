import pytest
import torch
from transformers import AutoModelForCausalLM, HfArgumentParser, LlamaConfig

from run_gpt2_on_mqar import ExperimentArgs, enable_llama_bidirectional_attention


def _llama_model():
    config = LlamaConfig(
        vocab_size=41,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
    )
    return AutoModelForCausalLM.from_config(config, attn_implementation='eager')


def test_runner_exposes_bidirectional_attention():
    assert ExperimentArgs.__dataclass_fields__['bidirectional_attention'].default is False
    args = HfArgumentParser(ExperimentArgs).parse_args_into_dataclasses([
        '--exp_path', 'unused',
        '--per_device_batch_size', '1',
        '--bidirectional_attention',
    ])[0]
    assert args.bidirectional_attention is True


def test_bidirectional_llama_uses_future_tokens():
    torch.manual_seed(0)
    model = _llama_model().eval()
    sequences = [torch.tensor([[1, 2, 3]]), torch.tensor([[1, 2, 4]])]

    causal_logits = [model(input_ids=sequence).logits[:, 0] for sequence in sequences]
    assert torch.allclose(causal_logits[0], causal_logits[1])

    enable_llama_bidirectional_attention(model)
    bidirectional_logits = [model(input_ids=sequence).logits[:, 0] for sequence in sequences]
    assert not torch.allclose(bidirectional_logits[0], bidirectional_logits[1])


def test_bidirectional_llama_preserves_same_position_loss():
    model = enable_llama_bidirectional_attention(_llama_model().eval())
    input_ids = torch.tensor([[1, 2, 3]])
    labels = torch.tensor([[-100, 7, -100]])
    output = model(input_ids=input_ids, labels=labels)
    expected = torch.nn.functional.cross_entropy(output.logits[:, 1], labels[:, 1])
    assert torch.allclose(output.loss, expected)


def test_bidirectional_attention_rejects_non_llama():
    class NotLlama:
        config = type('Config', (), {'model_type': 'gpt2'})()

    with pytest.raises(ValueError, match='requires a Llama'):
        enable_llama_bidirectional_attention(NotLlama())
