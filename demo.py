from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from rich.live import Live # rich will help visualize diffusion sampling
from rich.console import Console

device = 'cuda'
model_name = "distilbert-diffusion-TinyStories"
model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

seq_len = 512
num_steps = 512 # can increase for better quality
times = torch.linspace(1, 0, num_steps + 1, device=device) # linear reverse process time steps

# initialize the fully masked sequence
x = torch.full((1, seq_len), tokenizer.mask_token_id, dtype=torch.int64, device=device, requires_grad=False)
mask = torch.ones((1, seq_len), dtype=torch.bool, device=device, requires_grad=False)
attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device, requires_grad=False) # attend to all tokens

# sampling process based on Algorithm 4 from https://arxiv.org/abs/2502.09992
model.eval()
console = Console()
with torch.no_grad():
    with Live("", refresh_per_second=10, console=console) as live:
        for t, s in zip(times[:-1], times[1:]):
            logits = model(x, attention_mask=attention_mask).logits
            x[mask] = logits[mask].argmax(-1) # greedily predict the masked tokens
            decoded = tokenizer.batch_decode(x, skip_special_tokens=True)[0]
            live.update(decoded)

            remask_probs = torch.rand((1, seq_len), device=device) < s/t
            mask = mask & remask_probs
            x[mask] = tokenizer.mask_token_id # remask each of the predicted tokens with probability s/t