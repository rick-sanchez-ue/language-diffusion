import math
from transformers import AutoTokenizer, AutoModelForMaskedLM, get_cosine_schedule_with_warmup
import datasets
import torch
from accelerate import Accelerator
from tqdm import tqdm

model_name = "distilbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

num_epochs = 3
batch_size = 80 
seq_len = 512
gradient_accumulation_steps = 1
log_steps = 50
mixed_precision = "fp16"
lr = 1e-4
weight_decay = 0.01
warmup_ratio = 0.05 # 5% warmup steps

dataset_name = "roneneldan/TinyStories"
dataset = datasets.load_dataset(dataset_name, split="train")
def tok_fn(examples):
    return tokenizer(examples["text"], max_length=seq_len, 
                     padding="max_length", truncation=True, add_special_tokens=False)
tok_dataset = dataset.map(tok_fn, batched=True, remove_columns=["text"])
tok_dataset = tok_dataset.with_format("torch")
dataloader = torch.utils.data.DataLoader(tok_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
total_steps = num_epochs * math.ceil(len(dataloader) / gradient_accumulation_steps)
lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(warmup_ratio * total_steps), num_training_steps=total_steps)

accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, mixed_precision=mixed_precision)
model, optimizer, dataloader, lr_scheduler = accelerator.prepare(model, optimizer, dataloader, lr_scheduler)

# training loop adapted from Algorithms 1 & 2 from https://arxiv.org/abs/2502.09992
model.train()
for epoch in range(num_epochs):
    loss_cumsum = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not accelerator.is_main_process)
    for step, inputs in enumerate(pbar):
        input_ids = inputs['input_ids']
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, 
                                    device=accelerator.device) # attend to all tokens
        # sample t and randomly mask each token with probability t
        t = torch.rand(batch_size, 1, device=accelerator.device).clamp_min(1e-4).expand(batch_size, seq_len)
        mask = torch.bernoulli(t).bool()
        corrupted = input_ids.masked_fill(mask, tokenizer.mask_token_id)
        labels = input_ids.masked_fill(~mask, -100) # compute loss only on masked tokens
        with accelerator.accumulate(model):
            outputs = model(input_ids=corrupted, attention_mask=attention_mask)
            logits = outputs.logits
            per_tok_loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)),
                labels.view(-1), reduction="none", ignore_index=-100).view(batch_size, seq_len)
            loss = (per_tok_loss / t).mean() # weight by time step
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            loss_cumsum += accelerator.gather(loss.detach()).mean().item()
            if accelerator.is_main_process and (step + 1) % log_steps == 0:
                pbar.set_postfix({"Loss": f"{loss_cumsum / log_steps:.4f}"})
                loss_cumsum = 0
accelerator.end_training()

save_directory = "distilbert-diffusion-TinyStories"
accelerator.unwrap_model(model).save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)