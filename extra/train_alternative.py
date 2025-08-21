import math
from transformers import AutoTokenizer, AutoModelForMaskedLM, get_cosine_schedule_with_warmup
import datasets
import torch
from tqdm import tqdm

device = "cuda"
model_name = "distilbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)

# training args
num_epochs = 3
batch_size = 80
max_length = 512
gradient_accumulation_steps = 2
log_steps = 50
mixed_precision = "fp16"

# load and tokenize the dataset
dataset_name = "roneneldan/TinyStories" 
dataset = datasets.load_dataset(dataset_name, split="train")
def tok_fn(examples):
    return tokenizer(examples["text"], max_length=max_length, 
                     padding="max_length", truncation=True, add_special_tokens=False)
tok_dataset = dataset.map(tok_fn, batched=True, remove_columns=["text"])
tok_dataset = tok_dataset.with_format("torch")
dataloader = torch.utils.data.DataLoader(tok_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# define the optimizer and learning rate scheduler
lr = 1e-4
weight_decay = 0.01
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
total_steps = num_epochs * math.ceil(len(dataloader) / gradient_accumulation_steps)
warmup_ratio = 0.05 # 5% warmup
warmup_steps = int(warmup_ratio * total_steps)
lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

model.train()
for epoch in range(num_epochs):
    loss_cumsum = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for step, inputs in enumerate(pbar):
        input_ids = inputs['input_ids']

        # sample timesteps and mask the sequences
        t = torch.rand(batch_size, 1, device=device).clamp_min(1e-4).expand(batch_size, max_length)
        mask = torch.bernoulli(t).bool()
        corrupted = input_ids.masked_fill(mask, tokenizer.mask_token_id)
        labels = input_ids.masked_fill(~mask, -100) # ground truth (ignore all unmasked tokens)

        outputs = model(input_ids=corrupted) # attend to and predict padding tokens too!
        logits = outputs.logits
        per_tok_loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)),
        labels.view(-1), reduction="none", ignore_index=-100).view(batch_size, max_length)
        loss = (per_tok_loss / t).mean() # weight by time step
        (loss / gradient_accumulation_steps).backward()
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
        loss_cumsum += loss.item()
        if (step + 1) % log_steps == 0:
            pbar.set_postfix({"Loss": f"{loss_cumsum / log_steps:.4f}"})
            loss_cumsum = 0