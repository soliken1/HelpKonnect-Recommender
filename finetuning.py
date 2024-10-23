from openai import OpenAI
client = OpenAI()

file = client.files.create(
  file=open("data/dataset_finetune.jsonl", "rb"),
  purpose="fine-tune"
)

client.fine_tuning.jobs.create(
  training_file=file.id,
  model="gpt-4o-mini-2024-07-18"
)

