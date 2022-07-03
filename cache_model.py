from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datetime import datetime

def format_timedelta(td):
    seconds = td.total_seconds()
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    if seconds < 1:
        return "<1 sec"
    return '{} {} {} {}'.format(
            "" if int(days) == 0 else str(int(days)) + ' days',
            "" if int(hours) == 0 else str(int(hours)) + ' hours',
            "" if int(minutes) == 0 else str(int(minutes))  + ' mins',
            "" if int(seconds) == 0 else str(int(seconds))  + ' secs'
        )

    
t1 = datetime.now()
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
print('⌚ Model tokenizer created', format_timedelta(datetime.now()-t1))

t1 = datetime.now()
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
print('⌚ Model loaded (.from_pretrained)', format_timedelta(datetime.now()-t1))

t1 = datetime.now()

model.half().cuda()

print('⌚ Model half().cuda()', format_timedelta(datetime.now()-t1))
