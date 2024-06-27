import gradio as gr
import torch
torch.cuda.empty_cache()
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-7b-base")
model = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-7b-base")

# small model for testing
#tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")
#model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama_v1.1")

model = model.to(device)
model.eval()

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [29, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def predict(message, history):
    history_transformer_format = history + [[message, ""]]
    stop = StopOnTokens()

    messages = "".join(["".join(["\n<human>:"+item[0], "\n<bot>:"+item[1]])
                for item in history_transformer_format])

    model_inputs = tokenizer([messages], return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, timeout=20., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_length=500,
        do_sample=True,
        top_p=1.0,
        temperature=0.7,
        repetition_penalty=1.5,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=StoppingCriteriaList([stop])
        )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    partial_message = ""
    for new_token in streamer:
        if new_token != '<':
            partial_message += new_token
            yield partial_message

gr.ChatInterface(predict).launch(server_name='0.0.0.0', server_port=8443, share=True)


