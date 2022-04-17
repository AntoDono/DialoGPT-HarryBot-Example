import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("AntoDono/DialoGPT-Harry")

model = AutoModelForCausalLM.from_pretrained("AntoDono/DialoGPT-Harry")

started = False
debug = False
chat_history_token_display = False

while True:
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(
        input(">> User:") + tokenizer.eos_token, return_tensors='pt')
    # print(new_user_input_ids)

    if (debug):
        print("==User_input_ids==")
        print(new_user_input_ids)
        print(new_user_input_ids.shape)
        print()
    # append the new user input tokens to the chat history
    if not started:
        started = True
        bot_input_ids = new_user_input_ids
    else:
        # Once the chat history goes over 50 token the bot dies
        if (bot_input_ids.shape[0]*bot_input_ids.shape[1] + new_user_input_ids.shape[0] * new_user_input_ids.shape[1] > 50):
            bot_input_ids = new_user_input_ids
        else:
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)

    if (debug):
        print("==Bot_input_ids==")
        print(bot_input_ids)
        print(bot_input_ids.shape)
        print()

    # generated a response while limiting the total chat history to 1000 tokens,
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=200,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=100,
        top_p=0.7,
        temperature=0.8
    )

    if (chat_history_token_display):
        print(f"Chat history Token: {bot_input_ids.shape[0]*bot_input_ids.shape[1] + new_user_input_ids.shape[0] * new_user_input_ids.shape[1]}")

    # pretty print last ouput tokens from bot
    print("HarryBot: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
