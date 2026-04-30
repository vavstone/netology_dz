import ollama

def chat(model):
    client = ollama.Client(host='http://localhost:11434')
    messages_hist = [{"role":"system","content":"Ты вежливый и умный собеседник. Отвечай исключительно на русском языке."}]
    print(f'#Диалог с LLM {model}')
    print('#Для завершения диалога наберите quit')
    while True:
        request = input("Вы: ")
        if request=="quit":
            break
        print(f'{model}: ')
        messages_hist.append({"role": "user", "content": request})
        stream = client.chat(
            model=model,
            messages=messages_hist,
            stream=True
        )
        response = ''
        for chunk in stream:
            content = chunk["message"]["content"]
            if content:
                response += content
                print(content, end='', flush=True)
        messages_hist.append({"role": "assistant", "content": response})
        print()
        while len(messages_hist) > 21:
            removed_req = messages_hist.pop(1)
            removed_resp = messages_hist.pop(1)
            print()
            print('#Для экономии контекста удалены запрос и ответ:')
            print(f'#Вы:{removed_req["content"]}')
            print(f'#{model}:{removed_resp["content"]}')
            print('#Прошу прощения за доставленные неудобства...')
            print()

chat("gemma3:4b")