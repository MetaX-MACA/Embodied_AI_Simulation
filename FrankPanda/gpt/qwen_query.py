import time
import json

from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:8300/v1", 
)

def qwen_query(system, user_contents, assistant_contents, model='Qwen2.5-Coder-14B-Instruct', save_path=None, temperature=0.8, debug=False):
    for user_content, assistant_content in zip(user_contents, assistant_contents):
        user_content = user_content.split("\n")
        assistant_content = assistant_content.split("\n")
        
        for u in user_content:
            print(u)
        print("=====================================")
        for a in assistant_content:
            print(a)
        print("=====================================")

    for u in user_contents[-1].split("\n"):
        print(u)

    if debug:
        import pdb; pdb.set_trace()
        return None

    print("=====================================")

    start = time.time()
    
    num_assistant_mes = len(assistant_contents)
    messages = []

    messages.append({"role": "system", "content": "{}".format(system)})
    for idx in range(num_assistant_mes):
        messages.append({"role": "user", "content": user_contents[idx]})
        messages.append({"role": "assistant", "content": assistant_contents[idx]})
    messages.append({"role": "user", "content": user_contents[-1]})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )

    result = response.choices[0].message.content

    end = time.time()
    used_time = end - start

    print(result)
    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump({"used_time": used_time, "res": result, "system": system, "user": user_contents, "assistant": assistant_contents}, f, ensure_ascii=False, indent=4)

    return result


if __name__ == "__main__":
    system = "You are a helpful assistant."
    user_contents = ["Hello",]
    assistant_contents = []
    # save_path = './test_res.json'
    save_path = None
    qwen_query(system, user_contents, assistant_contents, save_path=save_path)

    # response = client.chat.completions.create(
    #     model="Qwen2.5-Coder-14B-Instruct",
    #     messages=[
    #         {"role": "user", "content": "Hello"},
    #     ],
    #     max_tokens=300,
        
    # )
    # result = response.choices[0].message.content
    # print(f"Response: {result}")