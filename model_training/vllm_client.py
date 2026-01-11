
import os
import argparse
from openai import OpenAI

def run_client(
    base_url: str = "http://localhost:8000/v1",
    model_name: str = "teachingassistant",
    prompt: str = "你好",
    temperature: float = 0.7,
    max_tokens: int = 512,
    stream: bool = True
):
    """
    vLLM OpenAI-Compatible Client
    """
    # Initialize client
    client = OpenAI(
        base_url=base_url,
        api_key="EMPTY", # vLLM 默认不需要 Key
    )
    
    messages = [
        {"role": "system", "content": "You are a helpful teaching assistant."},
        {"role": "user", "content": prompt}
    ]
    
    print(f"Sending request to {base_url} (model: {model_name})...\n")
    print(f"User: {prompt}\n")
    print("Assistant: ", end="", flush=True)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream 
        )
        
        if stream:
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
            print() # Newline
        else:
            print(response.choices[0].message.content)
            
    except Exception as e:
        print(f"\nError: {e}")
        print("Tip: Make sure the vLLM server is running. (e.g., python -m vllm.entrypoints.openai.api_server ...)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM Inference Client")
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1", help="vLLM API URL")
    parser.add_argument("--model", type=str, default="teachingassistant", help="Model name registered in vLLM")
    parser.add_argument("--prompt", type=str, default="Explaining Newton's Second Law", help="Input prompt")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature")
    
    args = parser.parse_args()
    
    run_client(
        base_url=args.url,
        model_name=args.model,
        prompt=args.prompt,
        temperature=args.temp
    )
