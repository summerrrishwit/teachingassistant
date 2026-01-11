
import argparse
import ollama

def run_client(model: str, prompt: str):
    """
    Ollama API Client
    """
    print(f"Chatting with local Ollama model: {model}...")
    print(f"User: {prompt}\n")
    print("Assistant: ", end="", flush=True)

    try:
        stream = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            stream=True,
        )

        for chunk in stream:
            content = chunk['message']['content']
            print(content, end='', flush=True)
            
        print() # Newline

    except Exception as e:
        print(f"\nError: {e}")
        print("Tip: Make sure Ollama is running and the model is created. (ollama serve / ollama create ...)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ollama Inference Client")
    parser.add_argument("--model", type=str, default="teachingassistant", help="Model name in Ollama")
    parser.add_argument("--prompt", type=str, default="Explain quantum entanglement simply", help="Input prompt")
    
    args = parser.parse_args()
    run_client(model=args.model, prompt=args.prompt)
