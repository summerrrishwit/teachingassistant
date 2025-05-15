import os
from config import FRAME_DIR, prompt_dict
import base64
import ollama


def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def get_response(text, question, full_transcript, prompt_key, summarized_transcript):
    # Prepare the prompt
    # try:
    #     relevant_chunk = query_faiss_index(question, top_k=1)[0]
    # except Exception as e:
    #     relevant_chunk = full_transcript
    prompt = prompt_dict[prompt_key]

    prompt_inputs = []
    for path in os.listdir(FRAME_DIR):
        if path.endswith(".jpg"):
            image_path = os.path.join(FRAME_DIR, path)
            prompt_inputs.append(image_to_base64(image_path))

    if prompt_key == "video_qa":
        prompt = prompt.format(text=text, question=question, global_context=summarized_transcript)
    elif prompt_key == "bullet_points":
        # Use the full transcript for bullet points
        prompt = prompt.format(text=full_transcript)
    elif prompt_key == "qa_style":
        # Use the full transcript for question-answer pairs
        prompt = prompt.format(text=full_transcript)

    response = ollama.generate(
        model="gemma3:4b",
        prompt = prompt_key,
        images = prompt_inputs)
    
    answer = response['response'].strip()
    
    return answer