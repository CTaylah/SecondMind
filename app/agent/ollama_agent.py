import ollama


def ollama_message(context, query):

    prompt = f"""
        You are a personal notes assistant.

        Authority rules:
        - The provided context documents are the primary source of truth.
        - If the context documents contain relevant information, use them.
        - You MAY use general background knowledge to add explanation or fill gaps,
            but you MUST NOT contradict the context.
        - If you use background knowledge, clearly label it as "Background knowledge".
        - If the answer cannot be determined from context or background knowledge,
        say "I don't know".

        Context:
        {context}

        Question:
        {query}
    """

    response = ollama.chat(
        model="llama3.2:1b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2},
        stream=True
    )
    return response