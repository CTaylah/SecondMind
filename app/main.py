import sys
from pathlib import Path
from ingestion.embedding import EmbeddingManager
import agent.ollama_agent as oa

TEST_DIRECTORY = Path("/home/cardell/Documents/BackupAcademic")

def repl():
    embedding_manager = EmbeddingManager("notes_hf_1", TEST_DIRECTORY)


    print("Notes Assistant")
    print("Type ':quit' to exit, ':sources' to show last sources\n")

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            sys.exit(0)

        if not user_input:
            continue

        if user_input == ":quit":
            print("Goodbye.")
            break

        # ---- Query pipeline ----

        context = embedding_manager.make_query(user_input, 5)
        print("\nAssistant:")
        for chunk in oa.ollama_message(context, user_input):
            print(chunk["message"]["content"], end="", flush=True)
        print("\n")


        # print(response.message)



if __name__ == "__main__":
    repl()
