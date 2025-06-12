from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LocalJudge:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        self.device = self.model.device

    def judge(self, question, answer_1, answer_2):
        default_prompt = """We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.
            Please rate the helpfulness, relevance, accuracy, level of details of their responses. 

            Each assistant receiCan you explain the basics of quantum computing?ves an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
            Please first provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment. 
            Then, output two lines indicating the scores for Assistant 1 and 2, respectively.

            Output with the following format:
            Evaluation evidence: <your evaluation explanation here>
            Score of the Assistant 1: <score>
            Score of the Assistant 2: <score>"""
        prompt = f"[Question]\n{question}\n\n[The Start of Assistant 1's Answer]\n{answer_1}\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer_2}\n[The End of Assistant 2's Answer]\n\n[System]\n{default_prompt}\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=2048)
        judgment = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return judgment


if __name__ == "__main__":
    local_judge = LocalJudge(model_path=r"./models/autoj-13b")
    result = local_judge.judge(question="""Can you explain the basics of quantum computing?""", answer_1="""Quantum computing is a type of computing that uses quantum-mechanical phenomena, such as superposition and entanglement, to perform operations on data. Unlike classical computers, which use binary digits (bits) that can only be in one of two states (0 or 1), quantum computers use quantum bits (qubits) that can exist in multiple states simultaneously. This allows quantum computers to perform certain types of calculations much faster than classical computers.
The basic idea of quantum computing is to use qubits to represent the states of the system being modeled, and then manipulate those qubits using quantum algorithms to solve a problem. There are several different approaches to building quantum computers, including:
1. Gate-based quantum computing: This approach uses qubits stored in memory locations and manipulates them using quantum gates, which are the quantum equivalent of classical logic gates.
2. Topological quantum computing: This approach uses qubits that are linked by quantum connections, and manipulates them using topological quantum operations.
3. Adiabatic quantum computing: This approach uses a quantum circuit that evolves gradually over time, and manipulates the qubits using quantum gates at specific points in the evolution.
While quantum computers are still in the early stages of development, they have the potential to solve certain problems much faster than classical computers, and could have a wide range of applications, including cryptography, simulation, optimization, and machine learning.""", answer_2="""Certainly! Quantum computing is a type of computing that uses quantum-mechanical phenomena to process information. 

In classical computing, a bit can exist in one of two states - 0 or 1. In quantum computing, a qubit can exist in a superposition of states, which means it can exist in a combination of 0 and 1 at the same time. This ability to exist in multiple states simultaneously allows for the processing of large amounts of information simultaneously and can significantly speed up certain types of computations.

Quantum computers use a series of quantum gates - which are similar to the logic gates used in classical computers - to manipulate the state of the qubits, allowing for the computations to be completed. There are several different technologies used to build quantum computers, including superconducting quantum processors, trapped-ion quantum processors, and photonic quantum processors.

One of the most important applications of quantum computing is in solving problems that are intractable for classical computers. For example, quantum computers can be used to factor large numbers, which is the basis for many encryption protocols - leading to exciting possibilities in code-breaking, cryptography, and cybersecurity in the future.""")

    with open("output.txt", "w") as f:
        f.write(result)