import os
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def collect_prm800_data(data_type='prm'):

    with open(prm800_data_path, "r") as f:
        dataset = json.load(f)

    merged_data = []
    print(f"Processing {len(dataset)} data...")
    
    for i, data in enumerate(dataset):

        # make sure data exists
        if data and isinstance(data, dict):
            problem = data["problem"]
            partial_solution = data["partial_solution"]
            correct_step = data["correct_last_step"]
            incorrect_step = data["incorrect_last_step"]

            data_id = str(data["id"])
            merged_data.append({"problem": problem, "partial_solution": partial_solution, "next_step": correct_step, f"{data_type}_human_label": 1, "file_name": data_id+"_c.json"})
            merged_data.append({"problem": problem, "partial_solution": partial_solution, "next_step": incorrect_step, f"{data_type}_human_label": -1, "file_name": data_id+"_i.json"})

        else:
            print(f"line {i} does not contain valid data.")

    # filter existing files
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            lines = f.readlines()
            existing_files = [json.loads(line)["file_name"] for line in lines]
        merged_data = [data for data in merged_data if data["file_name"] not in existing_files]
        print(f"should process {len(merged_data)} files.")

    return merged_data

def collect_process_bench_data():

    with open(process_bench_data_path, "r") as f:
        dataset = json.load(f)

    print(f"All {len(dataset)} data...")

    # filter existing files
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            lines = f.readlines()
            existing_files = [json.loads(line)["file_name"] for line in lines]
        remaining_data = [data for data in dataset if data["file_name"] not in existing_files]
    else:
        remaining_data = dataset

    print(f"should process {len(remaining_data)} files.")
    return remaining_data

def collect_delta_bench_data():

    with open(delta_bench_data_path, "r") as f:
        dataset = json.load(f)

    print(f"All {len(dataset)} data...")

    # filter existing files
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            lines = f.readlines()
            existing_files = [json.loads(line)["file_name"] for line in lines]
        remaining_data = [data for data in dataset if data["file_name"] not in existing_files]
    else:
        remaining_data = dataset

    print(f"should process {len(remaining_data)} files.")
    return remaining_data

            
def generate_critique_batch(dataset):
    """
    Inputs: problem, partial solution, next step
    Outputs: label and critique
    Vesrion 2
    """

    system = """You are a helpful critic. Given a math Problem, a Partial Solution, the current Last Step of the solution, You need to provide a critique for the correctness of the Last Step.

You need to response a step-by-step analysis:
1. Analyzing the general thought of the Partial Solution. 
2. Critique. You should write a brief critique here. This part should also maintain logical coherence with the summary of the general thought of the Partial Solution.
3. Conclusion. At the end of the response, output <Answer>Correct</Answer> or <Answer>Incorrect</Answer> to represent the correctness of the Last Step."""
    
    prompt_template = """## Problem
{problem}

## Partial Solution
{partial_solution}

## Last Step
{last_step}"""


    llm = LLM(model=critic_path, gpu_memory_utilization=.95, tensor_parallel_size=2)
    # llm = LLM(model=critic_path, gpu_memory_utilization=.95, tensor_parallel_size=4)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=2048, n=1)
    batch_size = 512
    
    all_inputs = []
    for data in dataset:
        if data["partial_solution"] == "":
            data["partial_solution"] = "Let's solve this problem."

        prompt = prompt_template.format(problem=data["problem"], partial_solution=data["partial_solution"], last_step=data["next_step"])

        messages = [
            {"role": "system", "content": system},
            {"role": "user","content": prompt},
        ]
        input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        all_inputs.append(input)
        
    for i in tqdm.tqdm(range(0, len(all_inputs), batch_size)):
        
        # print(f"save_name: {output_file}, batch: {i}")
        batch_data = dataset[i:i+batch_size]
        batch_inputs = all_inputs[i:i+batch_size]
        batch_outputs = llm.generate(batch_inputs, sampling_params)
        
        for data, input, output in zip(batch_data, batch_inputs, batch_outputs):
            output_samples = output.outputs
            critiques = [o.text for o in output_samples]
            save_data = data.copy()
            save_data["system"] = system
            save_data["prompt"] = input
            save_data["response"] = critiques
            
            with open(os.path.join(output_file), "a", encoding='utf-8') as f:
                f.write(json.dumps(save_data, ensure_ascii=False)+"\n")
            
    return True
       

def filter_critique(data_type='prm'):
        
    with open(output_file, "r") as f:
        lines = [json.loads(l) for l in f.readlines()]
    lines = sorted(lines, key=lambda x: x["file_name"])
    
    data_total_num = len(lines)
    data_valid_num = 0
    correct_step_acc = []
    erroneous_step_acc = []
    
    for data in lines:
     
        correctness_label = 0
        response = data["response"][0]  # only one response
        
        start_tag = "<Answer>"
        end_tag = "</Answer>"
        start_index = response.find(start_tag) + len(start_tag)
        end_index = response.find(end_tag)
        result = response[start_index:end_index]
        if result.lower() == "correct":
            correctness_label = 1
            data_valid_num += 1
        elif result.lower() == "incorrect":
            correctness_label = -1
            data_valid_num += 1
            
        if correctness_label != 0:  # valid data
            if data[f"{data_type}_human_label"] == 1:  # correct step
                if correctness_label == 1:
                    correct_step_acc.append(1)
                else:
                    correct_step_acc.append(0)
            elif data[f"{data_type}_human_label"] == -1:  # erroneous step
                if correctness_label == -1:
                    erroneous_step_acc.append(1)
                else:
                    erroneous_step_acc.append(0)
              
    correct_acc = sum(correct_step_acc)/len(correct_step_acc)
    error_acc = sum(erroneous_step_acc)/len(erroneous_step_acc)
    print(f"File: {output_file}")
    print(f"Total number of data: {data_total_num}")
    print(f"Valid rato: {data_valid_num/data_total_num}")  # We will collect the invalid data and rerun them several times if valid ratio < 99%
    print(f"Correct Step Accuracy: {round(correct_acc, 3)}")
    print(f"Erroneous Step Accuracy: {round(error_acc, 3)}")
    print(f"Average Accuracy: {round((correct_acc+error_acc)/2, 3)}")
    print(f"Harmonic Mean: {round(2*correct_acc*error_acc/(correct_acc+error_acc), 3)}")


def filter_process_bench_critique(data_type='process_bench'):
        
    with open(output_file, "r") as f:
        lines = [json.loads(l) for l in f.readlines()]
    
    data_total_num = len(lines)
    data_valid_num = 0
    stat_dataset_acc = {}
    
    for data in lines:
        correctness_label = 0
        response = data["response"][0]  # only one response
        
        start_tag = "<Answer>"
        end_tag = "</Answer>"
        start_index = response.find(start_tag) + len(start_tag)
        end_index = response.find(end_tag)
        result = response[start_index:end_index]

        if result.lower() == "correct":
            correctness_label = 1
            data_valid_num += 1
        elif result.lower() == "incorrect":
            correctness_label = -1
            data_valid_num += 1
            
        # statistics of dataset accuracy
        if correctness_label != 0:  # valid data
            dataset_type = data["data_type"]
            if dataset_type not in stat_dataset_acc:
                stat_dataset_acc[dataset_type] = []

            if data[f"{data_type}_human_label"] == correctness_label:
                stat_dataset_acc[dataset_type].append(1)
            else:
                stat_dataset_acc[dataset_type].append(0)
   
    print(f"File: {output_file}")
    print(f"Total number of data: {data_total_num}")
    print(f"Valid rato: {data_valid_num/data_total_num}") # We will collect the invalid data and rerun them several times if valid ratio < 99%
    
    # print detailed statistics of accuracy
    average_acc = []
    for dataset_type, acc in stat_dataset_acc.items():
        if acc:
            average_acc.append(sum(acc)/len(acc))
            print(f"Dataset Type: {dataset_type}. Accuracy: {round(sum(acc)/len(acc), 3)}.")
    print(f"Average Accuracy: {round(sum(average_acc)/len(average_acc), 3)}")
  

if __name__ == "__main__":
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen2.5-7B-Instruct")
    
    prm800_data_path = "path/to/your/data"
    process_bench_data_path = "path/to/your/data"
    delta_bench_data_path = "path/to/your/data"
    critic_path = "path/to/your/model"

    output_file = os.path.join(critic_path, "critique_process_bench.json")
    dataset = collect_process_bench_data()
    if len(dataset) > 0:
        generate_critique_batch(dataset)
    filter_process_bench_critique(data_type='process_bench')
 
    output_file = os.path.join(critic_path, "critique_prm800.json")
    dataset = collect_prm800_data()
    if len(dataset) > 0:
        generate_critique_batch(dataset)
    filter_critique(data_type='prm')

    output_file = os.path.join(critic_path, "critique_delta_bench.json")
    dataset = collect_delta_bench_data()
    if len(dataset) > 0:
        generate_critique_batch(dataset)
    filter_critique(data_type='delta_bench')

 