from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import openai
from openai import OpenAI
import anthropic
# import google.generativeai as genai
import json
import re
import random
from tqdm import tqdm
import time
from datasets import load_dataset
import argparse
import requests
# from ai21 import AI21Client
# from ai21.models.chat import ChatMessage, ResponseFormat, DocumentSchema, FunctionToolDefinition
# from ai21.models.chat import ToolDefinition, ToolParameters

API_KEY = ""

def evaluate_in_parallel(subjects):
    """
    A parallel version of 'evaluate'. This function uses a ThreadPoolExecutor
    to call 'single_request' in parallel for all questions in each subject.
    It references existing functions and data structures from your script
    (like 'single_request', 'update_result', 'merge_result', etc.).
    Modify or merge as appropriate within your script.
    """
    client = get_client()
    test_df, dev_df = load_mmlu_pro()

    # If no subjects specified, evaluate all
    if not subjects:
        subjects = list(test_df.keys())
    print("assigned subjects", subjects)

    for subject in subjects:
        test_data = test_df[subject]
        output_res_path = os.path.join(args.output_dir, subject + "_result.json")
        output_summary_path = os.path.join(args.output_dir, subject + "_summary.json")

        # Load existing results and category stats
        res, category_record = update_result(output_res_path)

        # Identify which questions still need answers
        existing_keys = {(r["question_id"], r["question"]) for r in res}
        needed_data = [q for q in test_data
                       if (q["question_id"], q["question"]) not in existing_keys]

        # Create a list to hold new results
        new_results = []

        # Submit all needed questions to the executor in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_question = {}
            for question in needed_data:
                future = executor.submit(single_request, client, question, dev_df, res)
                future_to_question[future] = question

            # Collect results as they complete
            for future in as_completed(future_to_question):
                question_obj = future_to_question[future]
                pred, response, exist = future.result()

                # If for some reason single_request returns None, handle that
                if response is not None:
                    question_obj["pred"] = pred
                    question_obj["model_outputs"] = response
                else:
                    question_obj["pred"] = None
                    question_obj["model_outputs"] = ""

                new_results.append(question_obj)

        # Merge new results into the existing 'res' and update stats
        for item in new_results:
            label = item["answer"]
            category = item["category"]

            # Initialize category if needed
            if category not in category_record:
                category_record[category] = {"corr": 0.0, "wrong": 0.0}

            # Merge the question
            merge_result(res, item)

            # Tally correctness
            if item["pred"] is not None and item["pred"] == label:
                category_record[category]["corr"] += 1
            else:
                category_record[category]["wrong"] += 1

        # Save updated results & summaries
        save_res(res, output_res_path)
        save_summary(category_record, output_summary_path)


def get_client():
    if args.model_name in ["gpt-4", "gpt-4o", "o1-preview"]:
        openai.api_key = API_KEY
        client = openai
    elif args.model_name in ["deepseek-chat", "deepseek-coder"]:
        client = OpenAI(api_key="None", base_url="http://localhost:8000/v1")
    elif args.model_name in ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest",
                             "gemini-1.5-flash-8b", "gemini-002-pro", "gemini-002-flash"]:
        genai.configure(api_key=API_KEY)
        generation_config = {
            "temperature": 0.0,
            "top_p": 1,
            "max_output_tokens": 4000,
            "response_mime_type": "text/plain",
        }
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
        client = genai.GenerativeModel(
            model_name=args.model_name,
            safety_settings=safety_settings,
            generation_config=generation_config,
        )
    elif args.model_name in ["claude-3-opus-20240229", "claude-3-sonnet-20240229"]:
        client = anthropic.Anthropic(
            api_key=API_KEY,
        )
    elif args.model_name in ["jamba-1.5-large"]:
        client = AI21Client(api_key=API_KEY)
    elif args.model_name in ["iask"]:
        client = {"Authorization": f"Bearer {API_KEY}"}
    else:
        client = None
        print("For other model API calls, please implement the client definition method yourself.")
    return client


def call_api(client, instruction, inputs):
    print("========= Sending a request ==========")
    start = time.time()
    if args.model_name in ["gpt-4", "gpt-4o", "deepseek-chat", "deepseek-coder"]:
        message_text = [{"role": "user", "content": instruction + inputs}]
        completion = client.chat.completions.create(
          model=args.model_name,
          messages=message_text,
          temperature=0.6,
          max_tokens=400000,
        #   top_p=1,
        #   frequency_penalty=0,
        #   presence_penalty=0,
        )
        result = completion.choices[0].message.content
    elif args.model_name in ["o1-preview"]:
        message_text = [{"role": "user", "content": instruction + inputs}]
        completion = client.chat.completions.create(
          model=args.model_name,
          messages=message_text,
        )
        result = completion.choices[0].message.content
    elif args.model_name in ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-1.5-flash-8b"]:
        chat_session = client.start_chat(
            history=[]
        )
        result = chat_session.send_message(instruction + inputs).text
    elif args.model_name in ["claude-3-opus-20240229", "claude-3-sonnet-20240229"]:
        message = client.messages.create(
            model=args.model_name,
            max_tokens=4000,
            system="",
            messages=[
                {"role": "user", "content": instruction + inputs}
            ],
            temperature=0.0,
            top_p=1,
        )
        result = message.content[0].text
    elif args.model_name in ["jamba-1.5-large"]:
        message_text = [ChatMessage(content=instruction + inputs, role="user")]
        completion = client.chat.completions.create(
            model=args.model_name,
            messages=message_text,
            documents=[],
            tools=[],
            n=1,
            max_tokens=2048,
            temperature=0,
            top_p=1,
            stop=[],
            response_format=ResponseFormat(type="text"),
        )
        result = completion.choices[0].message.content
    elif args.model_name in ["iask"]:
        payload = {
            "prompt": instruction + inputs,
            "mode": "truth",
            "detail_level": "detailed",
            "stream": False
        }
        response = requests.post("https://api.iask.ai/v1/query", headers=client, json=payload, timeout=300)
        if response.status_code != 200:
            print("API call failed with status code", response.status_code, response.json())
            return response.json()["response"]["message"]
        else:
            result = response.json()["response"]["message"]
        return result
    else:
        print("For other model API calls, please implement the request method yourself.")
        result = None
    print("cost time", time.time() - start)
    return result


def load_mmlu_pro():
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df


def preprocess(test_df):
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    res = {}
    for each in res_df:
        if each["category"] not in res:
            res[each["category"]] = []
        res[each["category"]].append(each)
    return res


def format_example(question, options, cot_content=""):
    if cot_content == "":
        cot_content = "Let's think step by step."
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]
    example = "Question: {}\nOptions: ".format(question)
    choice_map = "ABCDEFGHIJ"
    for i, opt in enumerate(options):
        example += "{}. {}\n".format(choice_map[i], opt)
    if cot_content == "":
        example += "Answer: "
    else:
        example += "Answer: " + cot_content + "\n\n"
    return example


def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        print("1st answer extract failed\n" + text)
        return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


def single_request(client, single_question, cot_examples_dict, exist_result):
    exist = True
    q_id = single_question["question_id"]
    for each in exist_result:
        if q_id == each["question_id"] and single_question["question"] == each["question"]:
            pred = extract_answer(each["model_outputs"])
            return pred, each["model_outputs"], exist
    exist = False
    category = single_question["category"]
    cot_examples = cot_examples_dict[category]
    question = single_question["question"]
    options = single_question["options"]
    prompt = "The following are multiple choice questions (with answers) about {}. Think step by" \
             " step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n" \
        .format(category)
    for each in cot_examples:
        prompt += format_example(each["question"], each["options"], each["cot_content"])
    input_text = format_example(question, options)
    try:
        response = call_api(client, prompt, input_text)
        response = response.replace('**', '')
    except Exception as e:
        print("error", e)
        return None, None, exist
    pred = extract_answer(response)
    return pred, response, exist


def update_result(output_res_path):
    category_record = {}
    res = []
    success = False
    while not success:
        try:
            if os.path.exists(output_res_path):
                with open(output_res_path, "r") as fi:
                    res = json.load(fi)
                    for each in res:
                        category = each["category"]
                        if category not in category_record:
                            category_record[category] = {"corr": 0.0, "wrong": 0.0}
                        if not each["pred"]:
                            random.seed(12345)
                            x = random.randint(0, len(each["options"]) - 1)
                            if x == each["answer_index"]:
                                category_record[category]["corr"] += 1
                            else:
                                category_record[category]["wrong"] += 1
                        elif each["pred"] == each["answer"]:
                            category_record[category]["corr"] += 1
                        else:
                            category_record[category]["wrong"] += 1
            success = True
        except Exception as e:
            print("Error", e, "sleep 2 seconds")
            time.sleep(2)
    return res, category_record


def merge_result(res, curr):
    merged = False
    for i, single in enumerate(res):
        if single["question_id"] == curr["question_id"] and single["question"] == curr["question"]:
            res[i] = curr
            merged = True
    if not merged:
        res.append(curr)
    return res


def evaluate(subjects):
    client = get_client()
    test_df, dev_df = load_mmlu_pro()
    if not subjects:
        subjects = list(test_df.keys())
    print("assigned subjects", subjects)
    for subject in subjects:
        test_data = test_df[subject]
        output_res_path = os.path.join(args.output_dir, subject + "_result.json")
        output_summary_path = os.path.join(args.output_dir, subject + "_summary.json")
        res, category_record = update_result(output_res_path)

        for each in tqdm(test_data):
            label = each["answer"]
            category = subject
            pred, response, exist = single_request(client, each, dev_df, res)
            if response is not None:
                res, category_record = update_result(output_res_path)
                if category not in category_record:
                    category_record[category] = {"corr": 0.0, "wrong": 0.0}
                each["pred"] = pred
                each["model_outputs"] = response
                merge_result(res, each)
                if pred is not None:
                    if pred == label:
                        category_record[category]["corr"] += 1
                    else:
                        category_record[category]["wrong"] += 1
                else:
                    category_record[category]["wrong"] += 1
                save_res(res, output_res_path)
                save_summary(category_record, output_summary_path)
                res, category_record = update_result(output_res_path)
        save_res(res, output_res_path)
        save_summary(category_record, output_summary_path)


def save_res(res, output_res_path):
    temp = []
    exist_q_id = []
    for each in res:
        if each["question_id"] not in exist_q_id:
            exist_q_id.append(each["question_id"])
            temp.append(each)
        else:
            continue
    res = temp
    with open(output_res_path, "w") as fo:
        fo.write(json.dumps(res))


def save_summary(category_record, output_summary_path):
    total_corr = 0.0
    total_wrong = 0.0
    for k, v in category_record.items():
        if k == "total":
            continue
        cat_acc = v["corr"] / (v["corr"] + v["wrong"])
        category_record[k]["acc"] = cat_acc
        total_corr += v["corr"]
        total_wrong += v["wrong"]
    acc = total_corr / (total_corr + total_wrong)
    category_record["total"] = {"corr": total_corr, "wrong": total_wrong, "acc": acc}
    with open(output_summary_path, "w") as fo:
        fo.write(json.dumps(category_record))

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def evaluate_in_parallel_n(subjects, n=5):
    """
    A parallel version of 'evaluate' that uses a ThreadPoolExecutor with up to 'n' threads.
    It runs 'single_request' for each test sample in parallel and saves results/summary
    as they become available.
    """
    client = get_client()
    test_df, dev_df = load_mmlu_pro()
    print('====== Running in Parallel with threads=', n)
    if not subjects:
        subjects = list(test_df.keys())
    print("assigned subjects", subjects)

    for subject in subjects:
        test_data = test_df[subject]
        output_res_path = os.path.join(args.output_dir, subject + "_result.json")
        output_summary_path = os.path.join(args.output_dir, subject + "_summary.json")

        # Load and update existing results (if any)
        res, category_record = update_result(output_res_path)

        # Use a thread pool to process questions in parallel
        with ThreadPoolExecutor(max_workers=n) as executor:
            future_to_question = {
                executor.submit(single_request, client, each, dev_df, res): each
                for each in test_data
            }

            # Track progress with as_completed and tqdm
            for future in tqdm(as_completed(future_to_question), total=len(test_data)):
                question = future_to_question[future]
                label = question["answer"]
                category = subject

                try:
                    # Attempt to get the result within 180 seconds
                    pred, response, exist = future.result(timeout=200)
                except TimeoutError:
                    # Timed out, so mark this request as wrong
                    res, category_record = update_result(output_res_path)
                    if category not in category_record:
                        category_record[category] = {"corr": 0.0, "wrong": 0.0}

                    category_record[category]["wrong"] += 1

                    # Save summary (and possibly results) immediately after timeout handling
                    save_summary(category_record, output_summary_path)
                    res, category_record = update_result(output_res_path)

                    # Move on to the next future
                    continue


                # # Get results from single_request
                # pred, response, exist = future.result()
                if response is not None:
                    # Refresh local res/category_record
                    res, category_record = update_result(output_res_path)

                    # Update category record if needed
                    if category not in category_record:
                        category_record[category] = {"corr": 0.0, "wrong": 0.0}

                    # Store predictions
                    question["pred"] = pred
                    question["model_outputs"] = response
                    merge_result(res, question)

                    # Update stats
                    if pred is not None and pred == label:
                        category_record[category]["corr"] += 1
                    else:
                        category_record[category]["wrong"] += 1

                    # Save after each finished request
                    save_res(res, output_res_path)
                    save_summary(category_record, output_summary_path)

                    # Reload updated results
                    res, category_record = update_result(output_res_path)

        # Final save/summary after all are processed
        save_res(res, output_res_path)
        save_summary(category_record, output_summary_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", type=str, default="eval_results/")
    parser.add_argument("--model_name", "-m", type=str, default="gpt-4",
                        choices=["gpt-4", "gpt-4o", "o1-preview",
                                 "deepseek-chat", "deepseek-coder",
                                 "gemini-1.5-flash-latest",
                                 "gemini-1.5-pro-latest",
                                 "claude-3-opus-20240229",
                                 "gemini-1.5-flash-8b",
                                 "claude-3-sonnet-20240229",
                                 "gemini-002-pro",
                                 "gemini-002-flash"])
    parser.add_argument("--assigned_subjects", "-a", type=str, default="all")
    parser.add_argument("--num_threads", "-n", type=int, default=1)
    assigned_subjects = []
    args = parser.parse_args()

    if args.assigned_subjects == "all":
        assigned_subjects = []
    else:
        assigned_subjects = args.assigned_subjects.split(",")
    os.makedirs(args.output_dir, exist_ok=True)
    if args.num_threads > 0:
        evaluate_in_parallel_n(assigned_subjects, args.num_threads)
    else:
        evaluate(assigned_subjects)