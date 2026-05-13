from src import (
    generate_function_call,
    validate_functions,
    validate_prompt,
    load_vocabulary
)
from llm_sdk import Small_LLM_Model  # type: ignore
import argparse
import json
import time
import os


def main() -> None:
    """Entry point for the function calling program."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json"
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json")
    parser.add_argument(
        "--output",
        default="data/output/function_calls.json")
    args = parser.parse_args()
    try:
        model = Small_LLM_Model()
        path = model.get_path_to_vocab_file()
        str_id, id_str = load_vocabulary(path)
        validate_fun = validate_functions(args.functions_definition)
        validate_prom = validate_prompt(args.input)
        if validate_fun is None:
            print("Cannot continue: functions definition file is invalid.")
            return
        if validate_prom is None:
            print("Cannot continue: input file is invalid.")
            return

        results = []
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        for pro in validate_prom:
            data = generate_function_call(
                pro, validate_fun, model, str_id, id_str)
            results.append(data.model_dump())
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
