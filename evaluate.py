#region Imports
import csv
from pathlib import Path
from langchain_remedy import find_remedy
from langgraph_remedy import get_remedy_graph
#endregion

#region Setup
graph = get_remedy_graph()
#endregion

#region Helpers
def read_test_cases(csv_path):
    """
    Read test cases from a CSV file and return them as a list of dictionaries.

    Args:
        csv_path (str): Path to the CSV file containing test case data.

    Returns:
        List[Dict[str, str]]: A list of rows from the CSV file, where each row 
        is represented as a dictionary with column headers as keys.

    Raises:
        FileNotFoundError: If the provided CSV file path does not exist.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return [row for row in reader]
#endregion

#region Processing test cases
remedies_array = []
total_cases = 0
langchain_remedies_found = 0
langgraph_remedies_found = 0

cases = read_test_cases("remedy_test_cases.csv")

for case in cases:
    # Run through LangChain pipeline
    langchain_remedy = find_remedy(case["ailment_description"], case["remedy_type"], case["body_type"])
    found_using_langchain = not ("No remedy found" in langchain_remedy)

    # Prepare initial state for LangGraph
    input_state = {
        "ailment_description": case["ailment_description"],
        "body_type": case["body_type"],
        "remedy_type": case["remedy_type"],
        "context": "",
        "response": "",
        "is_specific": False,
        "stored_remedy_type": case["remedy_type"]
    }

    # Run through LangGraph pipeline
    langgraph_remedy = graph.invoke(input_state)
    # "None" is the terminal sentinel meaning no remedy found after all fallbacks
    found_using_langgraph = not ("None" in langgraph_remedy["response"])

    # Store only first 100 chars of each remedy preview to keep CSV compact
    remedies_array.append((
        case["ailment_description"],
        case["body_type"],
        case["remedy_type"],
        langchain_remedy[:100],
        langgraph_remedy["response"][:100]
    ))

    total_cases += 1
    langchain_remedies_found += int(found_using_langchain)
    langgraph_remedies_found += int(found_using_langgraph)
    print("Working on remedy for ailment description ", total_cases)
#endregion

#region Summary output
print("Summary:")
print(f"  Total cases              : {total_cases}")
print(f"  LangChain remedies found : {langchain_remedies_found}")
print(f"  LangGraph remedies found : {langgraph_remedies_found}")
#endregion

#region Save results
out_path = Path("results_compare.csv")

with out_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    # (desc, body, remedy, lc_preview, lg_preview)
    w.writerow(["ailment_description", "body_type", "remedy_type",
                "langchain_preview", "langgraph_preview"])  # header
    w.writerows(remedies_array)

print(f"Saved to {out_path.resolve()}")
#endregion
