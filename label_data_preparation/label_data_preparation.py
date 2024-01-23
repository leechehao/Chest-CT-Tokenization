import argparse
import json
import re

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="This is annotated data for a Boundary Detection task, coming from Label Studio in JSON format.")
    parser.add_argument("--report_file", type=str, required=True, help="This is a CSV file containing reports and metadata.")
    parser.add_argument("--output_file", type=str, required=True, help="The output results of the Boundary Detection task's annotated data, saved in CSV format.")
    args = parser.parse_args()

    df = pd.read_csv(args.report_file)
    hosp_id_map = {sample["REPORT"]: sample["HOSP_ID"] for _, sample in df.iterrows()}

    event_file = open(args.input_file)
    event_data = json.load(event_file)

    results = []
    for data in event_data:
        report = data["data"]["REPORT"]
        hosp_id = hosp_id_map[report]
        ents_sorted = sorted(data["annotations"][0]["result"], key=lambda x: x["value"]["start"])
        point = None
        tag = None
        for ent in ents_sorted:
            if point is not None and tag != "Others":
                clean_report = re.sub(r"([\[()\]:,])", r" \1 ", report[point:ent["value"]["start"]]).strip()
                clean_report = re.sub(r"\s+", " ", clean_report)
                results.append((clean_report, tag, hosp_id))
            point = ent["value"]["start"]
            tag = ent["value"]["labels"][0]

    df_results = pd.DataFrame(results, columns=["Text", "Tag", "Hosp_id"])
    df_results.to_csv(args.output_file, index=False)


if __name__ == '__main__':
    main()
