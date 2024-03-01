import re
import argparse

import mlflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking_uri", type=str, required=True, help="指定 MLflow 追蹤伺服器的 URI。")
    parser.add_argument("--run_id", type=str, required=True, help="MLflow 實驗運行的唯一標識符。")
    parser.add_argument("--text", type=str, required=True, help="待分詞的文本。")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    model = mlflow.pytorch.load_model(f"runs:/{args.run_id}/model")

    clean_text = re.sub(r"([\[()\]:,])", r" \1 ", args.text).strip()
    clean_text = re.sub(r"\s+", " ", clean_text)
    ents = model.hf_pipeline(clean_text)

    shift = 0
    results = list(clean_text)
    for ent in ents:
        results.insert(ent["start"] + shift, " ")
        shift += 1
    print("".join(results))


if __name__ == "__main__":
    main()
