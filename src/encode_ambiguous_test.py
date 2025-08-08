# uv run src/encode_ambiguous_test.py --strategy rag --experiment_name NACE2025_DATASET --llm_name Qwen/Qwen3-0.6B --third 1 --sample_size 50

import asyncio
import logging
import os

import config
from strategies.base import EncodeStrategy
from strategies.cag import CAGStrategy
from strategies.rag import RAGStrategy
from utils.data import get_ambiguous_data
import mlflow
from constants.paths import URL_SIRENE4_EXTRACTION
from evaluation.evaluator import Evaluator

config.setup()

async def run_encode(
    strategy_cls: EncodeStrategy,
    experiment_name: str,
    run_name: str,
    llm_name: str,
    third: int,
    prompts_from_file: bool,
    prompt_name: str,
    prompt_label: str,
    sample_size: int = None,
):
    logging.info("Define strategy ==========================")
    strategy = strategy_cls(
        generation_model=llm_name,
        prompt_name=prompt_name,
        prompt_label=prompt_label,
    )
    logging.info("Use get_ambiguous_data ==========================")
    data = get_ambiguous_data(strategy.mapping, third, only_annotated=True)
    if sample_size is not None:
        data = data.sample(n=sample_size, random_state=1)
        data = data.reset_index(drop=True)

    data_length = len(data)
    logging.info(f"Must proceed {data_length} prompts")

    logging.info("Get prompts (retrieval) ==========================")
    import time

    start_time = time.time()
    prompts = await strategy.get_prompts(data, load_prompts_from_file=prompts_from_file)
    print(f"Total time: {time.time() - start_time}")
    print(f"Nb of promts: {len(prompts)}")

    logging.info("Prompts retrieved !!! ==========================")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        outputs = strategy.call_llm(prompts, strategy.sampling_params)

        processed_outputs = strategy.process_outputs(outputs)

        results = data.merge(processed_outputs, left_index=True, right_index=True)

        output_path = strategy.save_results(results, third)

        metrics = Evaluator().evaluate(results, prompts)
        metrics["num_coded"] = results["codable"].sum()
        metrics["num_not_coded"] = len(results) - results["codable"].sum()
        metrics["pct_not_coded"] = round((len(results) - results["codable"].sum()) / len(results) * 100, 2)

        # Log MLflow parameters and metrics
        mlflow.log_params(
            {
                "LLM_MODEL": llm_name,
                "TEMPERATURE": strategy.sampling_params.temperature,
                "input_path": URL_SIRENE4_EXTRACTION,
                "output_path": output_path,
                "strategy": "cag" if isinstance(strategy, CAGStrategy) else "rag",
            }
        )

        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["rag", "cag"], required=True)
    parser.add_argument("--experiment_name", type=str, default="Test")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--llm_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--third", type=int, default=None)
    parser.add_argument("--prompts_from_file", action="store_true")
    parser.add_argument("--prompt_name", type=str, default="rag-classifier")
    parser.add_argument("--prompt_label", type=str, default="production")
    parser.add_argument("--sample_size", type=int, default=None)

    args = parser.parse_args()

    # args_list = [
    #     "--strategy",
    #     "rag",
    #     "--experiment_name",
    #     "NACE2025_DATASET",
    #     "--llm_name",
    #     "Qwen/Qwen3-0.6B",
    #     "--third",
    #     "1",
    #     "--sample_size",
    #     "50",
    # ]
    # args = parser.parse_args(args_list)

    assert "MLFLOW_TRACKING_URI" in os.environ, "Set MLFLOW_TRACKING_URI"

    STRATEGY_MAP = {
        "rag": RAGStrategy,
        "cag": CAGStrategy,
    }

    asyncio.run(
        run_encode(
            strategy_cls=STRATEGY_MAP[args.strategy],
            experiment_name=args.experiment_name,
            run_name=args.run_name,
            llm_name=args.llm_name,
            third=args.third,
            prompts_from_file=args.prompts_from_file,
            prompt_name=args.prompt_name,
            prompt_label=args.prompt_label,
            sample_size=args.sample_size,
        )
    )

    # strategy_cls = STRATEGY_MAP[args.strategy]
    # experiment_name = args.experiment_name
    # run_name = args.run_name
    # llm_name = args.llm_name
    # third = args.third
    # prompts_from_file = args.prompts_from_file

    # async def get_prompts(self, data: pd.DataFrame, load_prompts_from_file: bool = False) -> List[List[Dict]]:
    #     tasks = [self.create_prompt(row) for row in data.to_dict(orient="records")]
    #     prompts = await tqdm.gather(*tasks)
    #     return prompts
