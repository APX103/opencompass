from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import first_capital_postprocess
from opencompass.datasets import LijialunDataset

lijialun_reader_cfg = dict(
    input_columns="input",
    output_column="target",
    train_split='dev')

lijialun_all_sets = [
    "lijialun_data",
]

lijialun_datasets = []
for _name in lijialun_all_sets:
    lijialun_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt="""{{input}}"""
                    ),
                ]),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        )

    lijialun_eval_cfg = dict(
        evaluator=dict(type=AccEvaluator),
        pred_role="BOT",
        pred_postprocessor=dict(type=first_capital_postprocess),
    )

    lijialun_datasets.append(
        dict(
            abbr=f"{_name}",
            type=LijialunDataset,
            path="./data/lijialun/",
            name=_name,
            reader_cfg=lijialun_reader_cfg,
            infer_cfg=lijialun_infer_cfg,
            eval_cfg=lijialun_eval_cfg,
        ))

del _name
