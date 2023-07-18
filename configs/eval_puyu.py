from mmengine.config import read_base
from opencompass.models import PUYU
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    # choose a list of datasets
    from .datasets.collections.chat_medium import datasets
    # and output the results in a choosen format
    from .summarizers.medium import summarizer


api_meta_template = dict(
    round=[
            dict(role='user', api_role='user'),
            dict(role='assistant', api_role='assistant', generate=True),
    ],
)

models = [
    dict(abbr='InternLM-Chat V0.2.8',
        type=PUYU, path='InternLM-Chat V0.2.8',
        key='289817',  # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
        meta_template=api_meta_template,
        query_per_second=1,
        max_out_len=2048, max_seq_len=2048, batch_size=8),
]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=8,
        task=dict(type=OpenICLInferTask)),
)
