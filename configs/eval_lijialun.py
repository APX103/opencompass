from mmengine.config import read_base
from opencompass.models import Mock
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    # choose a list of datasets
    from .datasets.collections.lijialun_mini import datasets


api_meta_template = dict(
    round=[
            dict(role='HUMAN', api_role='user'),
            dict(role='BOT', api_role='assistant', generate=True),
    ],
)

models = [
    dict(abbr='Mock',
        type=Mock, path='Mock',
        max_out_len=2048, max_seq_len=2048, batch_size=8),
]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=8,
        task=dict(type=OpenICLInferTask)),
)