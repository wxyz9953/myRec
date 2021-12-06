# A recommendation repository for me

|  Model   | Paper  |
|  ----  | ----  |
| ESSM  | Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate (SIGIR 2018)|
|  MMoE | Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts (KDD 2018) |
| PLE | Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations (RecSys 2020) |
| AITM | Modeling the Sequential Dependence among Audience Multi-step Conversions with Multi-task Learning in Targeted Display Advertising (KDD 2021) |
| FM | Factorization Machines (ICDM 2010) |
| FwFM | Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising (WWW 2018)|
|FiBiNET | FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction (RecSys 2019) |


```python
import argparse
from myrec.config import Config
from myrec.utils.tools import init_environment
from myrec.utils import get_model
from myrec.evaluator import TopKEvaluator


parser = argparse.ArgumentParser(description="Run Model.")
args = parser.parse_args()

config = Config(config_path="config.json" , args=args)
init_environment(config['seed'])

dataset = SomeDataset(config)

model = get_model(config, dataset)

train_data, test_data = dataset.train_data, dataset.test_data
train_instance, _ = dataset.build_instance()
inputs , labels = train_instance

evaluator = TopKEvaluator(config, train_data, test_data)

for e in config["epochs"]:
    model.fit(inputs, labels, batch_size=config['batch_size'], epochs=1, verbose=0, shuffle=True)
    res_dict = evaluator.evaluate(model)

```