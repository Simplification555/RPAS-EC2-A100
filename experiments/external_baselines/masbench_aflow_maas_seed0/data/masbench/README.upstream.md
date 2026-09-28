---
pretty_name: MASBench
license: apache-2.0
task_categories:
- text-generation
dataset_info:
- config_name: breadth
  features:
  - name: reward_model_json
    dtype: string
  - name: extra_info_json
    dtype: string
  - name: prompt_json
    dtype: string
  - name: axis
    dtype: string
  - name: value
    dtype: string
  - name: data
    dtype: string
  splits:
  - name: test
    num_bytes: 6403466
    num_examples: 676
  - name: train
    num_bytes: 13462202
    num_examples: 2000
  download_size: 4385649
  dataset_size: 19865668
- config_name: combine
  features:
  - name: reward_model_json
    dtype: string
  - name: extra_info_json
    dtype: string
  - name: prompt_json
    dtype: string
  - name: axis
    dtype: string
  - name: data
    dtype: string
  splits:
  - name: train
    num_bytes: 238142438
    num_examples: 22948
  - name: test
    num_bytes: 0
    num_examples: 0
  download_size: 47223643
  dataset_size: 238142438
- config_name: depth
  features:
  - name: reward_model_json
    dtype: string
  - name: extra_info_json
    dtype: string
  - name: prompt_json
    dtype: string
  - name: axis
    dtype: string
  - name: value
    dtype: string
  - name: data
    dtype: string
  splits:
  - name: test
    num_bytes: 16033934
    num_examples: 1609
  - name: train
    num_bytes: 23788358
    num_examples: 3993
  download_size: 8853769
  dataset_size: 39822292
- config_name: horizon
  features:
  - name: reward_model_json
    dtype: string
  - name: extra_info_json
    dtype: string
  - name: prompt_json
    dtype: string
  - name: axis
    dtype: string
  - name: value
    dtype: string
  - name: data
    dtype: string
  splits:
  - name: test
    num_bytes: 15399660
    num_examples: 702
  - name: train
    num_bytes: 32345806
    num_examples: 2174
  download_size: 9702198
  dataset_size: 47745466
- config_name: parallel
  features:
  - name: reward_model_json
    dtype: string
  - name: extra_info_json
    dtype: string
  - name: prompt_json
    dtype: string
  - name: axis
    dtype: string
  - name: value
    dtype: string
  - name: data
    dtype: string
  splits:
  - name: test
    num_bytes: 4717494
    num_examples: 396
  - name: train
    num_bytes: 23940551
    num_examples: 1807
  download_size: 6433083
  dataset_size: 28658045
- config_name: robustness
  features:
  - name: reward_model_json
    dtype: string
  - name: extra_info_json
    dtype: string
  - name: prompt_json
    dtype: string
  - name: axis
    dtype: string
  - name: value
    dtype: string
  - name: data
    dtype: string
  splits:
  - name: test
    num_bytes: 26077436
    num_examples: 1200
  - name: train
    num_bytes: 33893634
    num_examples: 3000
  download_size: 9434830
  dataset_size: 59971070
configs:
- config_name: breadth
  data_files:
  - split: test
    path: breadth/test-*
  - split: train
    path: breadth/train-*
- config_name: combine
  data_files:
  - split: train
    path: combine/train-*
  - split: test
    path: combine/test-*
- config_name: depth
  data_files:
  - split: test
    path: depth/test-*
  - split: train
    path: depth/train-*
- config_name: horizon
  data_files:
  - split: test
    path: horizon/test-*
  - split: train
    path: horizon/train-*
- config_name: parallel
  data_files:
  - split: test
    path: parallel/test-*
  - split: train
    path: parallel/train-*
- config_name: robustness
  data_files:
  - split: test
    path: robustness/test-*
  - split: train
    path: robustness/train-*
---

# 🎼 MAS-Orchestra: Understanding and Improving Multi-Agent Reasoning Through Holistic Orchestration and Controlled Benchmarks

This is the proposed **MAS evaluation data** used in the recipe described in our paper:
📄 [*MAS-Orchestra: Understanding and Improving Multi-Agent Reasoning Through Holistic Orchestration and Controlled Benchmarks*](https://arxiv.org/abs/2601.14652)

For more details, please check the following resources:

- 🌐 **Project Page:** [https://mas-orchestra.salesforceresearch.ai/mas_r1/index.html](https://mas-orchestra.salesforceresearch.ai/mas_r1/index.html)
- 📚 **Live Demo:** [https://mas-orchestra.salesforceresearch.ai/mas_r1/demo](https://mas-orchestra.salesforceresearch.ai/mas_r1/demo)
- 💻 **Code Repository:** [https://github.com/SalesforceAIResearch/MAS-Orchestra](https://github.com/SalesforceAIResearch/MAS-Orchestra)


### Ethical Considerations
Users need to make their own assessment regarding any obligations or responsibilities under the corresponding licenses or terms and conditions pertaining to the original datasets and data. This release is for research purposes only in support of an academic paper.


## Citation

If you find our project helpful, please consider citing our paper 😊

```
@misc{Ke2026MASOrchestra,
        title        = {MAS-Orchestra: Understanding and Improving Multi-Agent Reasoning Through Holistic Orchestration and Controlled Benchmarks},
        author       = {Zixuan Ke and Yifei Ming and Austin Xu and Ryan Chin and Xuan-Phi Nguyen and Prathyusha Jwalapuram and Semih Yavuz and Caiming Xiong and Shafiq Joty},
        year         = {2026},
        eprint       = {2601.14652},
        archivePrefix= {arXiv},
        primaryClass = {cs.AI},
        note         = {ICML 2026},
      }
```
