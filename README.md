# SPC

**SPC: Evolving Self-Play Critic via Adversarial Games for LLM Reasoning**

The official implementation of SPC. [[arXiv]](https://arxiv.org/abs/2504.19162) [[Project]](https://chen-judge.github.io/SPC/) [[Hugging Face]](https://huggingface.co/papers/2504.19162)

**Jiaqi Chen**, Bang Zhang, Ruotian Ma, Peisong Wang, Xiaodan Liang, Zhaopeng Tu, Xiaolong Li, Kwan-Yee K. Wong.




<p align="center">
  <img src="figs/intro.png" alt="intro" width="80%">
</p>

If you have any questions, please contact me by email: [jqchen(at)cs.hku.hk](mailto:jqchen@cs.hku.hk)


## Environment 🔧

Please install these requirements:
```bash
pip install -r requirements.txt
```
For inference, please also install `vllm` (we use version 0.6.6).



## Data 📚
Please find our generated training data and evaluation datasets [here](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/jadge_connect_hku_hk/EkB9OYBHr_tGmGeJ5xxTncgBXFnln9nPP4jmCKNcQSSDIQ?e=oF7b6g).

`data_round0_sft_critic.json` contains SFT data and `data_round2_rl_critic.json` contains data generated in rounds 1 and 2 for RL training.

The three files in `data/eval` correspond to the datasets used for evaluating the critic.

## Checkpoints 🤗

We have uploaded the trained [SFT critic model](https://huggingface.co/judge/SPC-Critic-0/tree/main) (round 0) and [RL critic model](https://huggingface.co/judge/SPC-Critic-2/tree/main) (round 2) to Hugging Face!

## Reinforcement Finetuning 🔥
You can use the provided data to finetune the SFT critic model into the round 2 critic model. 

Please modify the data and model paths in the script as needed before running:
```bash
bash scripts/rl_critic.sh
```

## Evaluation 🚀
After training a critic model or directly using our provided checkpoints, please set the dataset and checkpoint paths in the following script to perform evaluation.

```bash
python3 eval/infer_batch.py
```

## Citation 🌟
<pre>
@article{chen2025spc,
  title={SPC: Evolving Self-Play Critic via Adversarial Games for LLM Reasoning},
  author={Chen, Jiaqi and Zhang, Bang and Ma, Ruotian and Wang, Peisong and Liang, Xiaodan and Tu, Zhaopeng and Li, Xiaolong and Wong, Kwan-Yee~K.},
  journal={arXiv preprint arXiv:2504.19162},
  year={2025}
}
</pre>