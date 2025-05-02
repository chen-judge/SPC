from arguments import OfflineWeightedPolicyTrainingArguments, CustomTrainingArguments
from transformers import (HfArgumentParser, PreTrainedModel, PreTrainedTokenizer, PretrainedConfig,
                        Qwen2Config, Qwen2Tokenizer, Qwen2ForCausalLM, AutoConfig, AutoTokenizer, AutoModelForCausalLM)
from utils import print_object_on_main_process, print_rank_0, getDataset, set_special_tokens
from typing import Dict, List, Any, Tuple, Union
from collators import offline_weighted_policy_data_collator
from trainers import OfflineWeightedPolicyTrainer
import wandb
import os


def data_transform(data_list: List[Dict[str, List]], args: OfflineWeightedPolicyTrainingArguments) -> List[Dict[str, Any]]:
    new_data_list = []
    for data in data_list:
        new_data = {
            "prompt": data[args.data_prompt_name],
            "answer": data[args.data_answer_name],
            "weight": data.get(args.data_weight_name, 1.0),
            "reward": data.get(args.data_reward_name, 1.0),
            "value":data.get(args.data_value_name, 0.),
            "type": 'sample' if 'reward' in data else 'sft'
        }
        new_data_list.append(new_data)

    if args.debug_mode:
        new_data_list = new_data_list[:100]

    return new_data_list


def loadTokenizerAndModel(args: CustomTrainingArguments) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    CLASS_MAP: Dict[str, Dict[str, Union[PreTrainedModel, PreTrainedTokenizer, PretrainedConfig]]] = {
        "qwen": {"config": AutoConfig, "tokenizer": AutoTokenizer, "model": AutoModelForCausalLM}
    }
    try:
        CONFIG_CLASS, TOKENIZER_CLASS, MODEL_CLASS = CLASS_MAP[args.model_type].values()
    except:
        raise ValueError(f"Do not support model type '{args.model_type}'")
    
    def _load_tokenizer(TOKENIZER_CLASS: PreTrainedTokenizer) -> PreTrainedTokenizer:
        tokenizer = TOKENIZER_CLASS.from_pretrained(
                args.model_name_or_path,
                truncation_side=args.truncation_side,
                padding_side=args.padding_side,
                trust_remote_code=True
            )
        tokenizer.model_max_length = args.model_max_length
        return tokenizer

    config = CONFIG_CLASS.from_pretrained(args.model_name_or_path)
    config.use_cache = False
    tokenizer = _load_tokenizer(TOKENIZER_CLASS)
    model = MODEL_CLASS.from_pretrained(args.model_name_or_path, config=config)
    set_special_tokens(tokenizer, model)

    return tokenizer, model


def main():
    parser = HfArgumentParser((OfflineWeightedPolicyTrainingArguments,))
    args: OfflineWeightedPolicyTrainingArguments = parser.parse_args_into_dataclasses()[0]
    
    # args.report_to = ["wandb"]
    # os.environ["WANDB_PROJECT"] = "spc" 
    # os.environ["WANDB_NAME"] = args.output_dir.split("/")[-1]
    # os.environ["WANDB_API_KEY"] = ""
    
    print_object_on_main_process("Arguments", args)

    print_rank_0("Loading data>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    if args.do_train:
        train_dataset = getDataset(args, data_transform=data_transform, type='train')
    eval_dataset = getDataset(args, data_transform=data_transform, type='eval')
    print_object_on_main_process("training set", train_dataset, split_line_color="green", object_color="cyan")
    print_object_on_main_process("evaluation set", eval_dataset, split_line_color="green", object_color="cyan")

    tokenizer, model = loadTokenizerAndModel(args)
    print_object_on_main_process("tokenizer", tokenizer, split_line_color="green", object_color="cyan")
    print_object_on_main_process("model", model, split_line_color="green", object_color="cyan")    

    _, ref_model = loadTokenizerAndModel(args)


    trainer = OfflineWeightedPolicyTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=offline_weighted_policy_data_collator(tokenizer, args)
        )

    if args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        if args.save_training_states:
            trainer.save_state()
        trainer.save_model(output_dir=args.output_dir)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_model(output_dir=args.output_dir)
        

if __name__ == '__main__':
    main()
    