import json
import os
import wandb

from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig, Trainer, TrainingArguments, GPTNeoXForCausalLM
from training.utility import DataArguments, set_seed, AdditionalTrainingArguments
from training.utility import train_and_save_model
from training.callback import FirstTokenAccuracyCallback, PreTrainingShuffleBiographyCallBack
from training.callback import FirstTokenAccuracyCalculationStrategy
from data_module import AttentionMaskType
from data_module import construct_pre_training_first_token_accuracy_data_module, construct_pre_training_data_module

import torch
from typing import Optional, Tuple, Union, List, Dict
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa, \
    _prepare_4d_attention_mask_for_sdpa
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from transformers.tokenization_utils_base import BatchEncoding
from transformers.modeling_utils import logging
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
import argparse

logger = logging.get_logger(__name__)


# region code of GPT2 when AttentionMaskType.MASK_PREVIOUS_BIOGRAPHY is used
def gpt2_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, input_shape[-1])

    if past_key_values is None:
        past_length = 0
        past_key_values = tuple([None] * len(self.h))
    else:
        past_length = past_key_values[0][0].size(-2)
    if position_ids is None:
        position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.wte(input_ids)
    position_embeds = self.wpe(position_ids)
    hidden_states = inputs_embeds + position_embeds

    # Attention mask.
    _use_sdpa = self._attn_implementation == "sdpa" and output_attentions is False and head_mask is None
    assert _use_sdpa and attention_mask is not None, "Only support sdpa with attention_mask"
    if attention_mask is not None:
        if attention_mask.dim() == 3:  # Add to mask previous biography entry
            attention_mask = attention_mask[:, None, :, :]
            new_attention_mask = torch.zeros_like(attention_mask,
                                                  device=attention_mask.device,
                                                  dtype=inputs_embeds.dtype)
            new_attention_mask = new_attention_mask.masked_fill(attention_mask, 0)
            new_attention_mask = new_attention_mask.masked_fill(~attention_mask, torch.finfo(inputs_embeds.dtype).min)
            attention_mask = new_attention_mask
        else:
            attention_mask = attention_mask.view(batch_size, -1)
        if self._attn_implementation == "flash_attention_2":
            attention_mask = attention_mask if 0 in attention_mask else None
        elif _use_sdpa:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask=attention_mask,
                input_shape=(batch_size, input_shape[-1]),
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_length,
            )
        else:
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

    # If a 2D or 3D attention mask is provided for the cross-attention
    # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
    if self.config.add_cross_attention and encoder_hidden_states is not None:
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
        if _use_sdpa:
            encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                mask=encoder_attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1]
            )
        elif not self._attn_implementation == "flash_attention_2":
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
    else:
        encoder_attention_mask = None

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # head_mask has shape n_layer x batch x n_heads x N x N
    head_mask = self.get_head_mask(head_mask, self.config.n_layer)

    if token_type_ids is not None:
        token_type_embeds = self.wte(token_type_ids)
        hidden_states = hidden_states + token_type_embeds

    hidden_states = self.drop(hidden_states)

    output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    presents = () if use_cache else None
    all_self_attentions = () if output_attentions else None
    all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
    all_hidden_states = () if output_hidden_states else None
    for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(hidden_states.device)
            # Ensure layer_past is on same device as hidden_states (might not be correct)
            if layer_past is not None:
                layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
            # Ensure that attention_mask is always on the same device as hidden_states
            if attention_mask is not None:
                attention_mask = attention_mask.to(hidden_states.device)
            if isinstance(head_mask, torch.Tensor):
                head_mask = head_mask.to(hidden_states.device)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.gradient_checkpointing and self.training:
            outputs = self._gradient_checkpointing_func(
                block.__call__,
                hidden_states,
                None,
                attention_mask,
                head_mask[i],
                encoder_hidden_states,
                encoder_attention_mask,
                use_cache,
                output_attentions,
            )
        else:
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

        hidden_states = outputs[0]
        if use_cache is True:
            presents = presents + (outputs[1],)

        if output_attentions:
            all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
            if self.config.add_cross_attention:
                all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

        # Model Parallel: If it's the last layer for that device, put things on the next device
        if self.model_parallel:
            for k, v in self.device_map.items():
                if i == v[-1] and "cuda:" + str(k) != self.last_device:
                    hidden_states = hidden_states.to("cuda:" + str(k + 1))

    hidden_states = self.ln_f(hidden_states)

    hidden_states = hidden_states.view(output_shape)
    # Add last hidden state
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
            if v is not None
        )

    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
        cross_attentions=all_cross_attentions,
    )


def pseudo_pad(tokenizer: GPT2TokenizerFast, *pad_args, **pad_kwargs) -> BatchEncoding:
    assert isinstance(tokenizer, GPT2TokenizerFast)
    raise NotImplementedError('The function is used to replace pad_without_fast_tokenizer_warning, '
                              'which costs a huge amount of time.')


# endregion


def train(args):
    # For both BIO pretraining and mix training, we employed a conventional set of optimization parameters:
    # the AdamW optimizer with a weight decay of 0.1, ε = 10−6, an initial learning rate of 0.001,
    # a 1000-step linear warmup, and cosine learning rate decay (from 0.001 decreasing to 0.0001).
    # We used a batch size of 96.
    #
    # documents about setting the lr_scheduler
    # https://github.com/huggingface/transformers/pull/29341
    # https://github.com/huggingface/transformers/pull/29341/files#diff-ac9b69e204d41f7ddf23e8bd5b53a2a0ffa64c4784101abeef3b5d4c14342656
    #
    # deepspeed setting
    # https://github.com/microsoft/DeepSpeed/issues/3488
    #
    # huggingface tutorial
    # https://huggingface.co/learn/nlp-course/en/chapter7/6

    # region read config
    pre_training_config = json.load(open(args.config_path, 'r'))
    wandb_config: Dict = pre_training_config['wandb']
    wandb_config['run_name'] = args.task_name
    shared_config: Dict = pre_training_config['shared']  # config that is shared by all tasks
    task_config: Dict = pre_training_config['task'][args.task_name]  # config that is different for each task
    del pre_training_config  # avoid misuse
    # endregion

    # region set trainer arguments
    output_dir = task_config['output_dir']
    # max_steps = 80_000,  # paper 🔵
    # warmup_steps = 1_000,  # paper 🔵
    max_steps = task_config['max_steps']  # set by myself 🟡
    warmup_steps = task_config['warmup_steps']  # set by myself 🟡
    training_args = TrainingArguments(
        output_dir=output_dir,  # set by myself 🟡
        optim='adamw_torch',  # default, paper 🔵
        per_device_train_batch_size=96,  # paper 🔵
        # per_device_train_batch_size=8,  # debug 🔴
        eval_strategy='no',  # default (no eval dataset)
        gradient_accumulation_steps=1,  # default
        max_steps=max_steps,
        # max_steps=100,  # debug 🔴
        weight_decay=0.1,  # paper 🔵
        adam_epsilon=1e-6,  # paper 🔵
        warmup_steps=warmup_steps,
        # warmup_steps=50,  # debug 🔴
        lr_scheduler_type="cosine_with_min_lr",  # paper 🔵
        lr_scheduler_kwargs={"min_lr": 0.0001},  # paper 🔵
        learning_rate=0.001,  # paper 🔵
        save_strategy='steps',  # set by myself 🟡
        save_steps=max_steps//10,  # set by myself 🟡
        # save_steps=10,  # debug 🔴
        # save_total_limit=1,  # set by myself 🟡
        bf16=True,  # set by myself 🟡
        logging_steps=1,  # set by myself 🟡
        report_to=['wandb'],  # set by myself 🟡
        # deepspeed='./ds_configs/stage2.json',
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # endregion

    # region set wandb
    # https://github.com/wandb/wandb/issues/5244#issuecomment-1666349612
    wandb.login(key=os.environ.get('WANDB_API'), relogin=True)  # NOTE: load WANDB_API from environment
    wandb.init(
        project=wandb_config['project'],
        name=wandb_config['run_name'],
        config={**wandb_config, 'all_config': {'shared': shared_config, 'task': task_config, 'wandb': wandb_config}},
    )
    # endregion

    # region construct tokenizer and model
    # model_path = "./model/gpt2/origin"
    model_path = shared_config['model_path']
    additional_training_args = AdditionalTrainingArguments(
        attention_mask_type=AttentionMaskType.ALL_TRUE,
        first_token_accuracy_calculation_strategy=FirstTokenAccuracyCalculationStrategy.EPOCH,
        first_token_accuracy_calculation_interval=5,
        pre_training_person_index_info_list=task_config['person_index_info_list']
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=512,  # paper 🔵
        # model_max_length=32,  # debug 🔴
        padding_side="right",
        use_fast=True,
    )
    assert tokenizer.pad_token is None
    tokenizer.pad_token = tokenizer.unk_token
    if 'gpt2' in model_path:
        if task_config['previous_output_dir'] != '':
            raise NotImplementedError('Only support pre-training from scratch')
        if additional_training_args.attention_mask_type == AttentionMaskType.MASK_PREVIOUS_BIOGRAPHY:
            GPT2Model.forward = gpt2_forward
        # FIXME: change n_ctx to n_positions so that the length of position embedding is correct
        # https://github.com/huggingface/transformers/commit/5b45422b58da508dab8b18852609a279517b944d
        config = AutoConfig.from_pretrained(
            model_path,
            vocab_size=len(tokenizer),
            n_ctx=tokenizer.model_max_length,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        model = GPT2LMHeadModel(config)
    elif 'gpt-neox' in model_path:
        if task_config['previous_output_dir'] == '':
            config = AutoConfig.from_pretrained(
                model_path,
                vocab_size=len(tokenizer),
                max_position_embeddings=tokenizer.model_max_length,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            # check following to see whether the max length is correctly set
            # https://github.com/huggingface/transformers/blob/080e14b24c8923e4bc18fcd54010fc7396c67bc0/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L595
            model = GPTNeoXForCausalLM(config)
        else:
            model = GPTNeoXForCausalLM.from_pretrained(os.path.join(task_config['previous_output_dir'], 'final_model'))
    else:
        raise NotImplementedError(f"model_path: {model_path}")
    # endregion

    # region construct data module and trainer
    data_args = DataArguments(
        biography_data_path=shared_config['biography_data_path']
    )
    # no eval_dataset
    train_dataset = construct_pre_training_data_module(tokenizer, data_args, additional_training_args)
    first_token_accuracy_dataset = construct_pre_training_first_token_accuracy_data_module(
        tokenizer, data_args, additional_training_args)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[
            FirstTokenAccuracyCallback(first_token_accuracy_dataset,
                                       additional_training_args.first_token_accuracy_calculation_strategy,
                                       additional_training_args.first_token_accuracy_calculation_interval),
            PreTrainingShuffleBiographyCallBack
        ],
    )
    # endregion

    # region train and save model
    train_and_save_model(trainer, training_args, remove_all_checkpoint=False)
    # endregion


if __name__ == '__main__':
    my_env = os.environ.copy()
    my_env["PATH"] = '/dev_data/qsj/conda/envs/forgetting/bin/:' + my_env["PATH"]
    os.environ.update(my_env)
    set_seed()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    train(parser.parse_args())
