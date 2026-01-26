import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import math
from torch import nn
import torch.nn.functional as F
import pdb
from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import transformers

try:
    # For modern transformers
    from transformers.generation.utils import (
        GenerateDecoderOnlyOutput,
        GenerateEncoderDecoderOutput,
        SampleOutput, SampleEncoderDecoderOutput, SampleDecoderOnlyOutput,
        BeamSearchOutput, BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput
    )
except ImportError:
    # Fallback aliases for typing only
    GenerateDecoderOnlyOutput = Dict[str, Any]
    GenerateEncoderDecoderOutput = Dict[str, Any]
    SampleOutput = Dict[str, Any]
    SampleEncoderDecoderOutput = Dict[str, Any]
    SampleDecoderOnlyOutput = Dict[str, Any]
    BeamSearchOutput = Dict[str, Any]
    BeamSearchEncoderDecoderOutput = Dict[str, Any]

logits_processor = LogitsProcessorList()
logits_warper = LogitsProcessorList()
IMAGE_TOKEN_INDEX = -200
import matplotlib.pyplot as plt
from transformers import GenerationConfig
from transformers.generation.utils import GenerationMixin

class VisualZeroHook:
    def __init__(self, start_idx, end_idx, expand=True):
        self.start = start_idx
        self.end = end_idx
        self.expand = expand

    def __call__(self, module, args):
        hidden_states = args[0]
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        if self.expand:
            hidden_states_no_v = hidden_states.clone()
            hidden_states_no_v[:, self.start:self.end, :] = 0
            new_states = torch.cat([hidden_states, hidden_states_no_v], dim=0)
        else:
            B = hidden_states.shape[0]
            half = B // 2
            hidden_states[half:, self.start:self.end, :] = 0
        return (hidden_states,) + args[1:]

def _sample_vgd(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[SampleOutput, torch.LongTensor]:

        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
        if compile_forward:
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            if self.config._attn_implementation == "flash_attention_2":
                if generation_config.compile_config is not None and generation_config.compile_config.fullgraph:
                    logger.warning_once(
                        "When using Flash Attention 2 and a static cache, you cannot use the option `CompileConfig(fullgraph=True)` as "
                        "FA2 introduces graph breaks. We overrode the option with `fullgraph=False`."
                    )
                    generation_config.compile_config.fullgraph = False
            model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True

        if "Gemma3" in type(self.model.config).__name__:
            vs_id  = 255999
            ve_id  = 256000
            vstart = input_ids[0].tolist().index(vs_id)
            vend = torch.where(input_ids[0] == ve_id)[0].max().item()

        elif "Qwen3" in type(self.model.config).__name__ and 'vgd_input_ids' in model_kwargs:
            current_input_ids = model_kwargs['vgd_input_ids']
            img_context = getattr(self, 'img_context_token_id', 151671)
            matches = (current_input_ids == img_context).nonzero(as_tuple=True)
            if len(matches) > 1:
                seq_idx = matches[1]
            else:
                seq_idx = matches[0]
            vstart = seq_idx.min().item()
            vend = seq_idx.max().item()

        else:
            vs_id  = self.model.config.vision_start_token_id       # e.g., 151652
            ve_id  = self.model.config.vision_end_token_id         # e.g., 151653
            vstart = input_ids[0].tolist().index(vs_id)
            vend = torch.where(input_ids[0] == ve_id)[0].max().item()

        if 'visual_alpha' in model_kwargs:
            visual_alpha = model_kwargs['visual_alpha']
        else:
            visual_alpha = getattr(generation_config, 'visual_alpha', 0.0)
        
        # Expand inputs for VGD (Batch size 1 -> 2) to avoid shape mismatch in Qwen3-VL internals
        if input_ids.shape[0] == 1:
            input_ids = input_ids.repeat(2, 1)
            new_kwargs = {}
            for k, v in model_kwargs.items():
                if isinstance(v, torch.Tensor) and (v.shape[0] == 1 or k in ["pixel_values", "inputs_embeds", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]):
                    new_kwargs[k] = v.repeat(2, *([1] * (v.ndim - 1)))
                else:
                    new_kwargs[k] = v
            model_kwargs = new_kwargs
        
        # Register hook on the first layer of the model
        hooks = []
        layers = None
        if hasattr(self, 'language_model'):
             if hasattr(self.language_model, 'model'):
                 layers = self.language_model.model.layers
             elif hasattr(self.language_model, 'layers'):
                 layers = self.language_model.layers
        elif hasattr(self, 'model'):
             if hasattr(self.model, 'layers'):
                 layers = self.model.layers
             elif hasattr(self.model, 'language_model'):
                 layers = self.model.language_model.layers
        
        if layers is not None:
            hook = VisualZeroHook(vstart, vend, expand=False)
            hooks.append(layers[0].register_forward_pre_hook(hook))
            #for i, layer in enumerate(layers):
            #    hook = VisualZeroHook(vstart, vend, expand=False)
            #    hooks.append(layer.register_forward_pre_hook(hook))
            

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            forward_call = self if is_prefill else model_forward
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = forward_call(**model_inputs, return_dict=True)
            
            next_token_logits = outputs.logits[:, -1, :][0].unsqueeze(0).to(copy=True, dtype=torch.float32, device=input_ids.device)
            next_token_logits_no_v = outputs.logits[:, -1, :][1].unsqueeze(0).to(copy=True, dtype=torch.float32, device=input_ids.device)
            #pdb.set_trace()

            # VGD Logic: Uncond + alpha * (Cond - Uncond)
            vgd_logits = next_token_logits_no_v + visual_alpha * (next_token_logits - next_token_logits_no_v)
            next_token_scores = logits_processor(input_ids, vgd_logits)

            if is_prefill:
                is_prefill = False
            # ------------------------------------------------
            if synced_gpus and this_peer_finished:
                continue

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            if input_ids.shape[0] != next_tokens.shape[0] and next_tokens.shape[0] == 1:
                next_tokens = next_tokens.repeat(input_ids.shape[0])

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs
        
        for hook in hooks: hook.remove()

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids

def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[SampleOutput, torch.LongTensor]:
        
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
        if compile_forward:
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            # If we use FA2 and a static cache, we cannot compile with fullgraph
            if self.config._attn_implementation == "flash_attention_2":
                # only raise warning if the user passed an explicit compile-config
                if generation_config.compile_config is not None and generation_config.compile_config.fullgraph:
                    logger.warning_once(
                        "When using Flash Attention 2 and a static cache, you cannot use the option `CompileConfig(fullgraph=True)` as "
                        "FA2 introduces graph breaks. We overrode the option with `fullgraph=False`."
                    )
                    generation_config.compile_config.fullgraph = False
            model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids

_original_validate_model_kwargs = GenerationMixin._validate_model_kwargs

def _validate_model_kwargs_vgd(self, model_kwargs: Dict[str, Any]):
    # Keys to ignore during validation to prevent ValueError in transformers
    ignore_keys = {
        'visual_alpha', 'verbose', 'reuse', 'use_custom_prompt', 
        'use_vllm', 'presence_penalty', 'repetition_penalty', 'top_p', 'top_k',
        'vgd_input_ids'
    }
    
    # Temporarily remove custom keys so validation passes
    removed = {}
    for k in list(model_kwargs.keys()):
        if k in ignore_keys:
            removed[k] = model_kwargs.pop(k)
            
    try:
        # The original function modifies model_kwargs in-place and returns None
        output = _original_validate_model_kwargs(self, model_kwargs)
    finally:
        # Restore custom keys so they are available for VGD hook/sampling
        model_kwargs.update(removed)
        
    # FIX: Only update output if it is actually a dictionary (some older/custom versions might return a new dict)
    if output is not None and isinstance(output, dict) and output is not model_kwargs:
        output.update(removed)
        return output
        
    return model_kwargs

def evolve_guidance_sampling(temperature=1.0, do_sample=True, visual_alpha=1.0):
    transformers.generation.utils.GenerationMixin._validate_model_kwargs = _validate_model_kwargs_vgd
    if temperature > 0 and do_sample and visual_alpha > 0:
        print("Using Visual Guidance Sampling")
        transformers.generation.utils.GenerationMixin._sample = _sample_vgd
    else:
        print("Using Regular Sampling")
        transformers.generation.utils.GenerationMixin._sample = _sample