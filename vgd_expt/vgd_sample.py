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
from transformers.generation.beam_search import BeamScorer
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
    def __init__(self, start_idx, end_idx):
        self.start = start_idx
        self.end = end_idx

    def __call__(self, module, args):
        hidden_states = args[0]
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        B = hidden_states.shape[0]
        half = B // 2
        hidden_states[half:, self.start:self.end, :].zero_()
        return (hidden_states,) + args[1:]

class VisualTextDistortionHook:
    """
    Hook to add Gaussian noise to embeddings for variants of the following algos.
    "vis" for VCD (Visual Contrastive Decoding)
    "text for ICD (Instruction Contrastive Decoding)
    """
    def __init__(self, start_idx, end_idx, noise_type="vis", noise_scale=0.01):
        self.start = start_idx
        self.end = end_idx
        self.noise_type = noise_type
        self.noise_scale = noise_scale

    def __call__(self, module, args):
        hidden_states = args[0]
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        B = hidden_states.shape[0]
        half = B // 2
        if self.noise_type == "vis":
            noise = torch.randn_like(hidden_states[half:, self.start:self.end, :]) * self.noise_scale
            hidden_states[half:, self.start:self.end, :] += noise

        elif self.noise_type == "text":
            # 1. Text before image (Prefix/System prompts)
            noise_pre = torch.randn_like(hidden_states[half:, :self.start, :]) * self.noise_scale
            hidden_states[half:, :self.start, :] += noise_pre
                
            # 2. Text after image (User instructions/Suffix)
            noise_post = torch.randn_like(hidden_states[half:, self.end:, :]) * self.noise_scale
            hidden_states[half:, self.end:, :] += noise_post
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

        elif "Qwen3" in type(self.model.config).__name__ and 'vgd_input_ids' in model_kwargs: #INTERN-VL BRANCH
            current_input_ids = model_kwargs['vgd_input_ids']
            img_context = getattr(self, 'img_context_token_id', 151671)
            matches = (current_input_ids == img_context).nonzero(as_tuple=True)
            if len(matches) > 1:
                seq_idx = matches[1]
            else:
                seq_idx = matches[0]
            vstart = seq_idx.min().item()
            vend = seq_idx.max().item()
            
        elif "LlavaNextConfig" in type(self.model.config).__name__:
            image_token_id = getattr(self.model.config, 'image_token_index', 32000)
            vstart = torch.where(input_ids[0] == image_token_id)[0].min().item()
            vend = torch.where(input_ids[0] == image_token_id)[0].max().item() + 1

        else:
            vs_id  = self.model.config.vision_start_token_id       # e.g., 151652
            ve_id  = self.model.config.vision_end_token_id         # e.g., 151653
            vstart = input_ids[0].tolist().index(vs_id) + 1
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
            if "Qwen3VL" in type(self.model.config).__name__:
                for i, layer in enumerate(layers):
                    hook = VisualZeroHook(vstart, vend)
                    hooks.append(layer.register_forward_pre_hook(hook))
            else:
                hook = VisualZeroHook(vstart, vend)
                hooks.append(layers[0].register_forward_pre_hook(hook))
            

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            forward_call = self if is_prefill else model_forward
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = forward_call(**model_inputs, return_dict=True)
            
            next_token_logits = outputs.logits[:, -1, :][0].unsqueeze(0).to(copy=True, dtype=torch.float32, device=input_ids.device)
            next_token_logits_no_v = outputs.logits[:, -1, :][1].unsqueeze(0).to(copy=True, dtype=torch.float32, device=input_ids.device)
            #pdb.set_trace()

            # --- VGD Logic ---
            # VGD Logic: Uncond + alpha * (Cond - Uncond)
            vgd_logits = next_token_logits_no_v + visual_alpha * (next_token_logits - next_token_logits_no_v)
            next_token_scores = logits_processor(input_ids[:1], vgd_logits)

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

def _sample_contrastive(
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

        elif "Qwen3" in type(self.model.config).__name__ and 'vgd_input_ids' in model_kwargs: #INTERN-VL BRANCH
            current_input_ids = model_kwargs['vgd_input_ids']
            img_context = getattr(self, 'img_context_token_id', 151671)
            matches = (current_input_ids == img_context).nonzero(as_tuple=True)
            if len(matches) > 1:
                seq_idx = matches[1]
            else:
                seq_idx = matches[0]
            vstart = seq_idx.min().item()
            vend = seq_idx.max().item()

        elif "LlavaNextConfig" in type(self.model.config).__name__:
            image_token_id = getattr(self.model.config, 'image_token_index', 32000)
            vstart = torch.where(input_ids[0] == image_token_id)[0].min().item()
            vend = torch.where(input_ids[0] == image_token_id)[0].max().item() + 1

        else:
            vs_id  = self.model.config.vision_start_token_id       # e.g., 151652
            ve_id  = self.model.config.vision_end_token_id         # e.g., 151653
            vstart = input_ids[0].tolist().index(vs_id) + 1
            vend = torch.where(input_ids[0] == ve_id)[0].max().item()

        if 'vcd_alpha' in model_kwargs:
            vcd_alpha = model_kwargs['vcd_alpha']
        else:
            vcd_alpha = getattr(generation_config, 'vcd_alpha', 0.0)

        if 'icd_alpha' in model_kwargs:
            icd_alpha = model_kwargs['icd_alpha']
        else:
            icd_alpha = getattr(generation_config, 'icd_alpha', 0.0)

        if 'vord_margin' in model_kwargs:
            vord_margin = model_kwargs['vord_margin']
        else:
            vord_margin = getattr(generation_config, 'vord_margin', 0.0)

        if vcd_alpha > 0:
            noise_type = "vis"
            alpha = vcd_alpha

        elif icd_alpha > 0:
            noise_type = "text"
            alpha = icd_alpha

        elif vord_margin > 0:
            noise_type = "vis" # Distorts the image just like VCD
            alpha = vord_margin

        # Expand inputs (Batch size 1 -> 2) to avoid shape mismatch in Qwen3-VL internals
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
            if "Qwen3VL" in type(self.model.config).__name__:
                for i, layer in enumerate(layers):
                    hook = VisualTextDistortionHook(vstart, vend, noise_type=noise_type, noise_scale=0.01)
                    hooks.append(layer.register_forward_pre_hook(hook))
            else:
                hook = VisualTextDistortionHook(vstart, vend, noise_type=noise_type, noise_scale=0.01)
                hooks.append(layers[0].register_forward_pre_hook(hook))
            

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            forward_call = self if is_prefill else model_forward
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = forward_call(**model_inputs, return_dict=True)
            
            # Extract Logits
            next_token_logits_clean = outputs.logits[:, -1, :][0].unsqueeze(0).to(copy=True, dtype=torch.float32, device=input_ids.device)
            next_token_logits_distorted = outputs.logits[:, -1, :][1].unsqueeze(0).to(copy=True, dtype=torch.float32, device=input_ids.device)

            if vord_margin > 0:
                cd_logits = next_token_logits_clean.clone()
                mask = next_token_logits_distorted > (next_token_logits_clean + vord_margin)
                cd_logits[mask] = -float('inf')

            else:
                # --- CD Logic ---
                # CD Formula: Logits = (1 + alpha) * Logits_Clean - alpha * Logits_Distorted
                cd_logits = (1 + alpha) * next_token_logits_clean - alpha * next_token_logits_distorted

            next_token_scores = logits_processor(input_ids[:1], cd_logits)

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

def opera_beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        key_position: Optional[dict] = None,
        scale_factor: Optional[float] = 50.0,
        threshold: Optional[int] = 15,
        num_attn_candidates: Optional[int] = 5, 
        window_size: Optional[int] = 512, 
        penalty_weights: Optional[float] = 1.0,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.beam_search`] directly. Use generate()
        instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`generation.BeamSearchDecoderOnlyOutput`], [`~generation.BeamSearchEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.BeamSearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.BeamSearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.


        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForSeq2SeqLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     BeamSearchScorer,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

        >>> encoder_input_str = "translate English to German: How old are you?"
        >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


        >>> # lets run beam search using 3 beams
        >>> num_beams = 3
        >>> # define decoder start token ids
        >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
        >>> input_ids = input_ids * model.config.decoder_start_token_id

        >>> # add encoder_outputs to model keyword arguments
        >>> model_kwargs = {
        ...     "encoder_outputs": model.get_encoder()(
        ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
        ...     )
        ... }

        >>> # instantiate beam scorer
        >>> beam_scorer = BeamSearchScorer(
        ...     batch_size=1,
        ...     num_beams=num_beams,
        ...     device=model.device,
        ... )

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
        ...     ]
        ... )

        >>> outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Wie alt bist du?']
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only

        # initialise the history variables
        history_states = []
        history_rollback_locs = None
        beam_next_tokens = None
        beam_idx = None
        rollback_pos = 0
        max_rollback_time = torch.zeros(window_size)
        history_length = window_size
        reject_token_pos_gather = [[] for _ in range(window_size)]
        model_kwargs_ori = model_kwargs.copy()

        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # Define current states
            current_state = {}
            current_state["input_ids"] = input_ids.clone()
            current_state["beam_scorer"] = copy.deepcopy(beam_scorer)
            current_state["beam_indices"] = beam_indices.copy() if beam_indices is not None else None
            current_state["cur_len"] = cur_len

            # prepare model inputs 
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            # Load the previous self-attention weights
            if not "past_key_values" in model_kwargs.keys():
                attn_previous = outputs.attentions[-1].clone() # [batch_size * num_beams, num_head, q, kv]
            else:
                assert beam_idx is not None and attn_previous is not None
                attn_previous = torch.cat([attn_previous, torch.zeros_like(attn_previous).sum(-1, keepdim=True)], -1)
                attn_previous = torch.cat(
                    [attn_previous[beam_idx], outputs.attentions[-1].clone().max(1, keepdim=True).values.data], -2) # [batch_size * num_beams, num_head, q, kv]
            
            attn_previous = attn_previous.max(1, keepdim=True).values.data # [batch_size * num_beams, 1, q, kv]
            current_state["attn_previous"] = attn_previous.data.cpu()


            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # Select candidates
            if num_attn_candidates < 1:
                raise ValueError(
                    f"Num of candidates must be larger than 0, but it is currently {num_attn_candidates}."
                )
            candidate_token_scores, candidate_tokens = torch.topk(
                next_token_logits, num_attn_candidates, dim=-1, largest=True, sorted=True
            ) # [batch_size * num_beams, num_attn_candidates]
            current_state["candidate_tokens"] = candidate_tokens.clone()

            current_state["beam_scores"] = beam_scores.clone()
            current_state["beam_next_tokens"] = beam_next_tokens.clone() if beam_next_tokens is not None else None
            current_state["beam_idx"] = beam_idx.clone() if beam_idx is not None else None

            # Walk through all candidates to get their self-attention weights
            attn_last = []
            for candidate_id in range(num_attn_candidates):
                # update temporary generated ids, model inputs, and length for next step
                input_ids_tmp = torch.cat([input_ids, candidate_tokens[:, candidate_id].unsqueeze(-1)], dim=-1)

                model_kwargs_tmp = model_kwargs.copy()
                model_kwargs_tmp = self._update_model_kwargs_for_generation(
                    outputs, model_kwargs_tmp, is_encoder_decoder=self.config.is_encoder_decoder
                )

                # prepare model inputs
                model_inputs_tmp = self.prepare_inputs_for_generation(input_ids_tmp, **model_kwargs_tmp)

                # forward pass to get the self-attention maps of next token prediction
                outputs_tmp = self(
                    **model_inputs_tmp,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )

                attn_output = outputs_tmp.attentions[-1].clone()
                attn_output = attn_output.max(1, keepdim=True).values.data # [batch_size * num_beams, 1, 1, kv+1]
                attn_square = torch.cat([attn_previous, torch.zeros_like(attn_previous).sum(-1, keepdim=True)], -1) # [batch_size * num_beams, 1, q, kv+1]
                attn_square = torch.cat([attn_square, attn_output], -2) # [batch_size * num_beams, 1, q+1, kv+1]
                attn_last.append(attn_square) # [batch_size * num_beams, 1, q+1, kv+1]

            del input_ids_tmp, model_kwargs_tmp, model_inputs_tmp, outputs_tmp

            # Gather the attentions of all candidates
            attn_last = torch.cat(attn_last, 1) # [batch_size * num_beams, num_attn_candidates, q+1, kv+1]
            attn_last = attn_last / attn_last.sum(-1, keepdim=True)

            # Catch the self-attention weights with the size of local window
            # [batch_size * num_beams, num_attn_candidates, window_size, window_size]
            attn_pos = key_position
            attn_local = attn_last[:, :, attn_pos["response_start"]:, attn_pos["response_start"]:]

            # Scale up the self-attention weights and calculate the scores
            attn_local = scale_factor * attn_local
            attn_local_scores = torch.zeros((
                attn_local.shape[0], attn_local.shape[1], attn_local.shape[-1]), dtype=torch.float16).to(candidate_token_scores.device)
            for j in range(attn_local.shape[-1]):
                local_score = 1e-7 * attn_local[..., j:, j].prod(-1).data
                attn_local_scores[..., j] = local_score.to(torch.float32) # [batch_size * num_beams, num_attn_candidates, window_size]

            # We use the attention scores to penalize the first 10 tokens
            cur_response_lens = attn_local.shape[-1]
            attn_i = attn_last[:, :, -1, attn_pos["image_start"]:attn_pos["image_end"]+1].sum(-1)
            attn_scores = attn_i # [batch_size * num_beams, num_attn_candidates]

            # We use the rollback scores to penalize the subsequent tokens
            rollback_scores, rollback_locs = attn_local_scores.max(-1) # [batch_size * num_beams, num_attn_candidates]
            rollback_loc = rollback_locs.mode().values.data # [batch_size * num_beams]
            rollback_loc = rollback_loc.mode().values.data # [1]

            penalty_scores = - attn_scores if cur_response_lens <= 10 else rollback_scores # [batch_size * num_beams, num_attn_candidates]

            # incorporate with the history locations of the maximum of penalty scores
            if history_rollback_locs is None:
                history_rollback_locs = [rollback_locs.mode().values.data[:, None]]
            else:
                history_rollback_locs.append(rollback_locs.mode().values.data[:, None])
            rollback_loc_gathers = torch.cat(history_rollback_locs, -1)# [batch_size * num_beams, window_size]

            candidate_token_scores -= penalty_weights * penalty_scores
            current_state["candidate_token_scores"] = candidate_token_scores.clone()

            # history check
            if len(history_states) >= history_length:
                history_states.pop(0)
            history_states.append(current_state)

            # check if we need rollback
            try:
                if all((rollback_loc_gather == rollback_loc).long().sum() > int(threshold) for _, rollback_loc_gather in enumerate(rollback_loc_gathers)):
                    if rollback_loc < 10: # or rollback_loc + 1 < rollback_pos:
                        assert False
                    # locate the rollback position
                    rollback_pos = rollback_loc + 1
                    if max_rollback_time[rollback_pos] >= num_attn_candidates:
                        # print(f"Already reach the maximum rollback times at position {rollback_pos}, so shift the rollback position to {rollback_pos-1}")
                        rollback_pos = rollback_pos - 1
                        if max_rollback_time[rollback_pos] >= num_attn_candidates:
                            assert False
                        else:
                            max_rollback_time[rollback_pos] += 1
                    else:
                        max_rollback_time[rollback_pos] += 1
                    if cur_response_lens - rollback_pos > history_length + 1:
                        rollback_pos = max(1, cur_response_lens - history_length - 1)
                    # print(f"rollback from pos {cur_response_lens-1} to pos {rollback_pos} for the time {int(max_rollback_time[rollback_pos])}")

                    # discard the rollbacked states in history
                    for j in range(cur_response_lens-rollback_pos-2):
                        history_states.pop(-1)
                        history_rollback_locs.pop(-1)
                        reject_token_pos_gather[-(j+1)] = []

                    # Revive all of variables in the state of the rollback position
                    input_ids = history_states[-2]["input_ids"]
                    beam_scorer = history_states[-2]["beam_scorer"]
                    beam_indices = history_states[-2]["beam_indices"]
                    cur_len = history_states[-2]["cur_len"]

                    attn_previous = history_states[-2]["attn_previous"].to(input_ids.device)
                    candidate_token_scores = history_states[-2]["candidate_token_scores"]
                    candidate_tokens = history_states[-2]["candidate_tokens"]

                    beam_scores = history_states[-2]["beam_scores"]
                    beam_next_tokens = history_states[-1]["beam_next_tokens"]
                    beam_idx = history_states[-1]["beam_idx"]

                    # first inference to get model kwargs
                    if "images" in model_kwargs_ori.keys():
                        model_kwargs = model_kwargs_ori.copy()
                        model_kwargs["attention_mask"] = torch.cat([
                            model_kwargs["attention_mask"], torch.ones((
                                input_ids.shape[0], input_ids[:,:-1].shape[1] - model_kwargs["attention_mask"].shape[1]
                            )).to(input_ids.device)], 1)

                        model_inputs_tmp = self.prepare_inputs_for_generation(input_ids[:,:-1], **model_kwargs)
                    else:
                        answer_embeds = self.model.embed_tokens(input_ids[:,1:-1])
                        model_kwargs = model_kwargs_ori.copy()
                        model_kwargs["inputs_embeds"] = torch.cat([model_kwargs["inputs_embeds"], answer_embeds], 1)
                        model_kwargs["attention_mask"] = torch.cat(
                            [model_kwargs["attention_mask"], torch.ones_like(input_ids[:,1:-1]).to(input_ids.device)], 1)

                        model_inputs_tmp = self.prepare_inputs_for_generation(input_ids[:,1:-1], **model_kwargs)

                    outputs_tmp = self(
                        **model_inputs_tmp,
                        return_dict=True,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                    )
                    model_kwargs = self._update_model_kwargs_for_generation(
                        outputs_tmp, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
                    )

                    # another inference to get outputs and logits
                    model_inputs_tmp = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

                    outputs = self(
                        **model_inputs_tmp,
                        return_dict=True,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                    )
                    next_token_logits = outputs.logits[:, -1, :]
                    del outputs_tmp, model_inputs_tmp

                    # discard the last rollbacked state in history
                    history_states.pop(-1)
                    history_rollback_locs.pop(-1)
                    reject_token_pos_gather[rollback_pos+1] = []

                    # set penalty on the corresponding candidates
                    next_token_logits -= 999. + next_token_logits.min(-1, keepdim=True).values.data
                    next_token_logits = next_token_logits.view(batch_size, num_beams * vocab_size)
                    beam_idx = beam_idx.view(batch_size, num_beams)
                    beam_next_tokens = beam_next_tokens.view(batch_size, num_beams)
                    reject_token_pos = beam_idx * vocab_size + beam_next_tokens
                    if len(reject_token_pos_gather[rollback_pos]) > 0:
                        reject_token_pos = torch.cat([reject_token_pos_gather[rollback_pos], reject_token_pos], -1)
                    reject_token_pos_gather[rollback_pos] = reject_token_pos
                    next_token_logits = next_token_logits.scatter_(-1, reject_token_pos, -999.)
                    next_token_logits = next_token_logits.view(batch_size * num_beams, vocab_size)
                else:
                    assert False
            except:
                next_token_logits.fill_(-999.)
                next_token_logits = next_token_logits.scatter_(-1, candidate_tokens, candidate_token_scores)

            del attn_last, attn_local, attn_local_scores
            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
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

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            cur_response_lens = input_ids.shape[-1]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx)

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]

_original_validate_model_kwargs = GenerationMixin._validate_model_kwargs

def _validate_model_kwargs_vgd(self, model_kwargs: Dict[str, Any]):
    # Keys to ignore during validation to prevent ValueError in transformers
    ignore_keys = {
        'visual_alpha','vcd_alpha', 'icd_alpha', 'vord_margin', 'verbose', 'reuse', 'use_custom_prompt',
        'opera_alpha', 'opera_scale', # Added OPERA keys
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

def evolve_guidance_sampling(temperature=1.0, do_sample=True, 
                             visual_alpha=0.0, vcd_alpha=0.0, icd_alpha=0.0, vord_margin=0.0, 
                             opera_alpha=0.0, opera_scale=50.0):
    
    transformers.generation.utils.GenerationMixin._validate_model_kwargs = _validate_model_kwargs_vgd
    
    if opera_alpha > 0:
        # OPERA is a Beam Search strategy
        print(f"Using OPERA Beam Search | Alpha: {opera_alpha} | Scale: {opera_scale}")
        transformers.generation.utils.GenerationMixin.beam_search = opera_beam_search
        
    elif temperature > 0 and do_sample:
        if visual_alpha > 0:
            print(f"Using Visual Guidance Sampling | Alpha: {visual_alpha}")
            transformers.generation.utils.GenerationMixin._sample = _sample_vgd
        elif vcd_alpha > 0:
            print(f"Using VCD Sampling | Alpha: {vcd_alpha}")
            transformers.generation.utils.GenerationMixin._sample = _sample_contrastive
        elif vord_margin > 0:
            print(f"Using VORD Sampling | Margin: {vord_margin}")
            transformers.generation.utils.GenerationMixin._sample = _sample_contrastive
        elif icd_alpha > 0:
            print(f"Using ICD Sampling | Alpha: {icd_alpha}")
            transformers.generation.utils.GenerationMixin._sample = _sample_contrastive
        else:
            print("Using Regular Sampling")
            transformers.generation.utils.GenerationMixin._sample = _sample