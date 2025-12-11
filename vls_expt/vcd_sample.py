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
import copy

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

class VisualZeroHook:
    def __init__(self, start_idx, end_idx):
        self.start = start_idx
        self.end = end_idx

    def __call__(self, module, args):
        hidden_states = args[0]
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        
        # Zero using stored attributes
        hidden_states[:, self.start:self.end, :] = 0
        return (hidden_states,) + args[1:]

# Can check for individual attention on man, flower, cat
def save_attention_heatmap(outputs, out_path="./scores/attention.png", gamma_factor=1.0):
    """
    Compute averaged attention from model outputs and save as a heatmap.
    
    Args:
        outputs: Model output with `.attentions` (list/tuple of tensors per layer).
        out_path: Path to save the heatmap image.
        gamma_factor: Gamma correction factor (1.0 = no change).
    """
    # Stack attentions: shape (layers, batch, heads, seq_len, seq_len)
    # attn_stack = torch.stack(outputs.attentions, dim=0)
    
    # Average over layers, heads, and batch -> (seq_len, seq_len)
    #avg_attn = attn_stack.mean(dim=(0, 1, 2))
    attn_stack = outputs.attentions[-1]
    avg_attn = attn_stack.mean(dim=(0, 1))
    
    # Apply gamma correction
    enhanced_attn = torch.pow(avg_attn, 100.0 / gamma_factor)
    # use log to scale
    
    # Convert to NumPy
    heatmap = enhanced_attn.cpu().numpy()

    # Plot and save
    plt.figure(figsize=(8, 8), dpi=150)
    plt.imshow(heatmap, cmap="viridis", interpolation="nearest")
    plt.colorbar()
    plt.title("Averaged Attention")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    
    print(f"Attention heatmap saved to {out_path}")


def plot_dynamics(video_attn_history, prompt_attn_history, generated_attn_history, order, filename="scores/attention_trace.png"):
    # Create directory if it doesn't exist

    steps = range(len(video_attn_history))    
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Entropies on Left Axis
    ax.set_xlabel('Generated Token Index')
    ax.set_ylabel('Attention (Avg.)', color='black')
    l1, = ax.plot(steps, video_attn_history, 'r-', label=order[0], alpha=0.6)
    l2, = ax.plot(steps, prompt_attn_history, 'g-', label=order[1], alpha=0.6)
    l3, = ax.plot(steps, generated_attn_history, 'b-', label=order[2], alpha=0.6)
    ax.tick_params(axis='y', labelcolor='black')
    ax.set_ylim(0, 1.0)

    # Combined Legend
    lines = [l1, l2, l3]
    ax.legend(lines, [l.get_label() for l in lines], loc='upper right')

    plt.title("Attention Dynamics")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    # --- SAVE THE PLOT ---    
    # bbox_inches='tight' prevents axis labels from getting cut off
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {filename}")
    plt.close() # Good practice to close memory references

def process_logits(input_ids, logits, T=1.0):
    #A helper function for temperature warping and top-k sampling
    logits = logits_processor(input_ids, logits)
    logits = logits_warper(input_ids, logits)
    return logits

def compute_individual_video_mass(last_layer_attn, vision_token_start_idx, vision_token_end_idx, grid=144):
    # Sum over Q, then average over K
    # need to visualize the attn map
    #avg_attn = last_layer_attn[0].mean(dim=0).sum(0).detach().cpu()
    #man_mass = avg_attn[vision_token_start_idx:vision_token_start_idx + grid].mean().item() # 49 : 1057
    #flower_mass = avg_attn[vision_token_start_idx + grid:vision_token_start_idx + 2 * grid].mean().item() # 1058 : 2065
    #cat_mass = avg_attn[vision_token_start_idx + 2 * grid:vision_token_end_idx].mean().item() # 2065 : 3073
    avg_attn = last_layer_attn[0].mean(dim=0).squeeze(0).detach().cpu()
    man_mass = avg_attn[vision_token_start_idx:vision_token_start_idx + grid].sum().item()
    flower_mass = avg_attn[vision_token_start_idx + grid:vision_token_start_idx + 2 * grid].sum().item()
    cat_mass = avg_attn[vision_token_start_idx + 2 * grid:vision_token_end_idx].mean().item()
    
    return man_mass, flower_mass, cat_mass

def compute_attention_mass(last_layer_attn, vision_token_start_idx, vision_token_end_idx, prompt_end_index):
    avg_attn = last_layer_attn[0].mean(dim=0).squeeze(0).detach().cpu()
    video_mass = avg_attn[vision_token_start_idx:vision_token_end_idx].sum().item()
    total_input_mass = avg_attn[:prompt_end_index].sum().item()
    prompt_mass = total_input_mass - video_mass
    generated_mass = avg_attn[prompt_end_index:].sum().item()
    return video_mass, prompt_mass, generated_mass

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

        vs_id  = self.model.config.vision_start_token_id       # e.g., 151652
        ve_id  = self.model.config.vision_end_token_id         # e.g., 151653
        img_id = self.model.config.image_token_id
        vid_id = self.model.config.video_token_id
        vstart = input_ids[0].tolist().index(vs_id)
        vend = torch.where(input_ids[0] == ve_id)[0].max().item()

        model_kwargs_no_v = copy.deepcopy(model_kwargs)

        zero_visual_embed_hook = VisualZeroHook(vstart + 1, vend)
        visual_alpha = getattr(generation_config, 'visual_alpha', 0.0)

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            forward_call = self if is_prefill else model_forward

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = forward_call(**model_inputs, return_dict=True)

            if visual_alpha > 0:
                ### Start of VGD ###
                inputs_no_v = self.prepare_inputs_for_generation(input_ids, **model_kwargs_no_v)
                hook_handle = self.model.language_model.layers[0].register_forward_pre_hook(zero_visual_embed_hook)

                with torch.no_grad():
                    outputs_no_v = forward_call(**inputs_no_v, return_dict=True)
                hook_handle.remove()

                next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
                next_token_logits_no_v = outputs_no_v.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

                # VGD Logic: Uncond + alpha * (Cond - Uncond)
                next_token_logits = next_token_logits_no_v + visual_alpha * (next_token_logits - next_token_logits_no_v)
                 ### End of VGD ###
            else:
                next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
            
            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

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
            
            if visual_alpha > 0:
                model_kwargs_no_v = self._update_model_kwargs_for_generation(
                    outputs_no_v,
                    model_kwargs_no_v,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                    )
                del outputs_no_v

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

def evolve_guidance_sampling(temperature=1.0, do_sample=True, visual_alpha=1.0):
    if temperature > 0 and do_sample and visual_alpha > 0:
        print("Using Visual Guidance Sampling")
        transformers.generation.utils.GenerationMixin._sample = _sample
    else:
        print("Using Regular Sampling")