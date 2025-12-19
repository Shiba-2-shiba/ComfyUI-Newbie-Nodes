import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Any, Dict, Optional
import os
import sys
from safetensors.torch import load_file

try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available")


class NewBieCLIP:

    def __init__(self, text_encoder, tokenizer, clip_model, clip_tokenizer, device="cuda", cpu_offload=False, processor=None, enable_jina_weights=True, weight_baseline_mode="mean", weight_strength=1.0, mask_normalization=True):
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.clip_model = clip_model
        self.clip_tokenizer = clip_tokenizer
        self.processor = processor
        self.device = device
        self.cpu_offload = cpu_offload
        self.original_device = device
        self.enable_jina_weights = enable_jina_weights
        self.weight_baseline_mode = weight_baseline_mode
        self.weight_strength = weight_strength
        self.mask_normalization = mask_normalization
        
        self.text_encoder.eval()
        self.clip_model.eval()
        
        if self.cpu_offload:
            self.text_encoder = self.text_encoder.to("cpu")
            self.clip_model = self.clip_model.to("cpu")
            torch.cuda.empty_cache()

    def _move_to_device(self):
        if self.cpu_offload:
            self.text_encoder = self.text_encoder.to(self.original_device)
            self.clip_model = self.clip_model.to(self.original_device)

    def _move_to_cpu(self):
        if self.cpu_offload:
            self.text_encoder = self.text_encoder.to("cpu")
            self.clip_model = self.clip_model.to("cpu")
            torch.cuda.empty_cache()

    def encode_from_tokens(self, tokens, return_pooled=False):
        if isinstance(tokens, list) and tokens and isinstance(tokens[0], tuple):
            return self.encode_token_weights([tokens])
        
        self._move_to_device()
        
        if isinstance(tokens, str):
            tokens = self.tokenize(tokens)
        
        with torch.no_grad():
            if hasattr(tokens, 'input_ids'):
                input_ids = tokens.input_ids
                attention_mask = tokens.attention_mask
            else:
                input_ids = tokens
                attention_mask = torch.ones_like(input_ids)
            
            gemma_outputs = self.text_encoder(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                output_hidden_states=True,
            )
            cap_feats = gemma_outputs.hidden_states[-2]
            
            batch_size = input_ids.shape[0]
            if hasattr(self, '_last_text'):
                clip_text = self._last_text
            else:
                clip_text = [""] * batch_size
            
            clip_inputs = self.clip_tokenizer(
                clip_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=8000
            ).to(self.device)
            
            clip_text_pooled = self.clip_model.get_text_features(input_ids=clip_inputs.input_ids)
            
            extra_conds = {
                "cap_feats": cap_feats,
                "cap_mask": attention_mask.to(self.device),
                "clip_text_pooled": clip_text_pooled,
                "pooled_output": clip_text_pooled,
            }
            
            result = [[cap_feats, extra_conds]]
            if return_pooled:
                result = (result, clip_text_pooled)
        
        self._move_to_cpu()
        return result
    
    def encode_text(self, text):
        self._last_text = text if isinstance(text, list) else [text]
        tokens = self.tokenize(text)
        return self.encode_from_tokens(tokens)
    
    def encode_from_tokens_scheduled(self, tokens, return_pooled=False):
        return self.encode_from_tokens(tokens, return_pooled)
    
    def encode_token_weights(self, token_weight_pairs):
        self._move_to_device()

        to_encode = []
        token_weights_list = []
        char_weight_spans = []
        max_token_len = 0
        has_weights = False

        print(f"\n[NewbieCLIP] ===== 开始权重处理 (Optimized) =====")

        for x in token_weight_pairs:
            tokens = [a[0] for a in x]
            weights = [a[1] for a in x]

            text = " ".join(str(t) for t in tokens if str(t))
            text = text.replace('\x00\x02', '(').replace('\x00\x01', ')')

            char_weights = []
            current_pos = 0
            for i, (token, weight) in enumerate(x):
                token_str = str(token).replace('\x00\x02', '(').replace('\x00\x01', ')')
                if token_str:
                    end_pos = current_pos + len(token_str)
                    char_weights.append((current_pos, end_pos, weight))
                    current_pos = end_pos
                    if i < len(x) - 1:
                        next_token = str(x[i+1][0]).replace('\x00\x02', '(').replace('\x00\x01', ')')
                        if next_token:
                            current_pos += 1

            char_weight_spans.append((text, char_weights))

            section_weights = [a for a in x if a[1] != 1.0]
            if section_weights:
                has_weights = True

            max_token_len = max(len(tokens), max_token_len)
            to_encode.append(tokens)
            token_weights_list.append(weights)

        real_sections = len(to_encode)

        if has_weights or real_sections == 0:
            to_encode.append([""] * max_token_len)
            token_weights_list.append([1.0] * max_token_len)
            char_weight_spans.append(("", []))

        all_gemma_outputs = []
        all_clip_embeddings = []
        all_pooled = []
        all_attention_masks = []
        all_gemma_token_weights = []
        all_clip_token_weights = []

        original_prompt = getattr(self, '_original_prompt', None)

        for i, (text, char_weights) in enumerate(char_weight_spans):
            temp_offload = self.cpu_offload
            self.cpu_offload = False 
            
            try:
                if i == 0 and original_prompt:
                    result = self._encode_text_direct_weighted_with_offsets(
                        text, char_weights, original_text=original_prompt
                    )
                else:
                    result = self._encode_text_direct_weighted_with_offsets(
                        text, char_weights
                    )
            finally:
                self.cpu_offload = temp_offload

            if isinstance(result, dict):
                cap_feats = result['cap_feats']
                attention_mask = result['cap_mask']
                pooled = result['clip_text_pooled']
                clip_embeddings = result.get('clip_text_embeddings')
                gemma_token_weights = result.get('gemma_token_weights')
                clip_token_weights = result.get('clip_token_weights')
            else:
                cap_feats = result[0][0]
                extra_dict = result[0][1]
                pooled = extra_dict.get("clip_text_pooled")
                clip_embeddings = extra_dict.get("clip_text_embeddings")
                attention_mask = extra_dict.get("cap_mask")
                gemma_token_weights = None
                clip_token_weights = None

            all_gemma_outputs.append(cap_feats)
            all_clip_embeddings.append(clip_embeddings)
            all_pooled.append(pooled)
            all_attention_masks.append(attention_mask)
            all_gemma_token_weights.append(gemma_token_weights)
            all_clip_token_weights.append(clip_token_weights)

        import torch

        if has_weights:
            weighted_gemma_outputs = []
            weighted_clip_outputs = []

            if self.weight_baseline_mode == "mean":
                print(f"[NewbieCLIP] Gemma使用均值baseline模式 (Optimized)")
                for k in range(real_sections):
                    gemma_emb = all_gemma_outputs[k].clone()
                    unweighted_gemma = all_gemma_outputs[k]

                    batch_size = gemma_emb.shape[0]
                    seq_len = gemma_emb.shape[1]

                    context_mean = unweighted_gemma.mean(dim=1, keepdim=True)

                    if all_gemma_token_weights[k] is not None:
                        token_weights = all_gemma_token_weights[k]
                        for i in range(batch_size):
                            for j in range(min(seq_len, len(token_weights))):
                                weight = token_weights[j]
                                if weight != 1.0:
                                    gemma_emb[i, j] = (unweighted_gemma[i, j] - context_mean[i, 0]) * weight + context_mean[i, 0]
                    else:
                        for i in range(batch_size):
                            for j in range(min(seq_len, len(token_weights_list[k]))):
                                weight = token_weights_list[k][j]
                                if weight != 1.0:
                                    gemma_emb[i, j] = (unweighted_gemma[i, j] - context_mean[i, 0]) * weight + context_mean[i, 0]

                    weighted_gemma_outputs.append(gemma_emb)

            elif self.weight_baseline_mode == "compel":
                print(f"[NewbieCLIP] Using Compel mode")
                empty_gemma = all_gemma_outputs[-1]
                for k in range(real_sections):
                    gemma_emb = all_gemma_outputs[k].clone()
                    batch_size = gemma_emb.shape[0]
                    seq_len = gemma_emb.shape[1]
                    
                    if all_gemma_token_weights[k] is not None:
                        token_weights = all_gemma_token_weights[k]
                        for i in range(batch_size):
                            for j in range(min(seq_len, len(token_weights))):
                                weight = token_weights[j]
                                if weight > 1.0:
                                    empty_token = empty_gemma[i, 0] if empty_gemma.shape[1] > 0 else torch.zeros_like(gemma_emb[i, j])
                                    gemma_emb[i, j] = (gemma_emb[i, j] - empty_token) * weight + empty_token
                                elif weight < 1.0:
                                    gemma_emb[i, j] = gemma_emb[i, j] * weight
                    weighted_gemma_outputs.append(gemma_emb)

            elif self.weight_baseline_mode == "attn_mask":
                 for k in range(real_sections):
                    weighted_gemma_outputs.append(all_gemma_outputs[k])

            else:
                print(f"[NewbieCLIP] Gemma使用空字符串baseline模式")
                empty_gemma = all_gemma_outputs[-1]

                for k in range(real_sections):
                    gemma_emb = all_gemma_outputs[k].clone()
                    batch_size = gemma_emb.shape[0]
                    seq_len = gemma_emb.shape[1]

                    if all_gemma_token_weights[k] is not None:
                        token_weights = all_gemma_token_weights[k]
                        for i in range(batch_size):
                            for j in range(min(seq_len, len(token_weights))):
                                weight = token_weights[j]
                                if weight != 1.0:
                                    empty_token = empty_gemma[i, 0] if empty_gemma.shape[1] > 0 else torch.zeros_like(gemma_emb[i, j])
                                    gemma_emb[i, j] = (gemma_emb[i, j] - empty_token) * weight + empty_token
                    else:
                        for i in range(batch_size):
                            for j in range(min(seq_len, len(token_weights_list[k]))):
                                weight = token_weights_list[k][j]
                                if weight != 1.0:
                                    empty_token = empty_gemma[i, 0] if empty_gemma.shape[1] > 0 else torch.zeros_like(gemma_emb[i, j])
                                    gemma_emb[i, j] = (gemma_emb[i, j] - empty_token) * weight + empty_token

                    weighted_gemma_outputs.append(gemma_emb)

            if self.enable_jina_weights and all_clip_embeddings and all_clip_embeddings[0] is not None:
                 for k in range(real_sections):
                    if k >= len(all_clip_embeddings): break
                    clip_emb = all_clip_embeddings[k].clone()
                    empty_clip = all_clip_embeddings[-1] if all_clip_embeddings[-1] is not None else torch.zeros_like(clip_emb)
                    
                    token_weights = all_clip_token_weights[k] if all_clip_token_weights[k] is not None else token_weights_list[k]
                    
                    for i in range(clip_emb.shape[0]):
                        for j in range(min(clip_emb.shape[1], len(token_weights))):
                            weight = token_weights[j]
                            if weight != 1.0:
                                empty_token = empty_clip[i, 0] if empty_clip.shape[1] > 0 else torch.zeros_like(clip_emb[i, j])
                                clip_emb[i, j] = (clip_emb[i, j] - empty_token) * weight + empty_token
                    weighted_clip_outputs.append(clip_emb)
            else:
                for k in range(real_sections):
                    if k < len(all_clip_embeddings):
                        weighted_clip_outputs.append(all_clip_embeddings[k])


            if len(weighted_gemma_outputs) > 1:
                final_gemma = torch.cat(weighted_gemma_outputs, dim=1)
            else:
                final_gemma = weighted_gemma_outputs[0] if weighted_gemma_outputs else all_gemma_outputs[0]

            mask_list = []
            for k in range(real_sections):
                if k < len(all_attention_masks) and all_attention_masks[k] is not None:
                    mask_list.append(all_attention_masks[k])
                elif k < len(weighted_gemma_outputs):
                    batch_size = weighted_gemma_outputs[k].shape[0]
                    seq_len = weighted_gemma_outputs[k].shape[1]
                    mask_list.append(torch.ones(batch_size, seq_len, dtype=torch.long, device=self.device))
            
            final_attention_mask = torch.cat(mask_list, dim=1) if len(mask_list) > 1 else mask_list[0] if mask_list else None

            if weighted_clip_outputs and weighted_clip_outputs[0] is not None:
                clip_to_concat = [c for c in weighted_clip_outputs if c is not None]
                if clip_to_concat:
                    weighted_clip_concat = torch.cat(clip_to_concat, dim=1) if len(clip_to_concat) > 1 else clip_to_concat[0]
                    clip_pooled_final = weighted_clip_concat.mean(dim=1) 
                else:
                    clip_pooled_final = all_pooled[0]
            else:
                clip_pooled_final = all_pooled[0]

        else:
            final_gemma = all_gemma_outputs[0] if all_gemma_outputs else None
            final_attention_mask = all_attention_masks[0] if all_attention_masks else None
            clip_pooled_final = all_pooled[0] if all_pooled else None

        print(f"[NewbieCLIP] 权重处理完成")
        self._move_to_cpu()

        extra_conds = {
            "cap_feats": final_gemma,
            "cap_mask": final_attention_mask,
            "clip_text_pooled": clip_pooled_final,
            "pooled_output": clip_pooled_final,
        }

        return [[final_gemma, extra_conds]]
    
    def tokenize(self, text):
        import re
        if re.search(r'[()\[\]{}]', text):
            return self._parse_weights_extended(text)
        
        if isinstance(text, str):
            text = [text]
        
        return self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=8000)
    
    def _parse_weights_extended(self, text):
        from comfy.sd1_clip import escape_important, token_weights
        import re
        
        text = text.replace('\\[', '<<<ESC_LBRACKET>>>')
        text = text.replace('\\]', '<<<ESC_RBRACKET>>>')
        text = text.replace('\\{', '<<<ESC_LBRACE>>>')
        text = text.replace('\\}', '<<<ESC_RBRACE>>>')
        text = text.replace('\\(', '<<<ESC_LPAREN>>>')
        text = text.replace('\\)', '<<<ESC_RPAREN>>>')
        def process_brackets(text):
            pattern = r'\[+([^\[\]]*(?:<<<ESC_[LR]BRACKET>>>[^\[\]]*)*)\]+'
            def replace_brackets(match):
                full_match = match.group(0)
                content = match.group(1)
                left_count = 0
                i = 0
                while i < len(full_match) and full_match[i] == '[':
                    left_count += 1
                    i += 1
                weight = 0.9 ** left_count
                return f"({content}:{weight:.3f})"
            while re.search(pattern, text):
                text = re.sub(pattern, replace_brackets, text)
            return text
        
        def process_braces(text):
            pattern = r'\{+([^{}]*(?:<<<ESC_[LR]BRACE>>>[^{}]*)*)\}+'
            def replace_braces(match):
                full_match = match.group(0)
                content = match.group(1)
                left_count = 0
                i = 0
                while i < len(full_match) and full_match[i] == '{':
                    left_count += 1
                    i += 1
                weight = 1.2 ** left_count
                return f"({content}:{weight:.3f})"
            while re.search(pattern, text):
                text = re.sub(pattern, replace_braces, text)
            return text
        
        text = process_brackets(text)
        text = process_braces(text)
        text = text.replace('<<<ESC_LBRACKET>>>', '\\[')
        text = text.replace('<<<ESC_RBRACKET>>>', '\\]')
        text = text.replace('<<<ESC_LBRACE>>>', '\\{')
        text = text.replace('<<<ESC_RBRACE>>>', '\\}')
        text = text.replace('<<<ESC_LPAREN>>>', '\\(')
        text = text.replace('<<<ESC_RPAREN>>>', '\\)')
        
        return token_weights(escape_important(text), 1.0)
    
    def _encode_text_direct(self, text, original_text=None):
        if isinstance(text, str):
            text = [text]

        self._last_text = text
        self._move_to_device()

        with torch.no_grad():
            clip_text = original_text if original_text is not None else text
            if isinstance(clip_text, str):
                clip_text = [clip_text]

            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=8000
            )

            input_ids = tokens.input_ids.to(self.device)
            attention_mask = tokens.attention_mask.to(self.device)

            gemma_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            cap_feats = gemma_outputs.hidden_states[-2]

            clip_inputs = self.clip_tokenizer(clip_text, return_tensors="pt", padding=True, truncation=True, max_length=8000).to(self.device)
            clip_text_pooled = self.clip_model.get_text_features(input_ids=clip_inputs.input_ids)

            extra_conds = {
                "cap_feats": cap_feats,
                "cap_mask": attention_mask,
                "clip_text_pooled": clip_text_pooled,
                "pooled_output": clip_text_pooled,
            }

            result = [[cap_feats, extra_conds]]

        self._move_to_cpu()
        return result

    def _encode_text_direct_weighted(self, text, original_text=None):
        if isinstance(text, str):
            text = [text]

        self._last_text = text
        self._move_to_device()

        with torch.no_grad():
            clip_text = original_text if original_text is not None else text
            if isinstance(clip_text, str):
                clip_text = [clip_text]

            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=8000
            )

            input_ids = tokens.input_ids.to(self.device)
            attention_mask = tokens.attention_mask.to(self.device)

            gemma_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            cap_feats = gemma_outputs.hidden_states[-2]

            clip_inputs = self.clip_tokenizer(clip_text, return_tensors="pt", padding=True, truncation=True, max_length=8000).to(self.device)
            clip_text_pooled = self.clip_model.get_text_features(input_ids=clip_inputs.input_ids)
            
            clip_text_embeddings = None
            if self.enable_jina_weights and hasattr(self.clip_model, '_last_hidden_states') and self.clip_model._last_hidden_states is not None:
                clip_text_embeddings = self.clip_model._last_hidden_states

            extra_conds = {
                "cap_feats": cap_feats,
                "cap_mask": attention_mask,
                "clip_text_pooled": clip_text_pooled,
                "clip_text_embeddings": clip_text_embeddings,
            }

            result = [[cap_feats, extra_conds]]

        self._move_to_cpu()
        return result

    def _encode_text_direct_weighted_with_offsets(self, text, char_weights, original_text=None):
        if isinstance(text, str):
            text = [text]

        self._last_text = text
        self._move_to_device()

        with torch.no_grad():
            clip_text = original_text if original_text is not None else text
            if isinstance(clip_text, str):
                clip_text = [clip_text]

            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=8000,
                return_offsets_mapping=True
            )

            input_ids = tokens.input_ids.to(self.device)
            attention_mask = tokens.attention_mask.to(self.device)

            gemma_token_weights = None
            if hasattr(tokens, 'offset_mapping') and tokens.offset_mapping is not None:
                gemma_token_weights = self._align_weights_with_offsets(
                    tokens.offset_mapping[0], char_weights, text[0]
                )
                print(f"[NewbieCLIP] Gemma: 对齐了{len(gemma_token_weights)}个token权重")

            gemma_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            cap_feats = gemma_outputs.hidden_states[-2]

            clip_inputs = self.clip_tokenizer(
                clip_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=8000,
                return_offsets_mapping=True
            ).to(self.device)

            clip_token_weights = None
            if hasattr(clip_inputs, 'offset_mapping') and clip_inputs.offset_mapping is not None:
                clip_token_weights = self._align_weights_with_offsets(
                    clip_inputs.offset_mapping[0], char_weights, clip_text[0]
                )
                print(f"[NewbieCLIP] Jina CLIP: 对齐了{len(clip_token_weights)}个token权重")

            clip_text_embeddings = None
            clip_text_pooled = self.clip_model.get_text_features(input_ids=clip_inputs.input_ids)

            if self.enable_jina_weights and hasattr(self.clip_model, '_last_hidden_states') and self.clip_model._last_hidden_states is not None:
                clip_text_embeddings = self.clip_model._last_hidden_states
                print(f"[NewbieCLIP] Jina CLIP: Using hook to capture token embeddings, shape={clip_text_embeddings.shape}")

            result = {
                "cap_feats": cap_feats,
                "cap_mask": attention_mask,
                "clip_text_pooled": clip_text_pooled,
                "clip_text_embeddings": clip_text_embeddings,
                "gemma_token_weights": gemma_token_weights,
                "clip_token_weights": clip_token_weights
            }

        self._move_to_cpu()
        return result

    def _align_weights_with_offsets(self, offset_mapping, char_weights, text):
        token_weights = []
        for token_start, token_end in offset_mapping:
            if token_start == token_end:
                token_weights.append(1.0)
                continue

            weight_sum = 0.0
            overlap_sum = 0

            for char_start, char_end, weight in char_weights:
                overlap_start = max(token_start, char_start)
                overlap_end = min(token_end, char_end)

                if overlap_start < overlap_end:
                    overlap_len = overlap_end - overlap_start
                    weight_sum += weight * overlap_len
                    overlap_sum += overlap_len

            if overlap_sum > 0:
                token_weights.append(weight_sum / overlap_sum)
            else:
                token_weights.append(1.0)

        return token_weights

    def encode_with_image(self, user_text, image=None, system_text=""):
        self._move_to_device()
        
        if not hasattr(self, 'processor') or self.processor is None:
            return self.encode_text(user_text)
        
        with torch.no_grad():
            if image is not None:
                messages = []
                if system_text and system_text.strip():
                    messages.append({
                        "role": "system",
                        "content": [{"type": "text", "text": system_text.strip()}]
                    })
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_text}
                    ]
                })
                
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(self.device)
                
                gemma_outputs = self.text_encoder(
                    input_ids=inputs.input_ids,
                    pixel_values=inputs.pixel_values if hasattr(inputs, 'pixel_values') else None,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=True,
                )
                cap_feats = gemma_outputs.hidden_states[-2]
                attention_mask = inputs.attention_mask
            else:
                text = [user_text] if isinstance(user_text, str) else user_text
                tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=8000).to(self.device)
                gemma_outputs = self.text_encoder(
                    input_ids=tokens.input_ids,
                    attention_mask=tokens.attention_mask,
                    output_hidden_states=True,
                )
                cap_feats = gemma_outputs.hidden_states[-2]
                attention_mask = tokens.attention_mask
            
            text_for_clip = [user_text] if isinstance(user_text, str) else user_text
            clip_inputs = self.clip_tokenizer(text_for_clip, return_tensors="pt", padding=True, truncation=True, max_length=8000).to(self.device)
            clip_text_pooled = self.clip_model.get_text_features(input_ids=clip_inputs.input_ids)
            
            extra_conds = {
                "cap_feats": cap_feats,
                "cap_mask": attention_mask,
                "clip_text_pooled": clip_text_pooled,
            }
            
            result = [[cap_feats, extra_conds]]
        
        self._move_to_cpu()
        return result

    def get_clip_features(self, text):
        self._move_to_device()
        if isinstance(text, str):
            text = [text]
        with torch.no_grad():
            clip_inputs = self.clip_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=8000
            ).to(self.device)
            clip_text_pooled = self.clip_model.get_text_features(input_ids=clip_inputs.input_ids)
        self._move_to_cpu()
        return clip_text_pooled


class NewBieCLIPLoader:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gemma_model_path": ("STRING", {
                    "default": "",
                    "description": "Path to Gemma3-4B-IT safetensors directory"
                }),
                "jina_clip_path": ("STRING", {
                    "default": "jinaai/jina-clip-v2",
                    "description": "Path to Jina CLIP model (HuggingFace name or local path)"
                }),
            },
            "optional": {
                "device": (["cuda", "cpu"], {
                    "default": "cuda"
                }),
                "dtype": (["bf16", "fp16", "fp32"], {
                    "default": "bf16"
                }),
                "cpu_offload": ("BOOLEAN", {
                    "default": False,
                    "description": "Offload models to CPU memory after inference"
                }),
                "enable_jina_weights": ("BOOLEAN", {
                    "default": True,
                    "description": "Enable weight processing for Jina CLIP (requires more memory)"
                }),
                "weight_baseline_mode": (["mean", "empty", "compel", "attn_bias", "hybrid"], {
                    "default": "mean",
                    "description": "Weight baseline mode: 'mean' for context-aware, 'empty' for traditional, 'compel' for advanced, 'attn_bias' for attention-based, 'hybrid' for combined approach"
                }),
                "weight_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "description": "Global weight strength multiplier (0=no effect, 1=normal, 2=double strength)"
                }),
                "mask_normalization": ("BOOLEAN", {
                    "default": True,
                    "description": "Normalize attention mask after weight application to maintain distribution"
                }),
            }
        }
    
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "loaders"
    TITLE = "NewBie CLIP Loader"

    def load_gemma_model(self, model_path: str, device: str, dtype: torch.dtype):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required")
        
        if not os.path.exists(model_path):
            try:
                print(f"Loading Gemma model from HuggingFace: {model_path}")
                text_encoder = AutoModel.from_pretrained(
                    model_path,
                    dtype=dtype,
                    device_map=device,
                    trust_remote_code=True
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                tokenizer.padding_side = "right"
                try:
                    processor = AutoProcessor.from_pretrained(
                        model_path,
                        trust_remote_code=True
                    )
                except:
                    processor = None
                return text_encoder, tokenizer, processor
            except Exception as e:
                raise FileNotFoundError(f"Cannot load Gemma model from {model_path}: {e}")
        
        print(f"Loading Gemma model from local path: {model_path}")
        
        try:
            text_encoder = AutoModel.from_pretrained(
                model_path,
                dtype=dtype,
                device_map=device,
                trust_remote_code=True,
                local_files_only=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
            tokenizer.padding_side = "right"
            try:
                processor = AutoProcessor.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except:
                processor = None
            return text_encoder, tokenizer, processor
            
        except Exception as e:
            print(f"Direct loading failed: {e}")
            print("Falling back to manual loading...")
            
            config_path = os.path.join(model_path, "config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"config.json not found in {model_path}")
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True
            )
            tokenizer.padding_side = "right"
            try:
                processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            except:
                processor = None
            
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            text_encoder = AutoModel.from_config(config, torch_dtype=torch.float32)
            
            safetensors_path = os.path.join(model_path, "model.safetensors")
            if os.path.exists(safetensors_path):
                state_dict = load_file(safetensors_path, device="cpu")
            else:
                import glob
                shard_files = glob.glob(os.path.join(model_path, "model-*.safetensors"))
                if shard_files:
                    state_dict = {}
                    for shard_file in sorted(shard_files):
                        shard_dict = load_file(shard_file, device="cpu")
                        state_dict.update(shard_dict)
                else:
                    raise FileNotFoundError(f"No safetensors files found in {model_path}")
            
            if any(key.startswith("language_model.model.") for key in state_dict.keys()):
                state_dict = {
                    key.replace("language_model.model.", "language_model.") if key.startswith("language_model.model.") else key: value
                    for key, value in state_dict.items()
                }
            
            missing_keys, unexpected_keys = text_encoder.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"Unexpected keys: {len(unexpected_keys)}")
            
            text_encoder = text_encoder.to(device=device, dtype=dtype)
            
            return text_encoder, tokenizer, processor

    def load_jina_clip(self, model_path: str, device: str, dtype: torch.dtype, enable_jina_weights: bool = True):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required")

        print(f"Loading Jina CLIP model: {model_path}")
        print(f"Device: {device}, Dtype: {dtype}")

        # --- Flash Attention 無効化ハック ---
        # Jina CLIPのremote codeがflash_attnを強制的にインポートしてクラッシュするのを防ぐため、
        # 一時的にflash_attnが見つからないふりをするContext Managerを定義
        import sys
        class SuppressFlashAttn:
            def __enter__(self):
                self.saved_flash = sys.modules.get("flash_attn")
                self.saved_flash_2 = sys.modules.get("flash_attn.losses.cross_entropy")
                # flash_attnをNoneに設定してImportErrorを誘発させる
                sys.modules["flash_attn"] = None
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                # 復元
                if self.saved_flash is not None:
                    sys.modules["flash_attn"] = self.saved_flash
                else:
                    # 元々ロードされていなければ削除（Noneキーを残さない）
                    if "flash_attn" in sys.modules:
                        del sys.modules["flash_attn"]
                
                # サブモジュールも念のため復元処理
                if self.saved_flash_2 is not None:
                     sys.modules["flash_attn.losses.cross_entropy"] = self.saved_flash_2

        try:
            with SuppressFlashAttn():
                config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                config.use_flash_attn = False
                
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, 
                    trust_remote_code=True
                )
                
                clip_model = AutoModel.from_pretrained(
                    model_path,
                    config=config,
                    dtype=dtype,
                    device_map=device,
                    trust_remote_code=True
                )

            if enable_jina_weights and hasattr(clip_model, 'text_model'):
                print(f"[NewbieCLIP] Installing hook to capture hidden states")
                self._install_hidden_state_hook(clip_model, enable_jina_weights)

            return clip_model, tokenizer

        except Exception as e:
            print(f"Error in load_jina_clip: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _install_hidden_state_hook(self, clip_model, enable_jina_weights=True):
        if not enable_jina_weights:
            return

        clip_model._last_hidden_states = None

        def hook_fn(module, input, output):
            if hasattr(output, 'last_hidden_state'):
                clip_model._last_hidden_states = output.last_hidden_state
                print(f"[NewbieCLIP] Hook: Captured hidden states shape={output.last_hidden_state.shape}")
            elif isinstance(output, tuple):
                for out in output:
                    if isinstance(out, torch.Tensor) and out.dim() == 3:
                        clip_model._last_hidden_states = out
                        print(f"[NewbieCLIP] Hook: Captured hidden states from tuple, shape={out.shape}")
                        break

        if hasattr(clip_model.text_model, 'transformer'):
            handle = clip_model.text_model.transformer.register_forward_hook(hook_fn)
            clip_model._hook_handle = handle
            print(f"[NewbieCLIP] Hook registered on text_model.transformer")
        elif hasattr(clip_model.text_model, 'encoder'):
            handle = clip_model.text_model.encoder.register_forward_hook(hook_fn)
            clip_model._hook_handle = handle
            print(f"[NewbieCLIP] Hook registered on text_model.encoder")
        else:
            handle = clip_model.text_model.register_forward_hook(hook_fn)
            clip_model._hook_handle = handle
            print(f"[NewbieCLIP] Hook registered on text_model directly")

    def load_clip(
        self,
        gemma_model_path: str,
        jina_clip_path: str,
        device: str = "cuda",
        dtype: str = "bf16",
        cpu_offload: bool = False,
        enable_jina_weights: bool = True,
        weight_baseline_mode: str = "mean",
        weight_strength: float = 1.0,
        mask_normalization: bool = True
    ) -> Tuple[Any,]:
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32
        }
        torch_dtype = dtype_map[dtype]
        
        print(f"Loading NewBie CLIP models...")
        print(f"Gemma path: {gemma_model_path}")
        print(f"Jina CLIP path: {jina_clip_path}")
        print(f"Device: {device}, Dtype: {dtype}")
        
        text_encoder, tokenizer, processor = self.load_gemma_model(
            gemma_model_path, device, torch_dtype
        )
        
        clip_model, clip_tokenizer = self.load_jina_clip(
            jina_clip_path, device, torch_dtype, enable_jina_weights
        )
        
        newbie_clip = NewBieCLIP(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            clip_model=clip_model,
            clip_tokenizer=clip_tokenizer,
            device=device,
            cpu_offload=cpu_offload,
            processor=processor,
            enable_jina_weights=enable_jina_weights,
            weight_baseline_mode=weight_baseline_mode,
            weight_strength=weight_strength,
            mask_normalization=mask_normalization
        )
        
        return (newbie_clip,)


NODE_CLASS_MAPPINGS = {
    "NewBieCLIPLoader": NewBieCLIPLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NewBieCLIPLoader": "NewBie CLIP Loader",
}
