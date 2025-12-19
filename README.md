# ComfyUI-Newbie-Nodes (Experimental Fork)

This repository is an experimental fork of NewBieAI-Lab/ComfyUI-Newbie-Nodes.

The purpose of this fork is to introduce experimental features (FreeU, MagCache), improve compatibility, and refine the XML prompt building logic for better tag handling.

## ⚠️ Disclaimer

This is an experimental version. Some behaviors, especially the XML output format, differ from the original repository. Please use it at your own risk.

## Key Changes & Features

### 1. New Features: FreeU & MagCache

I have added new nodes to enhance image quality and inference speed.

### NewBie FreeU-Like Patch (NewBieFreeULikeNode)

Implements a FreeU-like adjustment specifically adapted for the NewBie (NextDiT) architecture.

Scales the backbone features in the first and middle stages of the model to improve contrast and texture details.

### MagCache System (MagCacheNewBie)

A caching mechanism to accelerate inference.

Skips computation for steps with small residuals based on calibration data.

Includes Calibration mode (to measure model behavior) and Inference mode (to apply caching).

## 2. Modified: XML Prompt Builder Logic

I have modified CharacterNode and GeneralTagBuilder to distinguish between "Tag style" and "Natural Language style" prompts.

Auto-Underscore for Tags:
 Inputs for standard tags (e.g., appearance, clothing) now automatically replace spaces with underscores (e.g., black hair $\to$ black_hair). This makes it easier to use Danbooru-style tags.

Tag Name Change:
 Changed the character name tag from <name> to <n> to shorten the prompt structure.

Added Caption Support:
 Added a new input field: description_nl (Natural Language).
 
 Text in this field is output to a <caption> tag and is NOT converted to underscores (preserves spaces). This is intended for natural language sentences.

## 3. Fixed: Flash Attention Compatibility

Disabled Flash Attention: Modified NewBieCLIPLoader to forcibly disable Flash Attention usage.

This ensures the nodes work correctly in environments where flash_attn is not installed or causes dependency errors.

# Credits
This project is based on the excellent work by NewBieAI-Lab.  
