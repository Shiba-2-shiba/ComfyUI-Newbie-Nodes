# src/nodes.py
import os
import copy
import re
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Optional

try:
    from server import PromptServer
except Exception:
    PromptServer = None

# --- Helpers ---
def prettify_xml_elem(elem: ET.Element) -> str:
    """
    Return a pretty-printed XML string for a single Element, 
    stripping the XML declaration to suit prompt formats.
    """
    try:
        rough = ET.tostring(elem, encoding="utf-8")
        parsed = minidom.parseString(rough)
        pretty = parsed.toprettyxml(indent="  ")
        # Remove the <?xml version="1.0" ?> line if present
        lines = pretty.split('\n')
        if lines and lines[0].strip().startswith('<?xml'):
            lines = lines[1:]
        # Remove empty lines from start/end
        return "\n".join([l for l in lines if l.strip()])
    except Exception:
        # Fallback
        try:
            compact = ET.tostring(elem, encoding="unicode")
            return compact.replace('><', '>' + os.linesep + '<')
        except Exception:
            return ""

def safe_parse_fragment(fragment: str) -> Optional[ET.Element]:
    fragment = (fragment or "").strip()
    if not fragment:
        return None
    try:
        return ET.fromstring(fragment)
    except Exception:
        return None

def sanitize_tag(text: Optional[str]) -> str:
    if not text:
        return "character"
    s = text.strip()
    s = s.replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_\-\.]", "", s)
    if not re.match(r"^[A-Za-z_]", s):
        s = "char_" + s
    return s or "character"

def extract_elements_from_fragment(fragment: str) -> Optional[List[ET.Element]]:
    fragment = (fragment or "").strip()
    if not fragment:
        return None
    parsed = safe_parse_fragment(fragment)
    if parsed is not None:
        if parsed.tag == "custom_section":
            return [copy.deepcopy(child) for child in parsed]
        else:
            return [copy.deepcopy(parsed)]
    wrapped = f"<custom_section>{fragment}</custom_section>"
    parsed_wrapped = safe_parse_fragment(wrapped)
    if parsed_wrapped is not None and parsed_wrapped.tag == "custom_section":
        return [copy.deepcopy(child) for child in parsed_wrapped]
    return None

def append_elements_to_parent(parent: ET.Element, elements: List[ET.Element]):
    for el in elements:
        if el.tag == "custom_section":
            append_elements_to_parent(parent, list(el))
        else:
            parent.append(copy.deepcopy(el))

def find_name_text_in_elements(elements: List[ET.Element]) -> Optional[str]:
    for el in elements:
        if el.tag in ["name", "n"] and (el.text or "").strip():
            return (el.text or "").strip()
        for desc in el.iter():
            if desc.tag in ["name", "n"] and (desc.text or "").strip():
                return (desc.text or "").strip()
    return None

def add_child_if_text(parent: ET.Element, tag: str, text: str):
    if text is None:
        return None
    t = (text or "").strip()
    if t == "":
        return None
    child = ET.SubElement(parent, tag)
    child.text = t
    return child

# --- New Helpers for Tag Formatting ---
def format_tags(text: str) -> str:
    """
    Split by comma, strip whitespace, and replace spaces with underscores.
    Used for standard tag fields.
    """
    if not text:
        return ""
    tags = []
    for t in text.split(","):
        clean_t = t.strip()
        if clean_t:
            tags.append(clean_t.replace(" ", "_"))
    return ", ".join(tags)

def format_natural_language(text: str) -> str:
    """
    Remove underscores and treat as natural language.
    Used for the caption fields.
    """
    if not text:
        return ""
    return text.strip().replace("_", " ")

# -------------------------
# Character Node
# -------------------------
class CharacterNode:
    CATEGORY = "Prompt/Builder"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "character_1"}), 
                "gender": ("STRING", {"default": "1girl"}),
            },
            "optional": {
                "appearance": ("STRING", {"multiline": True}),
                "clothing": ("STRING", {"multiline": True}),
                "body_type": ("STRING", {"multiline": True}),
                "expression": ("STRING", {"multiline": True}),
                "action": ("STRING", {"multiline": True}),
                "interaction": ("STRING", {"multiline": True}),
                "position": ("STRING", {"multiline": True}),
                "description_nl": ("STRING", {"multiline": True, "placeholder": "Natural language description (uses <caption>)"}),
                "custom_section": ("STRING",),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "build_character_xml"

    def build_character_xml(self, name, gender,
                            appearance="", clothing="", body_type="",
                            expression="", action="", interaction="", position="",
                            description_nl="", custom_section=""):
        elem_name = (name or "character").strip().replace(" ", "_") or "character"
        root = ET.Element(elem_name)

        # Standard tags with auto-underscore
        add_child_if_text(root, "n", name) 
        add_child_if_text(root, "gender", format_tags(gender))
        add_child_if_text(root, "appearance", format_tags(appearance))
        add_child_if_text(root, "clothing", format_tags(clothing))
        add_child_if_text(root, "body_type", format_tags(body_type))
        add_child_if_text(root, "expression", format_tags(expression))
        add_child_if_text(root, "action", format_tags(action))
        add_child_if_text(root, "interaction", format_tags(interaction))
        add_child_if_text(root, "position", format_tags(position))

        # Natural Language -> <caption>
        add_child_if_text(root, "caption", format_natural_language(description_nl))

        # Custom section
        cs = (custom_section or "").strip()
        if cs:
            elems = extract_elements_from_fragment(cs)
            if elems is not None:
                append_elements_to_parent(root, elems)
            else:
                add_child_if_text(root, "custom_section", cs)

        return (ET.tostring(root, encoding="unicode"),)


# -------------------------
# Character Custom Builder
# -------------------------
class CharacterCustomBuilder:
    CATEGORY = "Prompt/Builder"

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, 11):
            optional[f"tag_name_{i}"] = ("STRING",)
            optional[f"tag_content_{i}"] = ("STRING", {"multiline": True})
        return {"required": {}, "optional": optional}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "build_character_custom"

    def build_character_custom(self, **kwargs):
        root = ET.Element("custom_section")
        for i in range(1, 11):
            name_key = f"tag_name_{i}"
            content_key = f"tag_content_{i}"
            tag_name = (kwargs.get(name_key, "") or "").strip()
            tag_content = (kwargs.get(content_key, "") or "").strip()
            if tag_name and tag_content:
                safe_tag = sanitize_tag(tag_name)
                child = ET.SubElement(root, safe_tag)
                child.text = tag_content
        return (ET.tostring(root, encoding="unicode"),)


# -------------------------
# General Tag Builder
# -------------------------
class GeneralTagBuilder:
    CATEGORY = "Prompt/Builder"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "count": ("STRING", {"multiline": True}),
                "artists": ("STRING", {"multiline": True}),
                "style": ("STRING", {"multiline": True}),
                "background": ("STRING", {"multiline": True}),
                "environment": ("STRING", {"multiline": True}),
                "perspective": ("STRING", {"multiline": True}),
                "atmosphere": ("STRING", {"multiline": True}),
                "lighting": ("STRING", {"multiline": True}),
                "quality": ("STRING", {"multiline": True}),
                "objects": ("STRING", {"multiline": True}),
                "other": ("STRING", {"multiline": True}),
                "description_nl": ("STRING", {"multiline": True, "placeholder": "Natural language description (uses <caption>)"}),
                "custom_section": ("STRING",),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "build_general_tags"

    def build_general_tags(self, count=None, artists=None, style=None, background=None,
                           environment=None, perspective=None, atmosphere=None,
                           lighting=None, quality=None, objects=None, other=None,
                           description_nl=None, custom_section=""):
        root = ET.Element("general_tags")
        
        # Auto-underscore for tags
        add_child_if_text(root, "count", format_tags(count))
        add_child_if_text(root, "artists", format_tags(artists))
        add_child_if_text(root, "style", format_tags(style))
        add_child_if_text(root, "background", format_tags(background))
        add_child_if_text(root, "environment", format_tags(environment))
        add_child_if_text(root, "perspective", format_tags(perspective))
        add_child_if_text(root, "atmosphere", format_tags(atmosphere))
        add_child_if_text(root, "lighting", format_tags(lighting))
        add_child_if_text(root, "quality", format_tags(quality))
        add_child_if_text(root, "objects", format_tags(objects))
        add_child_if_text(root, "other", format_tags(other))
        
        # Natural Language -> <caption>
        add_child_if_text(root, "caption", format_natural_language(description_nl))

        cs = (custom_section or "").strip()
        if cs:
            elems = extract_elements_from_fragment(cs)
            if elems is not None:
                if len(elems) == 1 and elems[0].tag == "general_tags":
                    append_elements_to_parent(root, list(elems[0]))
                else:
                    append_elements_to_parent(root, elems)
            else:
                add_child_if_text(root, "custom_section", cs)

        return (ET.tostring(root, encoding="unicode"),)


# -------------------------
# General Tag Custom Builder
# -------------------------
class GeneralTagCustomBuilder:
    CATEGORY = "Prompt/Builder"

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, 11):
            optional[f"tag_name_{i}"] = ("STRING",)
            optional[f"tag_content_{i}"] = ("STRING", {"multiline": True})
        return {"required": {}, "optional": optional}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "build_general_tag_custom"

    def build_general_tag_custom(self, **kwargs):
        root = ET.Element("general_tags")
        for i in range(1, 11):
            name_key = f"tag_name_{i}"
            content_key = f"tag_content_{i}"
            tag_name = (kwargs.get(name_key, "") or "").strip()
            tag_content = (kwargs.get(content_key, "") or "").strip()
            if tag_name and tag_content:
                safe_tag = sanitize_tag(tag_name)
                child = ET.SubElement(root, safe_tag)
                child.text = tag_content
        return (ET.tostring(root, encoding="unicode"),)


# -------------------------
# XML Assembler with SLOTS
# -------------------------
class XMLAssemblerSlots:
    CATEGORY = "Prompt/Builder"

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, 12):
            optional[f"slot_{i}"] = ("STRING",)
        return {"required": {}, "optional": optional}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "assemble_xml"

    def assemble_xml(self, **kwargs):
        # We will collect element objects, but not put them in a root element yet
        # to avoid the <document> wrapper and XML declaration.
        collected_elements = []

        for i in range(1, 12):
            key = f"slot_{i}"
            slot_text = (kwargs.get(key, "") or "").strip()
            if not slot_text:
                continue

            parsed = safe_parse_fragment(slot_text)
            if parsed is not None:
                if parsed.tag == "custom_section":
                    # Unpack custom section into a wrapper tag based on name
                    name_child = parsed.find('.//n') 
                    if not name_child:
                         name_child = parsed.find('.//name')
                    
                    wrapper_name = "character"
                    if name_child is not None and (name_child.text or "").strip():
                        wrapper_name = sanitize_tag(name_child.text)
                    
                    wrapper = ET.Element(wrapper_name)
                    append_elements_to_parent(wrapper, list(parsed))
                    collected_elements.append(wrapper)
                else:
                    collected_elements.append(copy.deepcopy(parsed))
            else:
                # Handle raw text or fragments
                elems = extract_elements_from_fragment(slot_text)
                if elems is not None:
                    if len(elems) > 1:
                        name_text = find_name_text_in_elements(elems)
                        wrapper_name = sanitize_tag(name_text) if name_text else "character"
                        wrapper = ET.Element(wrapper_name)
                        append_elements_to_parent(wrapper, elems)
                        collected_elements.append(wrapper)
                    else:
                        single = elems[0]
                        if single.tag == "custom_section":
                            name_text = find_name_text_in_elements(list(single))
                            wrapper_name = sanitize_tag(name_text) if name_text else "character"
                            wrapper = ET.Element(wrapper_name)
                            append_elements_to_parent(wrapper, list(single))
                            collected_elements.append(wrapper)
                        else:
                            collected_elements.append(single)
                else:
                    # Non-XML text -> wrap in character
                    wrapper = ET.Element("character")
                    wrapper.text = slot_text
                    collected_elements.append(wrapper)

        # Build final string: { \n [elements] \n }
        # Prettify each top-level element individually without XML declaration
        formatted_parts = []
        for elem in collected_elements:
            pretty_str = prettify_xml_elem(elem)
            if pretty_str:
                formatted_parts.append(pretty_str)
        
        # Join with newlines
        joined_content = "\n\n".join(formatted_parts)
        
        # Wrap in braces
        final_output = f"{{\n{joined_content}\n}}"

        # Output preview
        try:
            out_dir = os.path.join("outputs", "xml_prompts")
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "latest_preview.txt"), "w", encoding="utf-8") as f:
                f.write(final_output)
        except Exception:
            pass
            
        if PromptServer:
            try:
                PromptServer.instance.send_sync("xml_prompt_builder.preview", {"xml": final_output})
            except Exception:
                pass

        return (final_output,)


NODE_CLASS_MAPPINGS = {
    "Character Node": CharacterNode,
    "Character Custom Builder": CharacterCustomBuilder,
    "General Tag Builder": GeneralTagBuilder,
    "General Tag Custom Builder": GeneralTagCustomBuilder,
    "XML Assembler (slots)": XMLAssemblerSlots,
}
