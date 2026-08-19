import os
from cloning import walk_repository

# Tree-sitter configurations for common programming languages
LANGUAGE_CONFIGS = {
    "python": {
        "extensions": [".py"],
        "class_nodes": {"class_definition"},
        "function_nodes": {"function_definition"},
    },
    "javascript": {
        "extensions": [".js", ".jsx", ".mjs", ".cjs"],
        "class_nodes": {"class_declaration", "class"},
        "function_nodes": {"function_declaration", "method_definition", "arrow_function"},
    },
    "typescript": {
        "extensions": [".ts", ".tsx"],
        "class_nodes": {"class_declaration", "interface_declaration", "enum_declaration"},
        "function_nodes": {"function_declaration", "method_definition", "arrow_function"},
    },
    "go": {
        "extensions": [".go"],
        "class_nodes": {"type_spec"},  # Go structs are declared in type specs
        "function_nodes": {"function_declaration", "method_declaration"},
    },
    "rust": {
        "extensions": [".rs"],
        "class_nodes": {"struct_item", "enum_item", "trait_item", "impl_item"},
        "function_nodes": {"function_item"},
    },
    "java": {
        "extensions": [".java"],
        "class_nodes": {"class_declaration", "interface_declaration", "enum_declaration"},
        "function_nodes": {"method_declaration"},
    },
    "cpp": {
        "extensions": [".cpp", ".cc", ".cxx", ".h", ".hpp"],
        "class_nodes": {"class_specifier", "struct_specifier"},
        "function_nodes": {"function_definition"},
    },
    "c": {
        "extensions": [".c", ".h"],
        "class_nodes": {"struct_specifier", "union_specifier"},
        "function_nodes": {"function_definition"},
    }
}

# Build mapping from extension to language name
EXTENSION_TO_LANGUAGE = {}
for lang, config in LANGUAGE_CONFIGS.items():
    for ext in config["extensions"]:
        EXTENSION_TO_LANGUAGE[ext.lower()] = lang


def extract_node_name(node, source_code: bytes) -> str:
    """
    Extract the name identifier of a class or function node.
    """
    name_types = {"identifier", "field_identifier", "property_identifier", "type_identifier"}
    for child in node.children:
        if child.type in name_types:
            return source_code[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
    
    # Check recursive children up to a depth of 2 for identifier
    for child in node.children:
        for grand_child in child.children:
            if grand_child.type in name_types:
                return source_code[grand_child.start_byte:grand_child.end_byte].decode("utf-8", errors="replace")
                
    return "anonymous"


def split_large_code(text: str, start_line: int, end_line: int, type_name: str, name: str) -> list[dict]:
    """
    Splits large code definitions by lines keeping context/blocks as contiguous chunks.
    """
    lines = text.splitlines()
    chunks = []
    
    current_lines = []
    current_len = 0
    start_offset = 0
    
    for i, line in enumerate(lines):
        # Limit chunks to ~1000 characters
        if current_len + len(line) + 1 > 1000 and current_lines:
            chunks.append({
                "type": type_name,
                "name": name,
                "text": "\n".join(current_lines),
                "start_line": start_line + start_offset,
                "end_line": start_line + i
            })
            # Overlap: include last 3 lines for context continuity
            overlap_lines = current_lines[-3:] if len(current_lines) >= 3 else current_lines
            current_lines = list(overlap_lines)
            current_len = sum(len(l) + 1 for l in current_lines)
            start_offset = i - len(current_lines) + 1
            
        current_lines.append(line)
        current_len += len(line) + 1
        
    if current_lines:
        chunks.append({
            "type": type_name,
            "name": name,
            "text": "\n".join(current_lines),
            "start_line": start_line + start_offset,
            "end_line": end_line
        })
        
    return chunks


def traverse_ast(node, source_code: bytes, config: dict, current_class: str = None) -> list[dict]:
    """
    Recursively traverses the syntax tree to extract class-like containers
    and function/method-like definitions semantically.
    """
    chunks = []
    node_type = node.type
    
    # Class-like container
    if node_type in config["class_nodes"]:
        class_name = extract_node_name(node, source_code)
        class_text = source_code[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        
        if len(class_text) <= 1000:
            chunks.append({
                "type": "class",
                "name": class_name,
                "text": class_text,
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1
            })
        else:
            # Locate body/block node
            body_node = None
            for child in node.children:
                if child.type in ("block", "class_body", "declaration_list", "member_declaration_list"):
                    body_node = child
                    break
            
            header_end = body_node.start_byte if body_node else node.end_byte
            header_text = source_code[node.start_byte:header_end].decode("utf-8", errors="replace").strip()
            
            chunks.append({
                "type": "class_header",
                "name": class_name,
                "text": header_text,
                "start_line": node.start_point[0] + 1,
                "end_line": (body_node.start_point[0] if body_node else node.end_point[0]) + 1
            })
            
            nodes_to_traverse = body_node.children if body_node else node.children
            for child in nodes_to_traverse:
                chunks.extend(traverse_ast(child, source_code, config, current_class=class_name))
        return chunks

    # Function-like container
    elif node_type in config["function_nodes"]:
        func_name = extract_node_name(node, source_code)
        
        # Check if parent wraps decorators/attributes or export statement
        parent = node.parent
        start_byte = node.start_byte
        start_point = node.start_point
        if parent and parent.type in ("decorated_definition", "method_declaration", "export_statement"):
            start_byte = parent.start_byte
            start_point = parent.start_point
            
        func_text = source_code[start_byte:node.end_byte].decode("utf-8", errors="replace")
        prefix = f"# Parent: {current_class}\n" if current_class else ""
        full_text = prefix + func_text
        full_name = f"{current_class}.{func_name}" if current_class else func_name
        
        if len(full_text) <= 1000:
            chunks.append({
                "type": "method" if current_class else "function",
                "name": full_name,
                "text": full_text,
                "start_line": start_point[0] + 1,
                "end_line": node.end_point[0] + 1
            })
        else:
            chunks.extend(split_large_code(full_text, start_point[0] + 1, node.end_point[0] + 1, 
                                           "method" if current_class else "function", full_name))
        return chunks

    # Skips parent decorator / export nodes as they are processed with inner class/function
    elif node_type in ("decorated_definition", "export_statement"):
        for child in node.children:
            if child.type in config["class_nodes"] or child.type in config["function_nodes"]:
                chunks.extend(traverse_ast(child, source_code, config, current_class))
        return chunks

    # Structural scopes (like namespace, packages, modules) are traversed further
    if node_type in ("namespace_definition", "package_declaration", "translation_unit", "module"):
        for child in node.children:
            chunks.extend(traverse_ast(child, source_code, config, current_class))
            
    return chunks


def chunk_general_nodes(nodes, source_code: bytes) -> list[dict]:
    """
    Groups sequential module/global-level nodes into chunks of at most 1000 characters.
    """
    chunks = []
    current_chunk_nodes = []
    current_length = 0
    
    for node in nodes:
        node_len = node.end_byte - node.start_byte
        if current_length + node_len > 1000 and current_chunk_nodes:
            chunks.append(create_general_chunk(current_chunk_nodes, source_code))
            current_chunk_nodes = []
            current_length = 0
            
        current_chunk_nodes.append(node)
        current_length += node_len
        
    if current_chunk_nodes:
        chunks.append(create_general_chunk(current_chunk_nodes, source_code))
        
    return chunks


def create_general_chunk(nodes, source_code: bytes) -> dict:
    start_byte = nodes[0].start_byte
    end_byte = nodes[-1].end_byte
    text = source_code[start_byte:end_byte].decode("utf-8", errors="replace")
    return {
        "type": "general",
        "name": "",
        "text": text,
        "start_line": nodes[0].start_point[0] + 1,
        "end_line": nodes[-1].end_point[0] + 1
    }


def chunk_code_file(file_content: bytes, file_path: str) -> list[dict]:
    """
    Parses a code file using Tree-Sitter and splits it semantically.
    """
    ext = os.path.splitext(file_path)[1].lower()
    lang = EXTENSION_TO_LANGUAGE.get(ext)
    
    if not lang:
        return chunk_text_fallback(file_content.decode("utf-8", errors="replace"), file_path)
        
    try:
        from tree_sitter_language_pack import get_parser
        parser = get_parser(lang)
        tree = parser.parse(file_content)
        root = tree.root_node
    except Exception as exc:
        print(f"Warning: tree-sitter parsing failed for {file_path}: {exc}. Falling back to text splitting.")
        return chunk_text_fallback(file_content.decode("utf-8", errors="replace"), file_path)
        
    config = LANGUAGE_CONFIGS[lang]
    chunks = []
    general_nodes = []
    
    for child in root.children:
        if (child.type in config["class_nodes"] or 
            child.type in config["function_nodes"] or 
            child.type in ("decorated_definition", "export_statement")):
            
            if general_nodes:
                chunks.extend(chunk_general_nodes(general_nodes, file_content))
                general_nodes = []
            
            chunks.extend(traverse_ast(child, file_content, config))
        else:
            general_nodes.append(child)
            
    if general_nodes:
        chunks.extend(chunk_general_nodes(general_nodes, file_content))
        
    # Inject file path and language metadata into each chunk
    for chunk in chunks:
        chunk["file_path"] = file_path
        chunk["language"] = lang
        
    return chunks


def chunk_text_fallback(text: str, file_path: str) -> list[dict]:
    """
    Robust fallback splitter that preserves whole lines and targets ~1000 characters per chunk.
    Used for documentation and unsupported languages.
    """
    lines = text.splitlines()
    chunks = []
    current_lines = []
    current_len = 0
    start_line = 1
    
    for i, line in enumerate(lines):
        if current_len + len(line) + 1 > 1000 and current_lines:
            chunks.append({
                "type": "doc_chunk",
                "name": "",
                "text": "\n".join(current_lines),
                "start_line": start_line,
                "end_line": i,
                "file_path": file_path,
                "language": "text"
            })
            overlap_lines = current_lines[-3:] if len(current_lines) >= 3 else current_lines
            current_lines = list(overlap_lines)
            current_len = sum(len(l) + 1 for l in current_lines)
            start_line = i - len(current_lines) + 2
            
        current_lines.append(line)
        current_len += len(line) + 1
        
    if current_lines:
        chunks.append({
            "type": "doc_chunk",
            "name": "",
            "text": "\n".join(current_lines),
            "start_line": start_line,
            "end_line": len(lines),
            "file_path": file_path,
            "language": "text"
        })
        
    return chunks


def load_and_chunk_file(file_path: str, repo_root: str) -> list[dict]:
    """
    Reads a file and returns its structured chunks.
    """
    try:
        with open(file_path, "rb") as f:
            content = f.read()
    except Exception as exc:
        print(f"Warning: could not read file {file_path}: {exc}")
        return []
        
    # Standardize paths to use forward slashes for cross-platform links
    rel_path = os.path.relpath(file_path, repo_root).replace("\\", "/")
    
    ext = os.path.splitext(file_path)[1].lower()
    if ext in EXTENSION_TO_LANGUAGE:
        return chunk_code_file(content, rel_path)
    else:
        # Check if it's likely a text/markdown/doc file
        # We can decode and split
        try:
            text = content.decode("utf-8")
            return chunk_text_fallback(text, rel_path)
        except Exception:
            # If not decodable as utf-8, skip binary files
            return []


def load_and_chunk_repository(repo_path: str) -> list[dict]:
    """
    Walks the repository and generates chunks for all source files and docs.
    """
    all_chunks = []
    file_paths = walk_repository(repo_path)
    for file_path in file_paths:
        chunks = load_and_chunk_file(file_path, repo_path)
        all_chunks.extend(chunks)
    return all_chunks
