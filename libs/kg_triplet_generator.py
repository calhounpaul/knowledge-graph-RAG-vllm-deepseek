"""
Knowledge Graph Triplet Generator Module

This module leverages DeepSeek Chat API to generate knowledge graph triplets
from text input, creating structured data for knowledge graph construction.
"""

import json
from typing import List, Dict, Any, Optional, Union
from libs.query_llm import DeepSeekChat


class KnowledgeGraphTripletGenerator:
    """
    A class for generating knowledge graph triplets using DeepSeek Chat API.
    """
    
    def __init__(self, system_prompt: Optional[str] = None):
        """
        Initialize the triplet generator with a DeepSeek Chat client.
        
        Args:
            system_prompt: Optional custom system prompt for the conversation.
        """
        self.client = DeepSeekChat()
        
        if system_prompt is None:
            self.system_prompt = """
You are a knowledge graph expert. Your task is to extract comprehensive triplets from the provided text.
Each triplet should be in the format: [Subject, Predicate, Object].
Extract ALL meaningful relationships including:
- Entity-to-entity relationships
- Entity-to-property relationships
- Hierarchical relationships (is-a, part-of)
- Causal relationships (causes, results-in)
- Temporal relationships (before, after, during)
Be exhaustive and capture every possible relationship.
Return the triplets as a JSON array of arrays.
"""
        else:
            self.system_prompt = system_prompt
    
    def generate_triplets(self, text: str) -> List[List[str]]:
        """
        Generate triplets from the provided text.
        
        Args:
            text: The input text to extract triplets from.
            
        Returns:
            A list of triplets, where each triplet is a list of three strings:
            [subject, predicate, object]
        """
        # Create a conversation with the system prompt
        conversation = self.client.conversation(system_prompt=self.system_prompt)
        
        # Construct the prompt for triplet extraction
        prompt = f"""
Please extract knowledge graph triplets from the following text:

TEXT:
{text}

Extract key relationships in the format [Subject, Predicate, Object]. 
Return ONLY a valid JSON array of arrays without any additional explanation or text.
Example format:
[
    ["Einstein", "won", "Nobel Prize"],
    ["Relativity", "described_by", "Einstein"],
    ...
]
        """
        
        # Add the user message and get the response
        conversation.add_user_message(prompt)
        response = conversation.get_response()
        
        # Parse the JSON response to get the triplets
        try:
            # Find the JSON part in the response (in case there's explanatory text)
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                triplets = json.loads(json_str)
            else:
                # If no JSON format is found, try to parse the entire response
                triplets = json.loads(response)
                
            # Validate that triplets are in the expected format
            validated_triplets = []
            for triplet in triplets:
                if isinstance(triplet, list) and len(triplet) == 3:
                    validated_triplets.append([str(item) for item in triplet])
            
            return validated_triplets
        except json.JSONDecodeError:
            # If parsing fails, try to extract triplets using a more robust method
            return self._extract_triplets_fallback(response)
    
    def _extract_triplets_fallback(self, response: str) -> List[List[str]]:
        """
        Fallback method to extract triplets if JSON parsing fails.
        
        Args:
            response: The response from the LLM.
            
        Returns:
            A list of triplets.
        """
        # Create a new conversation to ask for properly formatted triplets
        conversation = self.client.conversation(system_prompt=self.system_prompt)
        
        prompt = f"""
The following response contains knowledge graph triplets, but they are not properly formatted as JSON.
Please convert this into a valid JSON array of triplets in the format [subject, predicate, object]:

{response}

Return ONLY a valid JSON array without any additional text.
        """
        
        conversation.add_user_message(prompt)
        retry_response = conversation.get_response()
        
        try:
            # Find the JSON part in the response
            json_start = retry_response.find("[")
            json_end = retry_response.rfind("]") + 1
            
            if json_start != -1 and json_end != -1:
                json_str = retry_response[json_start:json_end]
                triplets = json.loads(json_str)
                
                # Validate that triplets are in the expected format
                validated_triplets = []
                for triplet in triplets:
                    if isinstance(triplet, list) and len(triplet) == 3:
                        validated_triplets.append([str(item) for item in triplet])
                
                return validated_triplets
            else:
                # If still can't find JSON, return an empty list
                return []
        except json.JSONDecodeError:
            # If parsing still fails, return an empty list
            return []
    
    def batch_generate_triplets(self, texts: List[str], chunk_size: int = 2000) -> List[List[str]]:
        """
        Generate triplets from multiple texts or from a large text split into chunks.
        
        Args:
            texts: A list of text strings to process.
            chunk_size: The maximum size of text chunks if splitting is needed.
            
        Returns:
            A combined list of triplets from all texts.
        """
        all_triplets = []
        
        for text in texts:
            # If the text is too large, split it into chunks
            if len(text) > chunk_size:
                chunks = self._split_text(text, chunk_size)
                for chunk in chunks:
                    triplets = self.generate_triplets(chunk)
                    all_triplets.extend(triplets)
            else:
                triplets = self.generate_triplets(text)
                all_triplets.extend(triplets)
        
        # Remove duplicates
        unique_triplets = []
        triplet_strs = set()
        
        for triplet in all_triplets:
            triplet_str = json.dumps(triplet)
            if triplet_str not in triplet_strs:
                triplet_strs.add(triplet_str)
                unique_triplets.append(triplet)
        
        return unique_triplets
    
    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """
        Split a large text into smaller chunks for processing.
        
        Args:
            text: The text to split.
            chunk_size: The maximum size of each chunk.
            
        Returns:
            A list of text chunks.
        """
        # Try to split at paragraph boundaries
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                if current_chunk:
                    current_chunk += '\n\n' + paragraph
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If the paragraph itself is too large, split it into sentences
                if len(paragraph) > chunk_size:
                    sentences = [s.strip() + '.' for s in paragraph.split('.') if s.strip()]
                    current_chunk = ""
                    
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                            if current_chunk:
                                current_chunk += ' ' + sentence
                            else:
                                current_chunk = sentence
                        else:
                            chunks.append(current_chunk)
                            current_chunk = sentence
                else:
                    current_chunk = paragraph
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def save_triplets_to_json(self, triplets: List[List[str]], output_file: str) -> None:
        """
        Save the generated triplets to a JSON file.
        
        Args:
            triplets: The list of triplets to save.
            output_file: The path to the output JSON file.
        """
        # Convert triplets to a more readable format
        formatted_triplets = []
        for triplet in triplets:
            if len(triplet) == 3:
                formatted_triplets.append({
                    "subject": triplet[0],
                    "predicate": triplet[1],
                    "object": triplet[2]
                })
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_triplets, f, indent=2, ensure_ascii=False)


def generate_triplets_from_file(input_file: str, output_file: str, system_prompt: Optional[str] = None) -> None:
    """
    Generate triplets from a text file and save them to a JSON file.
    
    Args:
        input_file: The path to the input text file.
        output_file: The path to the output JSON file.
        system_prompt: Optional custom system prompt.
    """
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create the triplet generator
    generator = KnowledgeGraphTripletGenerator(system_prompt=system_prompt)
    
    # Generate triplets
    triplets = generator.generate_triplets(text)
    
    # Save triplets to JSON
    generator.save_triplets_to_json(triplets, output_file)