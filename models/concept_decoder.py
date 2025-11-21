"""
Concept Decoder: Expand semantic concepts to BPE tokens.

Maps concept IDs → text via stored expansion table or on-the-fly generation.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict
import json
import os


class ConceptExpansionTable:
    """
    Stores learned expansions: concept_id → BPE token sequence.
    
    Built during training by clustering frequent phrases/patterns.
    """
    
    def __init__(self, num_concepts: int = 2048):
        self.num_concepts = num_concepts
        # concept_id → list of BPE token IDs
        self.expansions: Dict[int, List[int]] = {}
        # concept_id → original text (for debugging)
        self.concept_texts: Dict[int, str] = {}
    
    def add_concept(self, concept_id: int, bpe_tokens: List[int], text: str = ""):
        """Store expansion for a concept."""
        self.expansions[concept_id] = bpe_tokens
        self.concept_texts[concept_id] = text
    
    def get_expansion(self, concept_id: int) -> Optional[List[int]]:
        """Get BPE expansion for concept."""
        return self.expansions.get(concept_id)
    
    def save(self, path: str):
        """Save expansion table to disk."""
        data = {
            'num_concepts': self.num_concepts,
            'expansions': {str(k): v for k, v in self.expansions.items()},
            'texts': {str(k): v for k, v in self.concept_texts.items()}
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """Load expansion table from disk."""
        if not os.path.exists(path):
            return
        with open(path, 'r') as f:
            data = json.load(f)
        self.num_concepts = data['num_concepts']
        self.expansions = {int(k): v for k, v in data['expansions'].items()}
        self.concept_texts = {int(k): v for k, v in data.get('texts', {}).items()}


class ConceptDecoder(nn.Module):
    """
    Decoder: Concept IDs + Control Symbols → Text.
    
    Generation flow:
    1. TRM outputs concept_id
    2. DQN decides: EMIT (direct) or EXPAND (to BPE children)
    3. If EXPAND: lookup expansion_table[concept_id] → BPE tokens
    4. If EMIT: concept represents abstract phrase (e.g., "greeting", "question")
    """
    
    def __init__(
        self,
        num_concepts: int = 2048,
        bpe_vocab_size: int = 50257,
        expansion_table_path: Optional[str] = None,
    ):
        super().__init__()
        
        self.num_concepts = num_concepts
        self.bpe_vocab_size = bpe_vocab_size
        
        # Expansion table (concept → BPE tokens)
        self.expansion_table = ConceptExpansionTable(num_concepts)
        if expansion_table_path:
            self.expansion_table.load(expansion_table_path)
        
        # Learnable concept embeddings (for generation without expansion)
        # Each concept has a "canonical" BPE sequence it represents
        self.concept_to_bpe = nn.Embedding(num_concepts, 8)  # Max 8 BPE tokens per concept
        nn.init.zeros_(self.concept_to_bpe.weight)
    
    def decode_sequence(
        self,
        concept_ids: torch.Tensor,
        control_symbols: torch.Tensor,
        expand_mask: torch.Tensor,
        tokenizer,
        show_thoughts: bool = True,
    ) -> List[str]:
        """
        Decode concept sequence to text.
        
        Args:
            concept_ids: [B, seq_len] concept IDs
            control_symbols: [B, seq_len] control symbol flags
            expand_mask: [B, seq_len] bool, True = expand to children
            tokenizer: BPE tokenizer for final decoding
        
        Returns:
            List of decoded text strings
        """
        batch_size = concept_ids.size(0)
        decoded_texts = []
        
        for b in range(batch_size):
            bpe_tokens = []
            
            for i in range(concept_ids.size(1)):
                cid = concept_ids[b, i].item()
                is_control = control_symbols[b, i].item()
                should_expand = expand_mask[b, i].item()
                
                # Control symbols
                if is_control:
                    if cid == self.num_concepts:  # <EXPAND>
                        continue  # Handled by expand_mask
                    elif cid == self.num_concepts + 1:  # <STOP>
                        break
                    elif cid == self.num_concepts + 2:  # <MERGE>
                        # Remove space between tokens
                        if bpe_tokens and bpe_tokens[-1] == tokenizer.encode(' ')[0]:
                            bpe_tokens.pop()
                    elif cid == self.num_concepts + 3:  # <THOUGHT>
                        if show_thoughts:
                            bpe_tokens.extend(tokenizer.encode("<THOUGHT>"))
                    elif cid == self.num_concepts + 4:  # <END_THOUGHT>
                        if show_thoughts:
                            bpe_tokens.extend(tokenizer.encode("<END_THOUGHT>"))
                    continue
                
                # Semantic concept
                if should_expand:
                    # Expand to children BPE tokens
                    expansion = self.expansion_table.get_expansion(cid)
                    if expansion:
                        bpe_tokens.extend(expansion)
                    else:
                        # Fallback: use learned embedding
                        learned_tokens = self.concept_to_bpe.weight[cid].long()
                        bpe_tokens.extend(learned_tokens[learned_tokens > 0].tolist())
                else:
                    # Direct emission (concept represents abstract idea)
                    # Use concept text if available
                    concept_text = self.expansion_table.concept_texts.get(cid, f"<C{cid}>")
                    concept_bpe = tokenizer.encode(concept_text)
                    bpe_tokens.extend(concept_bpe)
            
            # Decode BPE to text
            text = tokenizer.decode(bpe_tokens)
            decoded_texts.append(text)
        
        return decoded_texts
    
    def build_expansion_table_from_data(
        self,
        capsule_dataset_path: str,
        tokenizer,
        max_concepts: int = 2048,
    ):
        """
        Build expansion table from capsule dataset.
        
        Clusters frequent capsule sketches and assigns concept IDs.
        Stores BPE expansions from children tokens.
        """
        import torch
        from sklearn.cluster import MiniBatchKMeans
        
        print(f"\n📚 Building concept expansion table...")
        print(f"   Loading capsule dataset: {capsule_dataset_path}")
        
        data = torch.load(capsule_dataset_path, map_location='cpu')
        sketches = data['sketches']  # [N, k, D]
        texts = data.get('texts', [])
        
        # Flatten all capsule sketches
        all_sketches = sketches.reshape(-1, sketches.size(-1))  # [N*k, D]
        
        print(f"   Clustering {all_sketches.size(0)} capsules into {max_concepts} concepts...")
        
        # K-means clustering to find concept centroids
        kmeans = MiniBatchKMeans(
            n_clusters=max_concepts,
            batch_size=1000,
            max_iter=100,
            random_state=42
        )
        cluster_ids = kmeans.fit_predict(all_sketches.numpy())
        
        # Build expansion table from most common phrases per cluster
        print(f"   Building expansion mappings...")
        
        from collections import defaultdict
        cluster_texts = defaultdict(list)
        
        # Group texts by cluster
        idx = 0
        for n in range(len(texts)):
            for k in range(sketches.size(1)):
                cid = cluster_ids[idx]
                # Extract corresponding text chunk (approximate)
                if n < len(texts):
                    words = texts[n].split()
                    chunk_start = (k * len(words)) // sketches.size(1)
                    chunk_end = ((k + 1) * len(words)) // sketches.size(1)
                    chunk_text = ' '.join(words[chunk_start:chunk_end])
                    cluster_texts[cid].append(chunk_text)
                idx += 1
        
        # Store most frequent text per concept
        for concept_id in range(max_concepts):
            if concept_id in cluster_texts:
                # Take most common phrase (heuristic)
                texts_for_concept = cluster_texts[concept_id]
                if texts_for_concept:
                    # Use first as representative
                    representative_text = texts_for_concept[0]
                    bpe_tokens = tokenizer.encode(representative_text)
                    self.expansion_table.add_concept(
                        concept_id,
                        bpe_tokens,
                        representative_text
                    )
        
        print(f"   Built {len(self.expansion_table.expansions)} concept expansions")
        
        return self.expansion_table
