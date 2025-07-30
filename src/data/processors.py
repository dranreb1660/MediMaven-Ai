"""
Data Processing Module
=====================

Extracted from my notebooks/v1.1/00b_data_collection_and_preprocessing.ipynb
Contains classes for data processing, cleaning, and chunking.
"""

import json
import uuid
import re
import datetime
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Generator, Tuple
from itertools import islice

import pandas as pd
import spacy
import tiktoken
from tqdm import tqdm


class DataProcessor:
    """Main data processor - handles all the medical text chunking logic I developed."""
    
    def __init__(self, window_size: int = 400, stride: int = 100):
        """
        Initialize with my chunking parameters.
        
        Args:
            window_size: Max tokens per chunk (I use 400)
            stride: Token overlap between chunks (I use 100 for good context)
        """
        self.window_size = window_size
        self.stride = stride
        self.today = datetime.date.today().isoformat()
        self.meta_keys = {"url", "title", "type"}  # My metadata fields
        
        # My tokenizer setup
        self.enc = tiktoken.get_encoding("cl100k_base")
        
        # My spaCy setup for sentence boundaries
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
        self.nlp.add_pipe("sentencizer")
    
    def n_tokens(self, txt: str) -> int:
        """My fast token counter using tiktoken."""
        return len(self.enc.encode(txt))
    
    def sentence_chunks(self, text: str) -> Generator[str, None, None]:
        """
        My sentence-aware chunking algorithm - preserves sentence boundaries.
        This is key for maintaining coherent chunks for RAG.
        
        Args:
            text: Input text to chunk
            
        Yields:
            Text chunks with sentence boundaries preserved
        """
        doc = self.nlp(text)
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        
        buf, tok_count = [], 0
        for sent in sents:
            s_tok = self.n_tokens(sent)
            
            # Start new chunk if adding this sentence would overflow
            if buf and tok_count + s_tok > self.window_size:
                yield " ".join(buf)
                
                # My sliding window approach - maintain overlap for context
                while buf and tok_count > self.stride:
                    tok_removed = self.n_tokens(buf[0])
                    tok_count -= tok_removed
                    buf.pop(0)
            
            buf.append(sent)
            tok_count += s_tok
        
        # Don't forget the last chunk
        if buf:
            yield " ".join(buf)
    
    def normalize_section(self, rec: Dict, url: str) -> str:
        """My section normalization logic - fallback to URL path if no section."""
        return (
            rec.get("section_id")
            or rec.get("type") 
            or rec.get("section")
            or re.sub(r"https?://[^/]+/([^/#?]+).*", r"\\1", url)
        )
    
    def process_jsonl_files(self, file_paths: Dict[str, Path]) -> pd.DataFrame:
        """
        Process my scraped JSONL files and return chunked DataFrame.
        This handles Mayo, NHS, WebMD data from my scrapers.
        
        Args:
            file_paths: Dictionary mapping source names to file paths
            
        Returns:
            DataFrame with processed chunks
        """
        rows = []
        
        for source, fp in file_paths.items():
            with open(fp, "r", encoding="utf-8") as f:
                try:
                    # Get line count for progress bar
                    total_lines = int(subprocess.check_output(["wc", "-l", fp]).split()[0])
                except:
                    total_lines = None
                
                for line in tqdm(f, total=total_lines, desc=f"Processing {source}", 
                               unit="record", colour="green"):
                    try:
                        rec = json.loads(line)
                        url = rec["url"]
                        title = rec.get("title", "")
                        
                        # Extract all text fields (ignore metadata)
                        text_keys = [k for k, v in rec.items() 
                                   if k not in self.meta_keys and isinstance(v, str)]
                        texts = [rec[k] for k in text_keys 
                               if k in rec and rec[k].strip()]
                        
                        if not texts:  # Skip empty records
                            continue
                            
                        page_text = "\\n".join(texts)
                        
                        # Apply my sentence-aware chunking
                        for chunk_text in self.sentence_chunks(page_text):
                            rows.append({
                                "id": str(uuid.uuid4()),
                                "url": url,
                                "title": title,
                                "section": self.normalize_section(rec, url),
                                "source": source,
                                "text": chunk_text,
                                "retrieved_date": self.today,
                                "n_tokens": self.n_tokens(chunk_text),
                            })
                    except Exception as e:
                        print(f"Error processing line in {source}: {e}")
                        continue
        
        return pd.DataFrame(rows)
    
    def save_chunks(self, df: pd.DataFrame, output_path: Path) -> None:
        """Save my processed chunks to parquet - much faster than CSV."""
        print("Chunks per corpus:")
        print(df.groupby("source").size())
        
        df.to_parquet(output_path, index=False)
        print(f"\\n✅ Saved {len(df):,} chunks → {output_path}")


class ChunkProcessor:
    """My specialized processor for post-processing chunks."""
    
    def __init__(self):
        self.enc = tiktoken.get_encoding("cl100k_base")
    
    def validate_chunks(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        My chunk validation and cleaning pipeline.
        
        Args:
            df: DataFrame with chunks
            
        Returns:
            Tuple of (cleaned_df, validation_stats)
        """
        stats = {
            "original_count": len(df),
            "empty_text": 0,
            "too_long": 0,
            "duplicates": 0
        }
        
        # Remove empty text chunks
        mask_empty = df["text"].str.strip() == ""
        stats["empty_text"] = mask_empty.sum()
        df = df[~mask_empty]
        
        # Remove chunks that are too long (edge case in my chunking)
        mask_long = df["n_tokens"] > 500
        stats["too_long"] = mask_long.sum()
        df = df[~mask_long]
        
        # Remove exact duplicates (can happen with similar pages)
        before_dedup = len(df)
        df = df.drop_duplicates(subset=["text"], keep="first")
        stats["duplicates"] = before_dedup - len(df)
        
        stats["final_count"] = len(df)
        
        return df, stats
    
    def add_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add useful metadata fields I need for analysis."""
        df = df.copy()
        
        # Basic text statistics
        df["char_count"] = df["text"].str.len()
        df["word_count"] = df["text"].str.split().str.len()
        
        # Content type indicators for my analysis
        df["has_numbers"] = df["text"].str.contains(r"\\d+", regex=True)
        df["has_medical_terms"] = df["text"].str.contains(
            r"\\b(disease|symptom|treatment|diagnosis|medication|therapy)\\b", 
            case=False, regex=True
        )
        
        return df


class SynonymProcessor:
    """My CHV (Consumer Health Vocabulary) synonym injection system."""
    
    def __init__(self, chv_path: Optional[Path] = None):
        """
        Initialize my synonym processor.
        
        Args:
            chv_path: Path to my CHV concepts_terms_flat_file
        """
        self.chv_path = chv_path
        self.clin_to_lay = {}  # My clinical-to-lay term mapping
        
        if chv_path and chv_path.exists():
            self._load_chv_mappings()
    
    def _load_chv_mappings(self):
        """Load my CHV clinical-to-lay term mappings."""
        try:
            df = pd.read_csv(
                self.chv_path,
                sep="\\t",
                header=None,
                names=["cui", "term", "desc", "is_consumer", "is_umls", 
                       "is_disparaged", "score", "freq"] + [f"extra_{i}" for i in range(8)]
            )
            
            # Build consumer terms map
            consumer = df[df.is_consumer == 1].groupby("cui")["term"] \
                        .apply(lambda ts: ts.str.lower().unique().tolist())
            
            # Build my clinical-to-lay mapping
            for cui, grp in df.groupby("cui"):
                clin_terms = grp[grp.is_umls == 1]["term"].str.lower().unique()
                cons_terms = consumer.get(cui, [])
                for ct in clin_terms:
                    if cons_terms:
                        self.clin_to_lay[ct] = cons_terms
                        
            print(f"Loaded {len(self.clin_to_lay)} clinical-to-lay mappings")
            
        except Exception as e:
            print(f"Error loading CHV mappings: {e}")
    
    def inject_synonyms(self, text: str, max_synonyms: int = 2) -> str:
        """
        Inject lay synonyms for clinical terms - makes content more accessible.
        
        Args:
            text: Input text
            max_synonyms: Max synonyms I inject per term
            
        Returns:
            Text with synonyms injected
        """
        if not self.clin_to_lay:
            return text
        
        # Sort by length (longest first) for better matching
        sorted_terms = sorted(self.clin_to_lay.keys(), key=len, reverse=True)
        
        for term in sorted_terms:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            synonyms = self.clin_to_lay[term][:max_synonyms]
            
            if synonyms:
                replacement = f"{term} ({', '.join(synonyms)})"
                text = pattern.sub(replacement, text, count=1)
        
        return text
