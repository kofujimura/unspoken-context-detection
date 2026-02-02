"""
Keyword extraction for reason labels.
Based on SPEC.md Section 7.1.
"""
from typing import List, Set, Dict
from collections import Counter
import re


class KeywordExtractor:
    """Keyword extractor with morphological analysis support."""

    def __init__(self, language: str = "ja"):
        """
        Initialize extractor.

        Args:
            language: Language code ("ja" or other)
        """
        self.language = language
        self.tokenizer = None

        if language == "ja":
            try:
                from sudachipy import tokenizer as sudachi_tokenizer
                from sudachipy import dictionary

                self.tokenizer = dictionary.Dictionary().create()
                self.mode = sudachi_tokenizer.Tokenizer.SplitMode.C
            except ImportError:
                print("Warning: sudachipy not available, using fallback tokenizer")
                self.tokenizer = None

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract keywords from text.

        Args:
            text: Input text
            top_n: Number of top keywords to return

        Returns:
            List of keywords
        """
        if self.language == "ja" and self.tokenizer is not None:
            return self._extract_japanese_keywords(text, top_n)
        else:
            return self._extract_fallback_keywords(text, top_n)

    def _extract_japanese_keywords(self, text: str, top_n: int) -> List[str]:
        """
        Extract Japanese keywords using Sudachi.

        Args:
            text: Japanese text
            top_n: Number of keywords

        Returns:
            List of keywords (nouns and compound nouns)
        """
        tokens = self.tokenizer.tokenize(text, self.mode)
        keywords = []

        for token in tokens:
            pos = token.part_of_speech()[0]

            # Extract nouns
            if pos == "名詞":
                surface = token.surface()
                # Filter out single character nouns and common particles
                if len(surface) > 1 and not self._is_stopword(surface):
                    keywords.append(surface)

        # Count and return top N
        counter = Counter(keywords)
        return [word for word, _ in counter.most_common(top_n)]

    def _extract_fallback_keywords(self, text: str, top_n: int) -> List[str]:
        """
        Fallback keyword extraction (simple tokenization).

        Args:
            text: Input text
            top_n: Number of keywords

        Returns:
            List of keywords
        """
        # Simple tokenization by whitespace and punctuation
        words = re.findall(r'\w+', text.lower())

        # Filter short words and stopwords
        keywords = [
            w for w in words
            if len(w) > 2 and not self._is_stopword(w)
        ]

        counter = Counter(keywords)
        return [word for word, _ in counter.most_common(top_n)]

    def _is_stopword(self, word: str) -> bool:
        """
        Check if word is a stopword.

        Args:
            word: Word to check

        Returns:
            True if stopword
        """
        if self.language == "ja":
            ja_stopwords = {
                "これ", "それ", "あれ", "この", "その", "あの",
                "ここ", "そこ", "あそこ", "こちら", "そちら", "あちら",
                "です", "ます", "てる", "いる", "ある", "なる",
                "する", "できる", "なり", "こと", "もの", "よう",
                "ため", "とき", "場合", "ところ", "わけ"
            }
            return word in ja_stopwords
        else:
            en_stopwords = {
                "the", "is", "at", "which", "on", "and", "or",
                "to", "in", "for", "of", "with", "by", "from",
                "this", "that", "these", "those", "are", "was",
                "were", "been", "have", "has", "had"
            }
            return word in en_stopwords

    def compute_keyword_diff(
        self,
        text_minus: str,
        text_plus: str,
        top_n: int = 5
    ) -> List[str]:
        """
        Compute keyword difference: keywords in plus but not in minus.

        Args:
            text_minus: Past window text
            text_plus: Current window text
            top_n: Number of diff keywords to return

        Returns:
            List of new keywords
        """
        keywords_minus = set(self.extract_keywords(text_minus, top_n=50))
        keywords_plus = set(self.extract_keywords(text_plus, top_n=50))

        # Keywords in plus but not in minus
        diff = keywords_plus - keywords_minus

        # Count frequency in plus text and sort
        if self.language == "ja" and self.tokenizer is not None:
            tokens = self.tokenizer.tokenize(text_plus, self.mode)
            plus_words = [t.surface() for t in tokens]
        else:
            plus_words = re.findall(r'\w+', text_plus.lower())

        diff_counter = Counter([w for w in plus_words if w in diff])

        return [word for word, _ in diff_counter.most_common(top_n)]
