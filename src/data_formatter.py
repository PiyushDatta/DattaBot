from typing import Any, Dict, Union, Optional
from src.tokenizer import get_tokenizer
from src.util import DatasetType


class DattaBotDataFormatter:
    """
    Formats raw dataset items into text strings suitable for tokenization.
    Each dataset type has its own formatting logic.
    """

    def __init__(self, dataset_type: DatasetType):
        self.dataset_type = dataset_type
        self._formatter_map = {
            DatasetType.FINANCEQA: self._format_financeqa,
            DatasetType.MMLU_REDUX: self._format_mmlu_redux,
            DatasetType.AG_NEWS: self._format_ag_news,
            DatasetType.WIKITEXT: self._format_wikitext,
            DatasetType.OPENWEBTEXT: self._format_openwebtext,
        }

    def format(
        self, item: Union[Dict[str, Any], str], max_seq_len: Optional[int] = None
    ) -> str:
        formatter = self._formatter_map.get(self.dataset_type)
        if formatter:
            return formatter(item, max_seq_len=max_seq_len)
        return self._format_generic(item)

    # ------------------------
    # Helper for truncation
    # ------------------------
    @staticmethod
    def _truncate_text(text: str, max_seq_len: Optional[int]) -> str:
        """
        Truncate text to max_seq_len tokens using the tokenizer, accounting for BOS/EOS tokens.
        Adds "..." if truncation occurs.

        The text itself is not prepended/appended with BOS/EOS here; those are added later
        when encoding to be ingested (by a model) is called.
        """
        if max_seq_len:
            tokenizer = get_tokenizer()
            # encode without BOS/EOS
            tokens = tokenizer.encode(
                text, with_special_tokens=False, disallowed_special=()
            )
            # account for BOS/EOS tokens, reserve space for BOS and EOS
            effective_max_len = max_seq_len - 2
            if len(tokens) > effective_max_len:
                truncated_tokens = tokens[:effective_max_len]
                text = tokenizer.decode(truncated_tokens)
        return text

    # ------------------------
    # Dataset-specific formats
    # ------------------------
    def _format_financeqa(
        self, item: Dict[str, Any], max_seq_len: Optional[int] = None
    ) -> str:
        question = item.get("question", "").strip()
        answer = str(item.get("answer", "") or item.get("final_result", "")).strip()
        text_parts = [f"Question: {question}", f"Answer: {answer}"]

        context_blocks = []

        # Table
        table = item.get("table", [])
        if table:
            table_lines = [" | ".join(str(cell) for cell in row) for row in table]
            context_blocks.append("Table:\n" + "\n".join(table_lines))

        # Pre-text
        pre_text = item.get("pre_text", [])
        if pre_text:
            if isinstance(pre_text, list):
                pre_text = " ".join(pre_text)
            context_blocks.append(pre_text.strip())

        # Post-text
        post_text = item.get("post_text", [])
        if post_text:
            if isinstance(post_text, list):
                post_text = " ".join(post_text)
            context_blocks.append(post_text.strip())

        context = "\n\n".join(context_blocks)
        context = self._truncate_text(context, max_seq_len)

        return f"{text_parts[0]}\n\n{text_parts[1]}\n\nContext:\n{context}"

    def _format_mmlu_redux(
        self, item: Dict[str, Any], max_seq_len: Optional[int] = None
    ) -> str:
        question = item.get("question", "")
        choices = item.get("choices", [])
        answer = item.get("answer", "")

        text = (
            f"Question: {question}\n\n"
            + "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
            + f"\n\nAnswer: {answer}"
        )

        return self._truncate_text(text, max_seq_len)

    def _format_ag_news(
        self, item: Dict[str, Any], max_seq_len: Optional[int] = None
    ) -> str:
        text = ""
        if isinstance(item, dict):
            text = item.get("text", "")
            label = item.get("label", "")
            if label:
                text = f"{text}\nCategory: {label}"
        else:
            text = str(item)

        return self._truncate_text(text, max_seq_len)

    def _format_wikitext(
        self, item: Dict[str, Any], max_seq_len: Optional[int] = None
    ) -> str:
        text = item.get("text", str(item)) if isinstance(item, dict) else str(item)
        return self._truncate_text(text, max_seq_len)

    def _format_openwebtext(
        self, item: Dict[str, Any], max_seq_len: Optional[int] = None
    ) -> str:
        text = item.get("text", str(item)) if isinstance(item, dict) else str(item)
        return self._truncate_text(text, max_seq_len)

    def _format_generic(self, item: Union[Dict[str, Any], str]) -> str:
        if isinstance(item, dict):
            for field in ["text", "content", "document", "sentence"]:
                if field in item:
                    return str(item[field])
            return str(item)
        return str(item)

    def __repr__(self):
        return f"DattaBotDataFormatter(dataset_type={self.dataset_type.value})"
