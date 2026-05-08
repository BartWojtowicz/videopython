class Tokenizer:
    def encode(self, text: str) -> list[int]: ...
    def decode(self, token_ids: list[int]) -> str: ...

def get_tokenizer(
    multilingual: bool,
    *,
    num_languages: int = ...,
    language: str | None = ...,
    task: str | None = ...,
) -> Tokenizer: ...
