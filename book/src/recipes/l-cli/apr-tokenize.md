# apr-tokenize

BPE tokenizer training pipeline. Trains a Byte Pair Encoding vocabulary from a text corpus, shows merge history, and demonstrates tokenization with roundtrip verification.

```bash
cargo run --example cli_apr_tokenize -- --demo
```

**CLI equivalent**: `apr tokenize corpus.txt --method bpe --vocab-size 100`
