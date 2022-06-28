from transformers import PreTrainedTokenizerFast, AutoTokenizer

tokenizer_path = "/ceph/hpc/home/eujoeyo/group_space/robin/workspace/hface_transformer/oscar+wiki.64k.wordpiece.tokenizer.json"


def load_tokenizer(path, unk, mask, pad, bos, eos):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=path)
    tokenizer.bos_token = bos
    tokenizer.eos_token = eos
    tokenizer.cls_token = bos
    tokenizer.sep_token = eos
    tokenizer.mask_token = mask
    tokenizer.unk_token = unk
    tokenizer.pad_token = pad
    return tokenizer


def load_tokenizer_kb():
    tokenizer = AutoTokenizer.from_pretrained("KB/bert-base-swedish-cased")
    return tokenizer


"""
def load_tokenizer_megatron():
    args = get_args()
    if nltk_available and args.split_sentences:
        nltk.download("punkt", quiet=True)

    tokenizer = build_tokenizer(args)
"""


def write_vocab_to_file(tokenizer):
    voc = tokenizer.get_vocab()
    # od = collections.OrderedDict(sorted(voc.items()))
    od = dict(sorted(voc.items(), key=lambda item: item[1]))
    with open("joey/data/robin-vocab-new.txt", 'w') as f:
        for k, v in od.items():
            # print(f"k={k}, v={v}")
            f.write(k + "\n")


def main():
    tokenizer = load_tokenizer(tokenizer_path, "[UNK]", "[MASK]", "[PAD]", "[CLS]", "[SEP]")
    # tokenizer = load_tokenizer_kb()
    print("pad_token_id:", tokenizer.pad_token_id)
    print("unk_token_id:", tokenizer.unk_token_id)
    print("mask_token_id:", tokenizer.mask_token_id)
    print("cls_token_id:", tokenizer.cls_token_id)
    print("sep_token_id:", tokenizer.sep_token_id)

    print(repr(tokenizer.convert_ids_to_tokens(5)))
    # write_vocab_to_file(tokenizer)
    # tokenized = tokenizer("Hejsan!. [MASK] lolII< 3a(7) . [PAD]")
    tokenized = tokenizer("Hejsan!. 1233a(37) 23. 5 22 ris  !!\n")
    print(tokenized)
    decoded = tokenizer.decode(tokenized['input_ids'])
    print(decoded)

    tokenizer.save_pretrained("pretrained_model_hf/")

    # write_vocab_to_file(tokenizer)


if __name__ == '__main__':
    main()
