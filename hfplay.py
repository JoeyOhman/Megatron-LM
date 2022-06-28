import argparse

from transformers import AutoModelForMaskedLM, AutoConfig, AutoTokenizer, pipeline


def get_model_tokenizer(model_name_or_path):
    # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_auth_token=True)
    tokenizer = AutoTokenizer.from_pretrained("pretrained_model_hf_large_165K", use_auth_token=True)
    # tokenized = tokenizer("Hej där (123) [MASK]")
    # decoded = tokenizer.decode(tokenized['input_ids'])
    # print(decoded)
    # exit()
    # config = AutoConfig.from_pretrained(model_name_or_path)
    # print(config)

    # model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, config=config)
    model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, use_auth_token=True)

    return tokenizer, model


# huggingface-cli login
# from transformers import AutoModelForMaskedLM, AutoTokenizer
# from huggingface_hub import create_repo
# create_repo("megatron-bert-base-swedish-cased-600k", organization="KBLab", private=True)
def push_to_hub():
    # model = AutoModel.from_pretrained("pretrained_model_hf")
    # model = AutoModelForPretraining.from_pretrained("pretrained_model_hf")
    model = AutoModelForMaskedLM.from_pretrained("pretrained_model_hf_600K")
    tok = AutoTokenizer.from_pretrained("pretrained_model_hf_600K")
    # model.save_pretrained("new_bert", push_to_hub=True, repo_name="megatron-bert-base-swedish-cased-new", organization="KBLab")
    # model.push_to_hub("megatron-bert-base-swedish-cased-new", organization="KBLab")
    model.push_to_hub("megatron-bert-large-swedish-cased-165k", organization="KBLab", repo_url="https://huggingface.co/KBLab/megatron-bert-large-swedish-cased-165k")
    # model.push_to_hub("megatron-bert-base-swedish-cased-600k", organization="KBLab", repo_url="https://huggingface.co/KBLab/megatron-bert-base-swedish-cased-600k")
    model.push_to_hub("megatron-bert-base-swedish-cased-new", organization="KBLab", repo_url="https://huggingface.co/KBLab/megatron-bert-base-swedish-cased-new")
    model.push_to_hub("bert-base-swedish-lowermix-reallysimple-ner", organization="KBLab", repo_url="https://huggingface.co/KBLab/bert-base-swedish-lowermix-reallysimple-ner")
    model.push_to_hub("bert-base-swedish-cased-reallysimple-ner", organization="KBLab", repo_url="https://huggingface.co/KBLab/bert-base-swedish-cased-reallysimple-ner")
    tok.push_to_hub("bert-base-swedish-cased-reallysimple-ner", organization="KBLab")
    tok.push_to_hub("megatron-bert-base-swedish-cased-new", organization="KBLab")
    tok.push_to_hub("megatron-bert-base-swedish-cased-600k", organization="KBLab", repo_url="https://huggingface.co/KBLab/megatron-bert-base-swedish-cased-600k")


def main(model_name_or_path):
    tokenizer, model = get_model_tokenizer(model_name_or_path)
    print(f"#params: {(sum(p.numel() for p in model.parameters() if p.requires_grad)) // 1000000}M")
    model.eval()

    mlm = pipeline("fill-mask", model=model, tokenizer=tokenizer)

    text = "Huvudstaden i Frankrike är [MASK]."
    # text = "Det största djuret i Sverige är [MASK]."
    # text = "Invandrarna är [MASK]."
    res = mlm(text, top_k=10)
    top_k_tuples = [(r['score'], r['token_str']) for r in res]
    top_k = sorted(top_k_tuples, key=lambda x: x[0], reverse=True)

    print(f"{'Token':<15} Probability")
    for (p, tok) in top_k:
        print(f"{tok:<15} {round(p, 3)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name_or_path", type=str, help="Path to model+tokenizer directory.")
    args = parser.parse_args()
    main(args.model_name_or_path)
