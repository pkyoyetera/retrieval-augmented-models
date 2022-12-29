
import faiss 
import torch 

from datasets import load_dataset
from datasets import Features, Sequence, Value

from functools import partial 

from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast

from typing import List, Optional, Dict


def split_text(text: str, n=100, character=" ") -> List[str]:
    """Split the text every ``n``-th occurrence of ``character``"""
    text = text.split(character)
    return [character.join(text[i : i + n]).strip() for i in range(0, len(text), n)]



def split_documents(documents: dict) -> dict:
    """Split documents into passages"""
    titles, texts = [], []
    for title, text in zip(documents["title"], documents["text"]):
        if text is not None:
            for passage in split_text(text):
                titles.append(title if title is not None else "")
                texts.append(passage)
    return {"title": titles, "text": texts}


def embed(documents: Dict, encoder: DPRContextEncoder, tokenizer: DPRContextEncoderTokenizerFast) -> Dict:
    
    input_ids = tokenizer(
        documents["title"], documents["text"], truncation=True, padding="longest", return_tensors="pt"
    )["input_ids"]
    
    embeddings = encoder(input_ids.to(device=device), return_dict=True).pooler_output
    
    return {"embeddings": embeddings.detach().cpu().numpy()}


if __name__ == '__main__':
    print(f"Device name: {torch.cuda.get_device_name(0)}\nProperties: {torch.cuda.get_device_properties(0)}")

    device = torch.device('cuda:0')

    ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

    ctx_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
    ctx_encoder.to(device)

    wiki_full = load_dataset('wikipedia', '20200501.en', beam_runner='DirectRunner')

    wiki_full = wiki_full.map(split_documents, batched=True)

    new_features = Features({
        "text": Value("string"),
        "title": Value("string"),
        "embeddings": Sequence(Value("float32"))
    })

    print("Beginning embedding...")

    wiki_full = wiki_full.map(
        partial(embed, encoder=ctx_encoder, tokenizer=ctx_tokenizer),
        batched=True,
        batch_size=12,
        features=new_features,
    )

    wiki_full.save_to_disk('data/wiki-20200501/')

    print("Done embedding, starting to index...")

    index = faiss.IndexHNSWFlat(index_hnsw_args.d, index_hnsw_args.m, faiss.METRIC_INNER_PRODUCT)

    wiki_full.add_faiss_index("embeddings", custom_index=index)

    print("Done indexing, saving...")

    wiki_full.get_index("embeddings").save("data/index-full")

    print("All done!")

