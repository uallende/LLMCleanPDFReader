import argparse
import torch
import pdfplumber
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def main(doc_path, chunk_size, overlap):

    device = torch.cuda.current_device()
    name = "mistralai/Mistral-7B-Instruct-v0.1"
    model = AutoModelForCausalLM.from_pretrained(name, load_in_4bit=True, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(name)

    llm_grammar_loop(doc_path, device, model, tokenizer, chunk_size=100, overlap=0)

def create_chunks_of_text(document, chunk_size, overlap=0):
    
    word_split = document.split() # Split document into words
    chunk = ""
    overlap = 0
    print(f'Number of words: {len(word_split)}. Chunk size: {chunk_size}. Overlap: {overlap}')
    n_chunks = ((len(word_split) - chunk_size) // (chunk_size - overlap)) + 1
    lstd_text = []

    """Create chunks of text from a document."""
    for i in range(n_chunks):
        start = i * (chunk_size - overlap)
        end = start + chunk_size if i < n_chunks - 1 else len(word_split)
        chunk = " ".join(word_split[start:end])
        lstd_text.append(chunk)
        chunk = ""

    print(f'Number of chunks: {len(lstd_text)}')
    return lstd_text

def read_pdf(doc_path):
        
        doc = ""
        with pdfplumber.open(doc_path) as pdf:
    
            for page in pdf.pages:
                text = page.extract_text()
                doc += text

        doc_name = doc_path[doc_path.rfind("/") + 1:]
        print(f'Processing document {doc_name}. Number of pages: {len(pdf.pages)}')
        return doc, doc_name

def llm_grammar_loop(doc_path, device, model=None, tokenizer=None, chunk_size=100, overlap=0):

    document, doc_name = read_pdf(doc_path)
    lstd_text = create_chunks_of_text(document, chunk_size, overlap)
    fxd_text = []
    text_cue = "Corrected text:"
    tplt_prompt = f"""I need you to fix the grammatical errors and properly separate words that are stitched together within a text. Only reply with "Corrected text:" """

    for item in tqdm.tqdm(lstd_text, desc="Processing text"):
        
        prompt = f"""{tplt_prompt}. Original text: "{item}". Corrected text:"""
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=chunk_size*4)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        first_time_idx = output_text.find(text_cue)
        sec_time_idx = output_text.find(text_cue, first_time_idx + len(text_cue))
        
        corrected_text = output_text[sec_time_idx + len(text_cue):].strip()
        print("Corrected Text:", corrected_text)  # Debug print
        fxd_text.append(corrected_text)
        
    with open(f"reviewed_pdfs/{doc_name}.csv", "w") as f:
        for item in fxd_text:
            f.write("%s\n" % item)
        
    print(f"Document {doc_name} saved to folder reviewed_pdfs/")

if __name__ == "__main__":
    print('Running main function')
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument("--doc_path", type=str, default="default_path", help="The path of the document.")
    parser.add_argument("--chunk_size", type=int, default=150, help="Size of each text chunk.")
    parser.add_argument("--overlap", type=int, default=0, help="Overlap size between chunks.")

    args = parser.parse_args()
    main(args.doc_path, args.chunk_size, args.overlap)