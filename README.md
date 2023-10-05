# LLMCleanPDFReader

## Overview
LLMCleanPDFReader is a project aimed at cleaning up and correcting the text extracted from PDF documents. 
It utilizes a language model to correct grammatical errors and separate words that might have been stitched together during the PDF parsing process. 
This is a learning project and is designed to work efficiently on a small GPU with 8GB VRAM.

## Features
- PDF text extraction
- Text correction using language models
- Text chunking for efficient processing
- Command line interface for easy parameter adjustment

## Installation
Clone this repository to your local machine and navigate to the project directory.

\\\bash
git clone https://github.com/uallende/LLMCleanPDFReader.git
cd LLMCleanPDFReader
\\\

Install the required Python packages.

\\\bash
pip install -r requirements.txt
\\\

## Usage
Run the main script and pass in the required arguments.

\\\bash
python main_script.py --doc_path=path/to/document --chunk_size=150 --overlap=0
\\\

## Contributing
Feel free to fork the project, open a pull request, or report any issues you encounter.


