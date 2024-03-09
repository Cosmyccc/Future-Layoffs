import os
import uuid
import subprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import NotebookLoader
from langchain_community.llms import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import clean_and_tokenize

def github_clone_repo(github_url, repo_path):
    try:
        subprocess.run(f"git clone {github_url} {repo_path}", check=True)
        return True
    except subprocess.CalledProcessError as error:
        print("Failed to clone repository:", error)
        return False
    
def load_and_index_files(repo_path):
    file_extentions=['txt', 'md', 'markdown', 'rst', 'py', 'js', 'java', 'c', 'cpp', 'cs', 'go', 'rb', 'php', 'scala', 'html', 'htm', 'xml', 'json', 'yaml', 'yml', 'ini', 'toml', 'cfg', 'conf', 'sh', 'bash', 'css', 'scss', 'sql', 'gitignore', 'dockerignore', 'editorconfig', 'ipynb', 'ts', 'gitattributes', 'pdf', 'doc', 'docs', 'csv', 'tsx', 'jsx', 'cjs', 'mjs', 'prisma', 'kt', 'swift', 'go', 'r', 'rust', 'dart']
    
    file_type_counts={}
    documents_dict={}
    
    for ext in file_extentions:
        glob_pattern = f'**/*.{ext}'
        try:
            loader = None
            if ext=='ipynb':
                loader = NotebookLoader(str(repo_path), include_outputs=True, max_output_length=20, remove_newline=True)
            else:
                loader = DirectoryLoader(str(repo_path), glob_pattern=glob_pattern)
                
            loaded_documents = loader.load() if callable(loader.load) else []
            if loaded_documents:
                file_type_counts[ext] = len(loaded_documents)
                for doc in loaded_documents:
                    file_path= doc.metadata['source']
                    relative_path=os.path.relpath(file_path, repo_path)
                    file_id= str(uuid.uuid4())
                    doc.metadata['source']=relative_path
                    doc.metadata['file_id']=file_id
                    
                    documents_dict[file_id]=doc
        
        except Exception as error:
            print('Print loading file with pattern:, {glob_pattern}: {error}')
            continue
    
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    
    split_docs=[]
    for file_id, original_doc in documents_dict.items():
        text_splitter.split_documents([original_doc])
        for split_doc in split_docs:
            split_doc.metadata['file_id'] = original_doc.metadata['file_id']
            split_doc.metadata['source'] = original_doc.metadata['source']
            
        split_doc.extend(split_docs)
        
    index=None
    if split_docs:
        tokenzied_docs=[clean_and_tokenize(doc.page_content) for doc in split_docs]
        index=BM25Okapi(tokenzied_docs)
    
    return index, split_docs, file_type_counts, [doc.metadata['source'] for doc in split_docs]

def search_docs(query, index, documents, n_results=5):
    query_tokens=clean_and_tokenize(query)
    bm25_scores=index.get_scores(query_tokens)
    
    tfidf_vectorizer=TfidfVectorizer(tokenizer=clean_and_tokenize, lowercase=True, stop_words='english', use_idf=True, smooth_idf=True, sublinear_tf=True)
    tfidf_matrix=tfidf_vectorizer.fit_transform([doc.page_content for doc in documents])
    query_tfidf=tfidf_vectorizer.transform([query])
    
    cosine_sim_score=cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    
    combined_scores = bm25_scores * 0.5 + cosine_sim_score * 0.5
    
    unique_top_document_indices = list(set(combined_scores.argsort()[::-1]))[:n_results]

    return [documents[i] for i in unique_top_document_indices]
    

                