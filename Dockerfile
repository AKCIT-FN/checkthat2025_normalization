FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

RUN apt update -y
RUN apt install -y nano python3-setuptools locales
RUN locale-gen pt_BR.UTF-8
ENV LANG pt_BR.UTF-8
ENV LC_ALL pt_BR.UTF-8
RUN python3 -m pip install -U setuptools
RUN python3 -m pip install loguru
RUN python3 -m pip install transformers[accelerate,hf_xet,sentencepiece] datasets evaluate sentence-transformers huggingface-hub[hf_xet]
RUN python3 -m pip install nltk rouge_score bert_score
RUN python3 -m nltk.downloader punkt
RUN python3 -m nltk.downloader wordnet
RUN python3 -m nltk.downloader punkt_tab
RUN python3 -m nltk.downloader omw-1.4
