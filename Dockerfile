FROM continuumio/miniconda3:24.5.0-0

WORKDIR /artifact

COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

COPY . .

CMD ["conda", "run", "--no-capture-output", "-n", "graph-structure-privacy-analysis", "python", "runner-one-dataset.py", "cora"]
