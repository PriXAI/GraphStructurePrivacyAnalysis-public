FROM continuumio/miniconda3:24.5.0-0

WORKDIR /artifact

COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

ENV CONDA_DEFAULT_ENV=graph-structure-privacy-analysis
ENV PATH=/opt/conda/envs/graph-structure-privacy-analysis/bin:$PATH

COPY . .

CMD ["/bin/bash"]
