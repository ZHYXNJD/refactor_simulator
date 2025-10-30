FROM jingyunliu663/manhattan_mcts
LABEL authors="11249"

# STEP 2: 複製修改後的環境檔到映像檔中
COPY simulator.yaml /tmp/simulator.yaml

RUN conda env create -n my_torch_simulator -f /tmp/simulator.yaml && \
    conda clean -a

SHELL ["conda", "run", "-n", "my_torch_simulator", "/bin/bash", "-c"]

# STEP 5: 讓容器保持運行
CMD ["tail", "-f", "/dev/null"]





