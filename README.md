Estimativa do SoC de Baterias de Íons de Lítio

Esta aplicação foi desenvolvida para estimar o Estado de Carga (SoC) de baterias de íons de lítio utilizando técnicas de aprendizado de máquina. O sistema integra processamento de dados, treinamento de modelos (1D CNN com PyTorch) e uma interface gráfica interativa construída com Tkinter.

---

Visão Geral

- Objetivo: Estimar o SoC de baterias de íons de lítio com precisão, utilizando dados processados e modelos treinados.
- Modularidade: Arquitetura modular que facilita manutenção, escalabilidade e inclusão de novos modelos.
- Interface Gráfica: Permite o carregamento de dados, treinamento de modelos e realização de estimativas de forma intuitiva.

---

Requisitos do Sistema

- Sistema Operacional: Windows, macOS ou Linux.
- Python: Versão 3.8 ou superior.
- GPU (Opcional): CUDA Toolkit 12.4 (para aceleração via GPU).

---

Dependências

Todas as dependências necessárias estão listadas no arquivo requirements.txt. Algumas das principais bibliotecas utilizadas incluem:

- Processamento e Manipulação de Dados: pandas, numpy
- Visualização de Dados: matplotlib
- Manipulação de Arquivos: h5py, openpyxl
- Aprendizado de Máquina: scikit-learn, torch
- Interface Gráfica: tkinter
- Processamento de Imagens: pillow
- Outras Dependências: concurrent.futures (biblioteca padrão do Python)

Para instalar as dependências, execute:

    pip install -r requirements.txt

---

Estrutura do Projeto

.
├── scripts
│   ├── data
│   │   ├── data_processing.py       # Leitura, limpeza e processamento dos dados brutos.
│   │   ├── data_preparation.py      # Preparação dos dados para treinamento.
│   │   └── data_visualization.py    # Criação de gráficos e análise visual dos dados.
│   │
│   ├── gui
│   │   ├── SoC_Tool.py              # Arquivo principal que integra e inicia a aplicação.
│   │   ├── process_tab.py           # Aba para importação e processamento dos dados.
│   │   ├── visualize_tab.py         # Aba para visualização de dados e resultados.
│   │   ├── train_tab.py             # Aba para configuração e treinamento de modelos.
│   │   ├── estimate_tab.py          # Aba para estimativa do SoC com modelos treinados.
│   │   ├── progress_handler.py      # Gestão de progresso e feedback visual.
│   │   ├── metrics_display.py       # Exibição de métricas de desempenho.
│   │   ├── custom_model_manager.py  # Gerenciamento de modelos personalizados.
│   │   └── training_parameters.py   # Validação e captura dos parâmetros de treinamento.
│   │
│   └── models
│       └── model_training.py        # Fluxo principal para treinamento, avaliação e salvamento dos modelos.
│
├── logs
│   ├── data_processing.log
│   ├── data_preparation.log
│   ├── model_training.log
│   ├── custom_model_manager.log
│   └── training_parameters.log
│
├── requirements.txt
└── README.txt

---

Instalação e Configuração

1. Clone o repositório:

    git clone https://github.com/seu-usuario/nome-do-projeto.git
    cd nome-do-projeto

2. Instale as dependências:

    pip install -r requirements.txt

3. Configuração Opcional (GPU):
   Para utilizar a GPU, certifique-se de instalar o CUDA Toolkit 12.4 compatível com sua placa.
   (Mais informações: https://developer.nvidia.com/cuda-12-4-0-download-archive)

---

Uso da Aplicação

Iniciando a Aplicação:
    Execute o arquivo principal para abrir a interface gráfica:

    python SoC_Tool.py

Carregamento de Dados:
- Acesse a aba "Carregar Dados".
- Selecione o arquivo HDF5 contendo os dados processados das baterias.
- Certifique-se de que o arquivo está no formato esperado.

Treinamento dos Modelos:
- Navegue até a aba "Treinar Modelos".
- Configure os parâmetros de treinamento (número de épocas, taxa de aprendizado, tamanho do batch, etc.).
- Selecione os conjuntos de dados de treinamento, validação e teste.
- Clique em "Iniciar Treinamento".
- Acompanhe o progresso e as métricas de desempenho (MSE, MAE, RMSE) exibidas na interface.

Estimativa do SoC:
- Na aba "Estimar SoC", selecione o modelo previamente treinado.
- Carregue os dados de teste e insira os parâmetros de entrada necessários (por exemplo, tensão e corrente).
- Clique em "Estimar" para visualizar os resultados e os gráficos comparativos entre os modelos.

---

Adicionando Novos Modelos

Para integrar novos modelos de rede neural:

1. Defina a arquitetura:
   Crie uma nova classe no arquivo model_training.py que herde de nn.Module e defina a arquitetura desejada.

2. Atualize a criação de modelos:
   Modifique a função create_model em model_training.py para incluir seu novo modelo e definir seus parâmetros.

3. Ajuste a Interface:
   Atualize os scripts da interface gráfica (como SoC_Tool.py ou custom_model_manager.py) para permitir a seleção do novo modelo.

---

Manutenção e Extensão

- Logs:
  Utilize os arquivos de log para monitorar o processamento dos dados e o treinamento dos modelos. Eles auxiliam na identificação de problemas e na depuração do sistema.

- Atualização de Dependências:
  Para atualizar as bibliotecas Python, execute:

    pip install --upgrade -r requirements.txt

- Modularidade:
  A estrutura modular do código facilita a manutenção e a implementação de novas funcionalidades conforme a necessidade.

---

Exemplos de Execução

Exemplo 1: Treinamento de um Modelo Predefinido

1. Inicie a aplicação com "python SoC_Tool.py".
2. Navegue até a aba "Carregar Dados" e selecione o arquivo "dataset.hdf5".
3. Acesse a aba "Treinar Modelos".
4. Escolha o "Modelo 1" na lista de modelos disponíveis.
5. Configure os parâmetros de treinamento e clique em "Iniciar Treinamento".
6. Monitore o progresso e visualize as métricas de desempenho após a conclusão.

Exemplo 2: Estimativa do SoC

1. Na aba "Estimar SoC", selecione o modelo treinado desejado.
2. Carregue os dados de teste.
3. Insira os parâmetros de entrada (como tensão e corrente).
4. Clique em "Estimar" para obter os resultados e visualize os gráficos comparativos.

---

Contribuição

Contribuições são bem-vindas! Se você deseja colaborar com o projeto, por favor, abra uma issue ou envie um pull request com suas sugestões e melhorias.

---

Licença

Este projeto está licenciado sob a MIT License. Consulte o arquivo LICENSE para mais detalhes.

---

Contato

Para dúvidas, sugestões ou suporte, entre em contato através de jhoanpvar@gmail.com.
