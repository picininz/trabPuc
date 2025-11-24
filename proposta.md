# Plano de Experimento – Scoping e Planejamento

**Disciplina:** Medição e Experimentação em Engenharia de Software
**Trabalho Final:** Planejamento de Experimento para Projeto de Pesquisa

---

## 1. Identificação Básica

### 1.1 Título do Experimento

**Avaliação Comparativa da Efetividade de Modelos de Machine Learning para Detecção de Anomalias em Imagens de Pastagens**

### 1.2 ID / Código

**EXP-PAST-ML-2025-001**

### 1.3 Versão do Documento e Histórico de Revisão

* **Versão atual:** v1.0
* **Histórico:**

  * *v1.0 (23/11/2025)* – Criação inicial do plano de experimento para a disciplina de Medição e Experimentação em Engenharia de Software.

### 1.4 Datas (Criação e Última Atualização)

* **Data de criação:** 23/11/2025
* **Última atualização:** 23/11/2025

### 1.5 Autores (Nome, Área, Contato)

**Felipe Freitas Campos Picinin – Estudante de Engenharia de Software**
Contato: *[picinin.felipe2@gmail.com](mailto:picinin.felipe2@gmail.com)*

### 1.6 Responsável Principal (PI / Dono do Experimento)

**Felipe Freitas Campos Picinin**
Responsável pelas decisões metodológicas, execução do experimento, análise dos resultados e documentação da pesquisa.

### 1.7 Projeto / Produto / Iniciativa Relacionada

Este experimento está associado ao Trabalho de Conclusão de Curso (TCC) em Engenharia de Software, configurando-se como um projeto de **pesquisa aplicada** que investiga a efetividade comparativa de modelos de Machine Learning para **detecção automática de anomalias em pastagens**.

O estudo enquadra-se em:

* Avaliação empírica de técnicas de Machine Learning
* Experimentação controlada e medição de desempenho
* Análise comparativa baseada em evidências
* Validação científica de soluções tecnológicas

**Tecnologias envolvidas:** Python, PyTorch/TensorFlow, OpenCV, Scikit-learn, análise estatística.

---

## 2. Contexto e Problema

### 2.1 Descrição do Problema / Oportunidade

A pecuária extensiva no Brasil ocupa aproximadamente **159 milhões de hectares** e constitui uma das bases da economia nacional. A qualidade das pastagens impacta diretamente a produtividade, sustentabilidade ambiental e custos operacionais. Contudo, o monitoramento tradicional baseado em inspeção visual manual apresenta diversas limitações:

#### Problemas identificados:

* **Detecção tardia:** anomalias (degradação, pragas, solo exposto) são percebidas apenas quando o dano já é significativo.
* **Alto custo operacional:** grandes áreas exigem deslocamento constante de equipes técnicas.
* **Subjetividade:** avaliações variam entre observadores.
* **Cobertura limitada:** difícil monitoramento contínuo de propriedades extensas.

#### Oportunidade:

Com o avanço de drones e técnicas de visão computacional e Machine Learning, torna-se possível automatizar a identificação de anomalias em pastagens a partir de imagens aéreas. No entanto, ainda existe uma lacuna importante:

> **Não há clareza científica sobre quais modelos de ML são mais eficazes para detecção de anomalias em pastagens**, considerando suas características únicas (heterogeneidade, variabilidade sazonal, iluminação irregular).

---

### 2.2 Contexto Organizacional e Técnico

#### Contexto:

* **Tipo:** Pesquisa acadêmica em Engenharia de Software
* **Domínio:** Agricultura de precisão e pecuária
* **Equipe:** Pesquisador individual com orientação acadêmica
* **Processo:** Metodologia científica experimental baseada em dados quantitativos

#### Ambiente Técnico:

* **Linguagem:** Python 3.x
* **Frameworks de ML:** PyTorch ou TensorFlow
* **Processamento de imagens:** OpenCV
* **Análise estatística:** Scikit-learn, SciPy, pandas
* **Infraestrutura:** GPU local ou Google Colab
* **Versionamento:** Git/GitHub

#### Dados:

* Dataset de imagens aéreas de pastagens capturadas
* Resolução suficiente para distinguir padrões de anomalia
* Dados rotulados (degradação, solo exposto, pragas, plantas invasoras)

---

### 2.3 Trabalhos e Evidências Prévias (Internos e Externos)

#### Evidências externas – literatura:

**Machine Learning na agricultura:**

* CNNs amplamente utilizadas para doenças de plantas
* Modelos como ResNet, VGG, YOLO e U-Net com alto desempenho
* Transfer learning reduz necessidade de grandes datasets

**Sensoriamento remoto com drones:**

* Drones usados para monitoramento de safras
* Uso de NDVI para detecção de estresse
* Imagens multiespectrais aumentam precisão, mas também custos

**Comparações entre modelos:**

* Trade-offs entre acurácia e eficiência computacional
* Nenhum modelo universalmente superior; depende do contexto
* Meta-estudos reforçam a necessidade de experimentação específica

#### Evidências internas:

* Projeto ainda sem experimentos prévios
* Experiência do pesquisador com Python e fundamentos de ML
* Acesso à orientação acadêmica especializada

#### Lacunas identificadas:

* Poucos estudos comparativos focados em **pastagens brasileiras**
* Falta de datasets públicos rotulados com anomalias de pastagem
* Necessidade de avaliação empírica rigorosa em ambiente controlado

---

### 2.4 Referencial Teórico e Empírico Essencial

#### Conceitos fundamentais:

**Machine Learning e Deep Learning:**

* Aprendizado supervisionado
* Redes Neurais Convolucionais (CNNs)
* Transfer learning

**Detecção de anomalias:**

* Classificação binária e multi-classe
* Segmentação semântica
* Reconstrução por autoencoders

**Avaliação de modelos:**

* Acurácia, Precisão, Recall, F1-score
* Validação cruzada
* Trade-offs entre desempenho e custo computacional

#### Modelos candidatos:

* **ResNet** – eficaz para classificação profunda
* **MobileNet** – leve e eficiente
* **EfficientNet** – arquitetura otimizada
* **U-Net** – segmentação precisa
* **YOLO** – detecção em tempo real

#### Hipóteses teóricas (da literatura):

* Modelos mais profundos tendem a maior acurácia
* Transfer learning reduz tempo de treinamento
* Modelos leves têm melhor relação custo-benefício

#### Referências empíricas:

* Benchmarks mostram desempenho consistente de CNNs
* F1-scores em agricultura variam entre 0.75 e 0.95
* 1000–5000 imagens rotuladas costumam ser suficientes para transfer learning

# Plano de Experimento – Scoping e Planejamento

**Disciplina:** Medição e Experimentação em Engenharia de Software
**Trabalho Final:** Planejamento de Experimento para Projeto de Pesquisa

---

## 1. Identificação Básica

### 1.1 Título do Experimento

**Avaliação Comparativa da Efetividade de Modelos de Machine Learning para Detecção de Anomalias em Imagens de Pastagens**

### 1.2 ID / Código

**EXP-PAST-ML-2025-001**

### 1.3 Versão do Documento e Histórico de Revisão

* **Versão atual:** v1.0
* **Histórico:**

  * *v1.0 (23/11/2025)* – Criação inicial do plano de experimento para a disciplina de Medição e Experimentação em Engenharia de Software.

### 1.4 Datas (Criação e Última Atualização)

* **Data de criação:** 23/11/2025
* **Última atualização:** 23/11/2025

### 1.5 Autores (Nome, Área, Contato)

**Felipe Freitas Campos Picinin – Estudante de Engenharia de Software**
Contato: *[picinin.felipe2@gmail.com](mailto:picinin.felipe2@gmail.com)*

### 1.6 Responsável Principal (PI / Dono do Experimento)

**Felipe Freitas Campos Picinin**
Responsável pelas decisões metodológicas, execução do experimento, análise dos resultados e documentação da pesquisa.

### 1.7 Projeto / Produto / Iniciativa Relacionada

Este experimento está associado ao Trabalho de Conclusão de Curso (TCC) em Engenharia de Software, configurando-se como um projeto de **pesquisa aplicada** que investiga a efetividade comparativa de modelos de Machine Learning para **detecção automática de anomalias em pastagens**.

O estudo enquadra-se em:

* Avaliação empírica de técnicas de Machine Learning
* Experimentação controlada e medição de desempenho
* Análise comparativa baseada em evidências
* Validação científica de soluções tecnológicas

**Tecnologias envolvidas:** Python, PyTorch/TensorFlow, OpenCV, Scikit-learn, análise estatística.

---

## 2. Contexto e Problema

### 2.1 Descrição do Problema / Oportunidade

A pecuária extensiva no Brasil ocupa aproximadamente **159 milhões de hectares** e constitui uma das bases da economia nacional. A qualidade das pastagens impacta diretamente a produtividade, sustentabilidade ambiental e custos operacionais. Contudo, o monitoramento tradicional baseado em inspeção visual manual apresenta diversas limitações:

#### Problemas identificados:

* **Detecção tardia:** anomalias (degradação, pragas, solo exposto) são percebidas apenas quando o dano já é significativo.
* **Alto custo operacional:** grandes áreas exigem deslocamento constante de equipes técnicas.
* **Subjetividade:** avaliações variam entre observadores.
* **Cobertura limitada:** difícil monitoramento contínuo de propriedades extensas.

#### Oportunidade:

Com o avanço de drones e técnicas de visão computacional e Machine Learning, torna-se possível automatizar a identificação de anomalias em pastagens a partir de imagens aéreas. No entanto, ainda existe uma lacuna importante:

> **Não há clareza científica sobre quais modelos de ML são mais eficazes para detecção de anomalias em pastagens**, considerando suas características únicas (heterogeneidade, variabilidade sazonal, iluminação irregular).

---

### 2.2 Contexto Organizacional e Técnico

#### Contexto:

* **Tipo:** Pesquisa acadêmica em Engenharia de Software
* **Domínio:** Agricultura de precisão e pecuária
* **Equipe:** Pesquisador individual com orientação acadêmica
* **Processo:** Metodologia científica experimental baseada em dados quantitativos

#### Ambiente Técnico:

* **Linguagem:** Python 3.x
* **Frameworks de ML:** PyTorch ou TensorFlow
* **Processamento de imagens:** OpenCV
* **Análise estatística:** Scikit-learn, SciPy, pandas
* **Infraestrutura:** GPU local ou Google Colab
* **Versionamento:** Git/GitHub

#### Dados:

* Dataset de imagens aéreas de pastagens capturadas
* Resolução suficiente para distinguir padrões de anomalia
* Dados rotulados (degradação, solo exposto, pragas, plantas invasoras)

---

### 2.3 Trabalhos e Evidências Prévias (Internos e Externos)

#### Evidências externas – literatura:

**Machine Learning na agricultura:**

* CNNs amplamente utilizadas para doenças de plantas
* Modelos como ResNet, VGG, YOLO e U-Net com alto desempenho
* Transfer learning reduz necessidade de grandes datasets

**Sensoriamento remoto com drones:**

* Drones usados para monitoramento de safras
* Uso de NDVI para detecção de estresse
* Imagens multiespectrais aumentam precisão, mas também custos

**Comparações entre modelos:**

* Trade-offs entre acurácia e eficiência computacional
* Nenhum modelo universalmente superior; depende do contexto
* Meta-estudos reforçam a necessidade de experimentação específica

#### Evidências internas:

* Projeto ainda sem experimentos prévios
* Experiência do pesquisador com Python e fundamentos de ML
* Acesso à orientação acadêmica especializada

#### Lacunas identificadas:

* Poucos estudos comparativos focados em **pastagens brasileiras**
* Falta de datasets públicos rotulados com anomalias de pastagem
* Necessidade de avaliação empírica rigorosa em ambiente controlado

---

### 2.4 Referencial Teórico e Empírico Essencial

#### Conceitos fundamentais:

**Machine Learning e Deep Learning:**

* Aprendizado supervisionado
* Redes Neurais Convolucionais (CNNs)
* Transfer learning

**Detecção de anomalias:**

* Classificação binária e multi-classe
* Segmentação semântica
* Reconstrução por autoencoders

**Avaliação de modelos:**

* Acurácia, Precisão, Recall, F1-score
* Validação cruzada
* Trade-offs entre desempenho e custo computacional

#### Modelos candidatos:

* **ResNet** – eficaz para classificação profunda
* **MobileNet** – leve e eficiente
* **EfficientNet** – arquitetura otimizada
* **U-Net** – segmentação precisa
* **YOLO** – detecção em tempo real

#### Hipóteses teóricas (da literatura):

* Modelos mais profundos tendem a maior acurácia
* Transfer learning reduz tempo de treinamento
* Modelos leves têm melhor relação custo-benefício

#### Referências empíricas:

* Benchmarks mostram desempenho consistente de CNNs
* F1-scores em agricultura variam entre 0.75 e 0.95
* 1000–5000 imagens rotuladas costumam ser suficientes para transfer learning

---

## 3. Objetivos e Questões (Goal / Question / Metric)

### 3.1 Objetivo Geral (Goal Template)

**Analisar** diferentes modelos de Machine Learning (ResNet, MobileNet, EfficientNet, U-Net, YOLO)  
**com o propósito de** avaliar e comparar sua efetividade na detecção de anomalias em pastagens  
**com respeito à** acurácia de detecção, eficiência computacional, robustez e viabilidade prática  
**do ponto de vista** do pesquisador e potenciais usuários finais (produtores rurais e técnicos agrônomos)  
**no contexto de** um estudo experimental controlado utilizando dataset de imagens aéreas de pastagens brasileiras capturadas por drones.

---

### 3.2 Objetivos Específicos

**O1 – Avaliar a eficácia de detecção de anomalias**  
Determinar qual(is) modelo(s) apresenta(m) melhor desempenho na identificação correta de anomalias em pastagens (degradação, solo exposto, pragas, plantas invasoras) considerando métricas de classificação.

**O2 – Comparar a eficiência computacional dos modelos**  
Mensurar e comparar o custo computacional de cada modelo em termos de tempo de treinamento, tempo de inferência e consumo de recursos (memória, processamento), visando identificar alternativas viáveis para implementação em diferentes cenários operacionais.

**O3 – Avaliar a robustez dos modelos**  
Analisar a capacidade dos modelos de manter desempenho consistente sob diferentes condições de entrada (variações de iluminação, resolução, ângulos de captura, estações do ano) e verificar a estabilidade dos resultados através de validação cruzada.

**O4 – Determinar a viabilidade prática de implementação**  
Avaliar a relação custo-benefício entre desempenho e requisitos técnicos de cada modelo, considerando cenários reais de aplicação (dispositivos embarcados em drones, processamento em nuvem, sistemas edge computing).

**O5 – Identificar padrões de erro e limitações**  
Caracterizar os tipos de erros cometidos por cada modelo (falsos positivos, falsos negativos), identificar classes de anomalias mais desafiadoras e documentar limitações específicas de cada abordagem para orientar melhorias futuras.

---

### 3.3 Questões de Pesquisa / de Negócio

#### Relacionadas ao Objetivo O1 (Eficácia de Detecção):

**Q1.1:** Qual modelo apresenta a maior acurácia geral na detecção de anomalias em pastagens?

**Q1.2:** Qual modelo apresenta o melhor equilíbrio entre precisão e recall (F1-score) para diferentes tipos de anomalias?

**Q1.3:** Qual modelo demonstra maior capacidade de generalização em dados não vistos durante o treinamento?

#### Relacionadas ao Objetivo O2 (Eficiência Computacional):

**Q2.1:** Qual modelo requer menor tempo de treinamento mantendo desempenho aceitável?

**Q2.2:** Qual modelo apresenta menor tempo de inferência por imagem, viabilizando processamento em tempo real?

**Q2.3:** Qual modelo demanda menos recursos computacionais (memória e processamento) durante execução?

#### Relacionadas ao Objetivo O3 (Robustez):

**Q3.1:** Qual modelo mantém desempenho mais estável sob diferentes condições de iluminação e qualidade de imagem?

**Q3.2:** Qual modelo apresenta menor variância de desempenho entre diferentes folds de validação cruzada?

**Q3.3:** Qual modelo demonstra maior resiliência a ruídos e artefatos nas imagens?

#### Relacionadas ao Objetivo O4 (Viabilidade Prática):

**Q4.1:** Qual modelo oferece a melhor relação entre acurácia e custo computacional para implementação em dispositivos com recursos limitados?

**Q4.2:** Qual modelo apresenta melhor custo-benefício considerando tempo de desenvolvimento, treinamento e implantação?

**Q4.3:** Quais modelos são tecnicamente viáveis para processamento embarcado em drones de médio porte?

#### Relacionadas ao Objetivo O5 (Padrões de Erro):

**Q5.1:** Quais tipos de anomalias geram mais erros de classificação em cada modelo?

**Q5.2:** Qual modelo produz menor taxa de falsos positivos em áreas de pastagem saudável?

**Q5.3:** Qual modelo apresenta melhor desempenho na detecção de anomalias em estágios iniciais de degradação?

---

### 3.4 Métricas Associadas (GQM)

#### Tabela GQM Completa

| **Objetivo** | **Questão** | **Métricas Associadas** |
|--------------|-------------|-------------------------|
| **O1 – Avaliar a eficácia de detecção de anomalias** | Q1.1: Qual modelo apresenta a maior acurácia geral? | M1: Acurácia Global<br>M2: Acurácia por Classe |
| | Q1.2: Qual modelo apresenta o melhor equilíbrio entre precisão e recall? | M3: F1-Score Macro<br>M4: Precisão Média<br>M5: Recall Médio |
| | Q1.3: Qual modelo demonstra maior capacidade de generalização? | M6: AUC-ROC<br>M7: Acurácia no Conjunto de Teste |
| **O2 – Comparar a eficiência computacional** | Q2.1: Qual modelo requer menor tempo de treinamento? | M8: Tempo Total de Treinamento<br>M9: Tempo por Época |
| | Q2.2: Qual modelo apresenta menor tempo de inferência? | M10: Tempo Médio de Inferência por Imagem<br>M11: Taxa de Processamento (FPS) |
| | Q2.3: Qual modelo demanda menos recursos computacionais? | M12: Consumo Médio de Memória GPU<br>M13: Número de Parâmetros do Modelo |
| **O3 – Avaliar a robustez dos modelos** | Q3.1: Qual modelo mantém desempenho estável sob diferentes condições? | M1: Acurácia Global (por condição)<br>M14: Desvio Padrão da Acurácia entre Condições |
| | Q3.2: Qual modelo apresenta menor variância entre folds? | M15: Desvio Padrão do F1-Score (K-fold)<br>M3: F1-Score Macro (por fold) |
| | Q3.3: Qual modelo demonstra maior resiliência a ruídos? | M1: Acurácia Global (com ruído adicionado)<br>M16: Taxa de Degradação de Desempenho |
| **O4 – Determinar a viabilidade prática** | Q4.1: Qual modelo oferece melhor relação acurácia/custo? | M17: Índice de Eficiência (F1-Score / Tempo Inferência)<br>M13: Número de Parâmetros |
| | Q4.2: Qual modelo apresenta melhor custo-benefício total? | M8: Tempo Total de Treinamento<br>M18: Custo Computacional Estimado |
| | Q4.3: Quais modelos são viáveis para processamento embarcado? | M12: Consumo Médio de Memória GPU<br>M11: Taxa de Processamento (FPS) |
| **O5 – Identificar padrões de erro e limitações** | Q5.1: Quais tipos de anomalias geram mais erros? | M19: Taxa de Erro por Classe<br>M2: Acurácia por Classe |
| | Q5.2: Qual modelo produz menor taxa de falsos positivos? | M20: Taxa de Falsos Positivos (FPR)<br>M4: Precisão Média |
| | Q5.3: Qual modelo detecta melhor anomalias iniciais? | M5: Recall Médio (para classe "degradação leve")<br>M3: F1-Score (para classe "degradação leve") |

---

#### Tabela Detalhada de Métricas

| **Código** | **Nome da Métrica** | **Descrição** | **Unidade / Escala** |
|------------|---------------------|---------------|----------------------|
| **M1** | Acurácia Global | Proporção de predições corretas em relação ao total de predições realizadas | Percentual (0-100%) |
| **M2** | Acurácia por Classe | Proporção de predições corretas para cada classe específica de anomalia (degradação, solo exposto, pragas, invasoras) | Percentual (0-100%) por classe |
| **M3** | F1-Score Macro | Média harmônica entre precisão e recall, calculada como média simples entre todas as classes | Escala 0-1 (adimensional) |
| **M4** | Precisão Média | Proporção de verdadeiros positivos dentre todas as predições positivas, média entre classes | Percentual (0-100%) |
| **M5** | Recall Médio | Proporção de verdadeiros positivos identificados dentre todos os casos reais positivos, média entre classes | Percentual (0-100%) |
| **M6** | AUC-ROC | Área sob a curva ROC (Receiver Operating Characteristic), medindo capacidade de discriminação do modelo | Escala 0-1 (adimensional) |
| **M7** | Acurácia no Conjunto de Teste | Acurácia medida especificamente no conjunto de teste (dados não vistos), indicando generalização | Percentual (0-100%) |
| **M8** | Tempo Total de Treinamento | Duração total necessária para completar o treinamento do modelo até convergência | Minutos ou horas |
| **M9** | Tempo por Época | Duração média de cada época de treinamento | Segundos ou minutos |
| **M10** | Tempo Médio de Inferência por Imagem | Tempo médio necessário para processar uma única imagem e produzir predição | Milissegundos (ms) |
| **M11** | Taxa de Processamento (FPS) | Número de imagens processadas por segundo (frames per second) | Imagens/segundo |
| **M12** | Consumo Médio de Memória GPU | Quantidade média de memória da GPU utilizada durante inferência | Megabytes (MB) ou Gigabytes (GB) |
| **M13** | Número de Parâmetros do Modelo | Quantidade total de parâmetros treináveis na arquitetura do modelo | Milhões de parâmetros |
| **M14** | Desvio Padrão da Acurácia entre Condições | Medida de variabilidade da acurácia quando testado sob diferentes condições (iluminação, resolução, etc.) | Pontos percentuais (pp) |
| **M15** | Desvio Padrão do F1-Score (K-fold) | Medida de variabilidade do F1-Score entre diferentes folds na validação cruzada | Escala 0-1 (adimensional) |
| **M16** | Taxa de Degradação de Desempenho | Percentual de redução na acurácia quando ruído ou distorções são adicionados às imagens | Percentual (0-100%) |
| **M17** | Índice de Eficiência | Razão entre F1-Score e tempo de inferência, indicando eficiência prática | F1/ms (adimensional) |
| **M18** | Custo Computacional Estimado | Estimativa de custo total considerando tempo de GPU e recursos necessários para treinamento e implantação | Dólares ($) ou horas-GPU |
| **M19** | Taxa de Erro por Classe | Proporção de erros (falsos positivos + falsos negativos) para cada classe de anomalia | Percentual (0-100%) por classe |
| **M20** | Taxa de Falsos Positivos (FPR) | Proporção de casos negativos incorretamente classificados como positivos | Percentual (0-100%) |

---

## 4. Escopo e Contexto do Experimento

### 4.1 Escopo Funcional / de Processo (Incluído e Excluído)

#### **Template de Escopo:**

| **Categoria** | **Incluído no Experimento** | **Excluído do Experimento** |
|---------------|-----------------------------|-----------------------------|
| **Modelos Avaliados** | • ResNet (ResNet-50)<br>• MobileNet (MobileNetV2)<br>• EfficientNet (EfficientNet-B0)<br>• U-Net (arquitetura padrão)<br>• YOLO (YOLOv8) | • Modelos proprietários fechados<br>• Arquiteturas experimentais não publicadas<br>• Modelos com requisitos de licenciamento restritivo<br>• Variantes específicas além das citadas |
| **Tipos de Anomalias** | • Degradação de pastagem<br>• Solo exposto<br>• Pragas visíveis<br>• Plantas invasoras | • Doenças microscópicas<br>• Deficiências nutricionais não visíveis<br>• Problemas hídricos sem manifestação visual clara<br>• Anomalias em estágios imperceptíveis |
| **Dataset e Imagens** | • Imagens RGB capturadas por drones<br>• Resolução mínima de 1920x1080<br>• Pastagens brasileiras (Cerrado, Mata Atlântica)<br>• Imagens rotuladas por especialistas<br>• Condições variadas de iluminação natural | • Imagens multiespectrais ou hiperespectrais<br>• Imagens de satélite de baixa resolução<br>• Dados de sensores térmicos ou LIDAR<br>• Imagens de pastagens fora do Brasil<br>• Vídeos contínuos (apenas frames estáticos) |
| **Etapas do Pipeline** | • Pré-processamento de imagens<br>• Treinamento de modelos com transfer learning<br>• Validação cruzada (K-fold)<br>• Teste em conjunto separado<br>• Análise estatística comparativa<br>• Documentação de resultados | • Coleta de imagens em campo (dataset já existente)<br>• Desenvolvimento de novos algoritmos<br>• Rotulação de dados (já realizada)<br>• Implantação em produção<br>• Testes com usuários reais<br>• Integração com sistemas de gestão de fazendas |
| **Métricas e Análises** | • Métricas de classificação (acurácia, F1, precisão, recall)<br>• Métricas de eficiência (tempo, memória, FPS)<br>• Análise de robustez<br>• Análise estatística (testes de hipótese)<br>• Análise de custo-benefício | • Análises qualitativas extensivas com usuários<br>• Estudos de usabilidade de interface<br>• Análise econômica detalhada de ROI<br>• Avaliação de impacto ambiental<br>• Comparação com métodos não-ML |
| **Ambiente Técnico** | • Python 3.x<br>• PyTorch ou TensorFlow<br>• Hardware GPU (local ou Google Colab)<br>• Bibliotecas padrão (OpenCV, scikit-learn) | • Implementação em outras linguagens (C++, Java)<br>• Hardware especializado (TPUs, FPGAs)<br>• Ambientes de produção distribuídos<br>• Otimizações específicas de hardware |
| **Participantes** | • Pesquisador principal (execução)<br>• Orientador acadêmico (supervisão)<br>• Especialistas agrônomos (validação de rotulação) | • Desenvolvedores de software adicionais<br>• Equipes de produção rural<br>• Usuários finais em campo<br>• Consultores externos |
| **Documentação** | • Plano experimental detalhado<br>• Código-fonte documentado<br>• Análise estatística formal<br>• Relatório final científico<br>• Apresentação de resultados | • Manuais de usuário<br>• Documentação de API para produção<br>• Materiais de treinamento para usuários<br>• Documentação de manutenção operacional |

---

### 4.2 Contexto do Estudo

**Tipo de Organização:**  
Instituição de ensino superior (universidade), especificamente dentro de um programa de graduação em Engenharia de Software. Trata-se de pesquisa acadêmica aplicada com potencial para transferência tecnológica futura.

**Tipo de Projeto:**  
Trabalho de Conclusão de Curso (TCC) – pesquisa experimental controlada com foco em avaliação empírica e comparação de tecnologias de Machine Learning. O projeto possui caráter científico e metodologia rigorosa, mas mantém ênfase em aplicabilidade prática.

**Criticidade:**  
Média-Alta. Embora não seja um sistema crítico de segurança, os resultados têm potencial de impacto econômico significativo para o setor agropecuário. Erros de detecção podem levar a perdas de produtividade ou custos desnecessários, mas não representam riscos à vida ou segurança imediata.

**Experiência dos Participantes:**

* **Pesquisador Principal:** Estudante de graduação em Engenharia de Software com:
  * Conhecimento intermediário em Python
  * Fundamentos sólidos de Machine Learning
  * Experiência prévia com processamento de imagens
  * Familiaridade com ferramentas de análise de dados
  * Primeira experiência com experimentação científica formal

* **Orientador:** Professor doutor com expertise em:
  * Engenharia de Software Experimental
  * Métodos empíricos de pesquisa
  * Supervisão de projetos de Machine Learning

* **Especialistas de Domínio:** Agrônomos com:
  * Experiência prática em gestão de pastagens
  * Capacidade de validar rotulações de anomalias
  * Conhecimento das características regionais das pastagens brasileiras

**Tamanho e Recursos:**  
Projeto individual com suporte acadêmico. Recursos limitados a infraestrutura universitária, ferramentas open-source e plataformas gratuitas de computação em nuvem (Google Colab).

---

### 4.3 Premissas

As seguintes suposições são consideradas verdadeiras para viabilizar a execução do experimento:

1. **Disponibilidade de Dataset:** Existe um dataset de imagens aéreas de pastagens já capturadas, com volume suficiente (mínimo de 2000 imagens) e qualidade adequada para treinamento e validação dos modelos.

2. **Rotulação Confiável:** As imagens do dataset foram previamente rotuladas por especialistas agrônomos qualificados, com nível de concordância inter-anotadores adequado (Kappa > 0.7).

3. **Acesso a Recursos Computacionais:** Haverá acesso contínuo a GPU (local ou via Google Colab) durante todo o período do experimento, com capacidade suficiente para treinar os cinco modelos selecionados.

4. **Estabilidade de Ferramentas:** As bibliotecas e frameworks de Machine Learning (PyTorch/TensorFlow, OpenCV, scikit-learn) permanecerão estáveis e compatíveis durante a execução do experimento.

5. **Tempo Disponível:** O pesquisador terá dedicação de aproximadamente 20 horas semanais durante 4 meses para execução completa do experimento.

6. **Orientação Acadêmica:** Haverá disponibilidade regular do orientador para revisões metodológicas e discussão de resultados (mínimo de 1 reunião quinzenal).

7. **Conhecimento Técnico Suficiente:** O pesquisador possui ou conseguirá adquirir rapidamente o conhecimento técnico necessário para implementar e ajustar os modelos selecionados.

8. **Reprodutibilidade das Condições:** Será possível simular e controlar condições variadas de iluminação, ruído e qualidade de imagem através de técnicas de data augmentation.

9. **Validação Estatística:** Os resultados obtidos serão estatisticamente significativos com o tamanho de amostra disponível (poder estatístico adequado).

10. **Representatividade do Dataset:** As imagens disponíveis são representativas das condições reais encontradas em pastagens brasileiras extensivas.

---

### 4.4 Restrições

As seguintes limitações práticas impõem boundaries ao desenho do experimento:

**Restrições de Tempo:**

* Prazo máximo de 4 meses para conclusão completa (incluindo análise e documentação)
* Tempo de GPU limitado em plataformas gratuitas (Google Colab: sessões de 12h)
* Disponibilidade limitada do pesquisador (dedicação parcial, cursando outras disciplinas)

**Restrições Orçamentárias:**

* Orçamento zero para aquisição de hardware, software ou serviços em nuvem pagos
* Dependência de ferramentas open-source e recursos gratuitos
* Impossibilidade de contratar consultoria especializada ou serviços de rotulação

**Restrições Técnicas:**

* Capacidade limitada de GPU (VRAM máxima de 16GB em recursos gratuitos)
* Impossibilidade de testar em hardware especializado (TPUs, FPGAs)
* Limitação a modelos com implementações públicas disponíveis
* Sem acesso a ferramentas proprietárias de análise de imagens

**Restrições de Dados:**

* Impossibilidade de coletar novas imagens em campo (custo de drone e deslocamento)
* Dataset fixo sem possibilidade de expansão significativa
* Sem acesso a dados multiespectrais ou hiperespectrais
* Limitação a imagens já disponíveis sem controle sobre condições de captura

**Restrições de Escopo:**

* Experimento limitado a ambiente acadêmico (não é um projeto de produção)
* Sem possibilidade de validação com usuários reais em larga escala
* Impossibilidade de implementar sistema completo end-to-end
* Sem integração com sistemas de gestão de fazendas existentes

**Restrições Organizacionais:**

* Necessidade de aprovação do comitê de ética caso envolva dados sensíveis
* Conformidade com regulamentos acadêmicos da instituição
* Dependência da disponibilidade do orientador para aprovações
* Necessidade de seguir cronograma acadêmico institucional

**Restrições de Acesso:**

* Impossibilidade de acessar propriedades rurais para validação em campo
* Limitação de comunicação com potenciais stakeholders (produtores rurais)
* Sem acesso a especialistas agrônomos em tempo integral

---

### 4.5 Limitações Previstas

Os seguintes fatores podem afetar a **validade externa** (generalização) dos resultados:

**Limitações de Contexto Geográfico:**

* Dataset limitado a pastagens brasileiras (Cerrado e Mata Atlântica), podendo não generalizar para outros biomas ou países
* Variabilidade climática específica do período de captura pode não representar todas as estações
* Características de solo e vegetação regionais podem influenciar resultados

**Limitações de Representatividade do Dataset:**

* Tamanho do dataset pode ser insuficiente para capturar toda variabilidade de anomalias existentes
* Possível viés de seleção nas imagens capturadas (áreas mais acessíveis ou problemáticas)
* Distribuição desbalanceada entre classes de anomalias pode afetar resultados
* Ausência de casos raros ou extremos de degradação

**Limitações Tecnológicas:**

* Resultados obtidos com hardware de GPU específico podem não replicar em outros ambientes
* Transfer learning aplicado pode não generalizar para pastagens muito diferentes do dataset de pré-treinamento (ImageNet)
* Desempenho medido em ambiente controlado pode diferir de implementação em drones reais
* Latência de rede e limitações de banda não são consideradas

**Limitações Metodológicas:**

* Experimento conduzido por pesquisador único pode introduzir vieses não detectados
* Validação cruzada, embora robusta, não substitui testes em dados completamente independentes de diferentes regiões
* Métricas quantitativas não capturam aspectos qualitativos da experiência do usuário
* Falta de validação com usuários reais em campo limita avaliação de usabilidade

**Limitações Temporais:**

* Snapshot único no tempo – mudanças em bibliotecas de ML podem alterar resultados futuros
* Modelos podem rapidamente ficar obsoletos com surgimento de arquiteturas mais modernas
* Comparação limitada aos modelos selecionados, podendo haver alternativas superiores não avaliadas

**Limitações de Generalização para Produção:**

* Resultados experimentais podem não refletir desafios de implantação em produção (edge computing, bateria limitada, conectividade instável)
* Custo-benefício avaliado teoricamente pode diferir de análise econômica real em fazendas
* Integração com workflows existentes não é testada
* Manutenção e atualização de modelos ao longo do tempo não é considerada

**Limitações de Validação:**

* Rotulações humanas, mesmo de especialistas, possuem subjetividade e podem conter erros
* Impossibilidade de validação ground-truth absoluta para todas as imagens
* Falta de validação longitudinal (acompanhamento da evolução das anomalias ao longo do tempo)

**Observação:** Estas limitações serão explicitamente documentadas na seção de discussão do relatório final, e recomendações para estudos futuros abordarão como superá-las.

---

## 5. Stakeholders e Impacto Esperado

### 5.1 Stakeholders Principais

| **Grupo** | **Papel / Descrição** |
|-----------|-----------------------|
| **Pesquisador (autor do TCC)** | Executor principal do experimento, responsável por todas as etapas técnicas e análise de resultados |
| **Orientador Acadêmico** | Supervisor científico, garantindo rigor metodológico e qualidade da pesquisa |
| **Banca Avaliadora** | Professores e especialistas que avaliarão a qualidade científica e contribuição do trabalho |
| **Comunidade Acadêmica de Engenharia de Software** | Pesquisadores interessados em experimentação, métricas de ML e aplicações de IA |
| **Comunidade Acadêmica de Agricultura de Precisão** | Pesquisadores focados em aplicações de tecnologia no agronegócio |
| **Produtores Rurais / Pecuaristas** | Potenciais beneficiários futuros da tecnologia para monitoramento de pastagens (stakeholders indiretos) |
| **Técnicos Agrônomos e Zootecnistas** | Profissionais que poderiam utilizar ferramentas baseadas nos resultados para consultoria |
| **Desenvolvedores de Soluções AgTech** | Empresas e startups interessadas em incorporar ML em produtos para agropecuária |
| **Instituição de Ensino (Universidade)** | Interessada em produção científica de qualidade e potencial inovação tecnológica |

---

### 5.2 Interesses e Expectativas dos Stakeholders

| **Stakeholder** | **Interesses e Expectativas** |
|-----------------|-------------------------------|
| **Pesquisador (autor)** | • Desenvolver competências em experimentação científica e ML<br>• Produzir TCC de qualidade para conclusão do curso<br>• Gerar conhecimento aplicável e publicável<br>• Construir portfólio profissional com projeto relevante<br>• Obter aprovação e reconhecimento acadêmico |
| **Orientador Acadêmico** | • Garantir rigor metodológico e científico do trabalho<br>• Orientar aplicação correta de técnicas experimentais<br>• Contribuir para formação do aluno<br>• Potencial publicação científica em coautoria<br>• Reforçar reputação da linha de pesquisa |
| **Banca Avaliadora** | • Avaliar qualidade científica e contribuição original<br>• Verificar adequação metodológica<br>• Validar conclusões e análises estatísticas<br>• Garantir padrões acadêmicos da instituição |
| **Comunidade Acadêmica (ES)** | • Evidências empíricas sobre efetividade de modelos de ML<br>• Metodologia experimental replicável<br>• Insights sobre trade-offs entre modelos<br>• Contribuição para body of knowledge em experimentação |
| **Comunidade Acadêmica (AgTech)** | • Aplicação prática de ML em agricultura<br>• Benchmarks de desempenho para detecção de anomalias em pastagens<br>• Identificação de direções promissoras para pesquisa futura<br>• Base para estudos comparativos |
| **Produtores Rurais** | • Validação de viabilidade técnica de soluções automatizadas<br>• Indicação de custo-benefício de diferentes tecnologias<br>• Potencial redução de custos operacionais futura<br>• Melhoria na gestão de pastagens |
| **Técnicos Agrônomos** | • Ferramentas baseadas em evidência para recomendação<br>• Entendimento de limitações e potencial da tecnologia<br>• Insights sobre tipos de anomalias detectáveis automaticamente |
| **Desenvolvedores AgTech** | • Benchmarks técnicos para desenvolvimento de produtos<br>• Identificação de modelos mais promissores para implementação<br>• Evidências para decisões de arquitetura de sistemas<br>• Avaliação de viabilidade técnica e econômica |
| **Instituição de Ensino** | • Produção científica de qualidade<br>• Demonstração de excelência em formação<br>• Potencial inovação com impacto social e econômico<br>• Fortalecimento de parcerias com setor produtivo |

---

### 5.3 Impactos Potenciais no Processo / Produto

#### **Durante a Execução do Experimento:**

**Impactos Positivos:**

* **Aquisição de Competências:** Desenvolvimento de habilidades práticas em ML, experimentação e análise estatística pelo pesquisador
* **Geração de Conhecimento:** Produção de evidências empíricas úteis para comunidade acadêmica e indústria
* **Networking Acadêmico:** Potencial colaboração com especialistas agrônomos e pesquisadores de áreas correlatas
* **Visibilidade Institucional:** Projeto pode gerar apresentações em eventos e publicações científicas

**Impactos Negativos / Desafios:**

* **Carga de Trabalho Intensiva:** Demanda significativa de tempo do pesquisador, podendo impactar desempenho em outras disciplinas
* **Pressão de Prazo:** Necessidade de conclusão dentro do calendário acadêmico pode gerar stress
* **Risco de Bloqueios Técnicos:** Dificuldades técnicas imprevistas podem atrasar cronograma
* **Dependência de Recursos:** Falhas em infraestrutura (GPU, acesso a dados) podem comprometer execução

#### **Após a Conclusão do Experimento:**

**Impactos no Conhecimento Científico:**

* **Evidências Comparativas:** Contribuição para literatura com dados empíricos sobre efetividade de modelos de ML em contexto específico de pastagens
* **Metodologia Replicável:** Outros pesquisadores poderão replicar ou estender o estudo
* **Identificação de Limitações:** Documentação de desafios e limitações orienta pesquisas futuras

**Impactos no Produto / Tecnologia:**

* **Prova de Conceito:** Validação de viabilidade técnica pode motivar desenvolvimento de sistemas reais
* **Orientação de Escolhas Técnicas:** Desenvolvedores de soluções AgTech terão dados para decisões arquiteturais
* **Baseline para Comparações:** Resultados servem como referência para avaliação de melhorias futuras

**Impactos na Indústria / Sociedade:**

* **Potencial Econômico:** Resultados podem catalisar adoção de tecnologias de monitoramento automatizado, reduzindo custos e aumentando produtividade
* **Sustentabilidade Ambiental:** Detecção precoce de degradação pode contribuir para práticas mais sustentáveis de manejo de pastagens
* **Democratização do Conhecimento:** Publicação aberta de resultados e código beneficia pequenos produtores e startups

**Impactos no Processo de Pesquisa Futuro:**

* **Base para Extensões:** Trabalho pode ser estendido em mestrado/doutorado ou projetos de pesquisa aplicada
* **Colaborações Futuras:** Networking gerado pode resultar em parcerias acadêmicas ou industriais
* **Transferência Tecnológica:** Resultados positivos podem motivar criação de spin-offs ou licenciamento de tecnologia

**Riscos de Impacto Negativo:**

* **Má Interpretação de Resultados:** Conclusões podem ser generalizadas indevidamente se limitações não forem compreendidas
* **Expectativas Irreais:** Stakeholders industriais podem superestimar maturidade da tecnologia baseando-se em resultados experimentais
* **Uso Inadequado:** Implementações apressadas sem validação adequada podem gerar prejuízos
* **Obsolescência Rápida:** Avanços rápidos em ML podem tornar resultados menos relevantes em poucos anos

---

## 6. Riscos de Alto Nível, Premissas e Critérios de Sucesso

### 6.1 Riscos de Alto Nível (Negócio, Técnicos, etc.)

#### **Riscos de Negócio / Acadêmicos:**

| **Risco** | **Probabilidade** | **Impacto** | **Mitigação** |
|-----------|-------------------|-------------|---------------|
| **R1: Não atender requisitos acadêmicos da banca** | Média | Alto | • Revisões periódicas com orientador<br>• Alinhamento precoce com critérios de avaliação<br>• Peer review com colegas |
| **R2: Prazo insuficiente para conclusão** | Média-Alta | Alto | • Cronograma detalhado com buffers<br>• Priorização de objetivos essenciais<br>• Planejamento de entregas parciais |
| **R3: Resultados inconclusivos ou sem significância estatística** | Média | Médio | • Cálculo prévio de poder estatístico<br>• Dataset suficientemente grande<br>• Múltiplas métricas de avaliação |
| **R4: Baixo impacto científico (contribuição limitada)** | Baixa | Médio | • Revisão cuidadosa da literatura<br>• Foco em lacuna claramente identificada<br>• Discussão aprofundada de limitações |

#### **Riscos Técnicos:**

| **Risco** | **Probabilidade** | **Impacto** | **Mitigação** |
|-----------|-------------------|-------------|---------------|
| **R5: Indisponibilidade ou falhas em recursos computacionais (GPU)** | Média | Alto | • Uso de múltiplas plataformas (Colab + local)<br>• Checkpoints frequentes durante treinamento<br>• Redução de complexidade se necessário |
| **R6: Incompatibilidades entre bibliotecas de ML** | Baixa-Média | Médio | • Uso de ambientes virtuais isolados<br>• Fixação de versões de dependências<br>• Testes prévios de compatibilidade |
| **R7: Problemas na qualidade ou rotulação do dataset** | Média | Alto | • Validação prévia da qualidade das rotulações<br>• Análise de concordância inter-anotadores<br>• Limpeza e pré-processamento cuidadosos |
| **R8: Dificuldade técnica em implementar modelos complexos** | Média | Médio | • Uso de implementações pré-existentes quando possível<br>• Tutoriais e documentação oficial<br>• Suporte de comunidades online |
| **R9: Overfitting severo dos modelos** | Média | Médio | • Validação cruzada rigorosa<br>• Técnicas de regularização<br>• Data augmentation<br>• Monitoramento de gap treino-validação |
| **R10: Perda de dados ou código por falhas** | Baixa | Alto | • Versionamento com Git/GitHub<br>• Backups regulares em múltiplos locais<br>• Documentação inline do código |

#### **Riscos Operacionais:**

| **Risco** | **Probabilidade** | **Impacto** | **Mitigação** |
|-----------|-------------------|-------------|---------------|
| **R11: Indisponibilidade do orientador em momentos críticos** | Baixa | Médio | • Agendamento antecipado de reuniões<br>• Comunicação assíncrona eficiente<br>• Autonomia na tomada de decisões técnicas |
| **R12: Problemas pessoais ou de saúde do pesquisador** | Baixa | Alto | • Buffer no cronograma<br>• Seguro saúde ativo<br>• Plano de contingência com orientador |
| **R13: Mudanças em requisitos ou escopo acadêmico** | Baixa | Médio | • Documentação clara de escopo desde início<br>• Revisões formais de progresso<br>• Flexibilidade metodológica planejada |

#### **Riscos Externos:**

| **Risco** | **Probabilidade** | **Impacto** | **Mitigação** |
|-----------|-------------------|-------------|---------------|
| **R14: Indisponibilidade de plataformas gratuitas (Colab)** | Baixa | Médio | • Diversificação de recursos computacionais<br>• Alternativas identificadas (Kaggle, AWS free tier) |
| **R15: Mudanças drásticas em frameworks de ML durante execução** | Baixa | Baixo | • Fixação de versões<br>• Ambientes containerizados se possível |

---

### 6.2 Critérios de Sucesso Globais (Go / No-Go)

#### **Critérios de Sucesso Mínimos (Must-Have):**

Para que o experimento seja considerado bem-sucedido e o TCC aprovado, os seguintes critérios **DEVEM** ser atendidos:

**CS1 – Execução Completa do Protocolo Experimental:**

* ✅ Todos os 5 modelos selecionados devem ser treinados e avaliados
* ✅ Validação cruzada K-fold (mínimo K=5) deve ser executada para todos os modelos
* ✅ Conjunto de teste independente deve ser utilizado para avaliação final

**CS2 – Coleta de Métricas Essenciais:**

* ✅ Pelo menos 8 das 10 métricas distintas definidas devem ser coletadas para todos os modelos
* ✅ Métricas de acurácia, F1-Score e tempo de inferência são **obrigatórias**
* ✅ Dados devem ser registrados de forma reproduzível

**CS3 – Análise Estatística Adequada:**

* ✅ Comparação estatística entre modelos deve ser realizada (testes de hipótese apropriados)
* ✅ Intervalos de confiança ou medidas de variabilidade devem ser reportados
* ✅ Significância estatística das diferenças deve ser verificada (p-valor < 0.05)

**CS4 – Documentação Científica Completa:**

* ✅ Relatório final deve seguir estrutura acadêmica padrão (IMRaD)
* ✅ Metodologia deve estar descrita com detalhes suficientes para replicação
* ✅ Limitações e ameaças à validade devem ser explicitamente discutidas
* ✅ Código-fonte deve estar disponível e documentado

**CS5 – Conclusões Baseadas em Evidências:**

* ✅ Pelo menos 3 das 5 questões de pesquisa por objetivo devem ser respondidas com dados
* ✅ Recomendações devem estar claramente justificadas pelos resultados
* ✅ Deve haver identificação clara de qual(is) modelo(s) apresenta(m) melhor desempenho em cada aspecto avaliado

---

#### **Critérios de Sucesso Desejáveis (Should-Have):**

Para que o experimento seja considerado de **alta qualidade**, é desejável que:

**CS6 – Resultados Estatisticamente Significativos:**

* 🎯 Diferenças entre modelos devem ser estatisticamente significativas (não apenas numéricas)
* 🎯 Pelo menos um modelo deve demonstrar desempenho claramente superior em acurácia (diferença > 5%)

**CS7 – Análise de Robustez Bem-Sucedida:**

* 🎯 Variação de desempenho entre condições adversas deve ser < 15% para pelo menos um modelo
* 🎯 Pelo menos 2 modelos devem demonstrar robustez adequada (desvio padrão baixo na validação cruzada)

**CS8 – Viabilidade Prática Demonstrada:**

* 🎯 Pelo menos 2 modelos devem atender requisitos de processamento em tempo real (> 10 FPS)
* 🎯 Pelo menos 1 modelo deve ser viável para implementação embarcada (< 100MB, < 1GB RAM)

**CS9 – Contribuição Científica Relevante:**

* 🎯 Resultados devem possibilitar submissão a conferência ou periódico científico
* 🎯 Insights gerados devem ser úteis para desenvolvedores de soluções AgTech

**CS10 – Código e Dados Compartilháveis:**

* 🎯 Código deve estar em repositório público (GitHub) sob licença aberta
* 🎯 Dataset (ou amostra representativa) deve ser disponibilizado se possível

---

#### **Critérios de Descontinuação (No-Go):**

O experimento deve ser **reavaliado ou descontinuado** se:

**NG1 – Problemas Críticos de Dados:**

* ❌ Dataset apresenta problemas graves de qualidade que invalidam resultados (descoberto após análise inicial)
* ❌ Rotulações são inconsistentes (Kappa inter-anotadores < 0.5)
* ❌ Tamanho do dataset é insuficiente para validação estatística (< 500 imagens úteis)

**NG2 – Inviabilidade Técnica:**

* ❌ Impossibilidade de treinar os modelos devido a limitações computacionais insuperáveis
* ❌ Frameworks de ML apresentam bugs críticos que impedem implementação

**NG3 – Falta de Significância:**

* ❌ Todos os modelos apresentam desempenho equivalente a baseline trivial (acurácia < 60%)
* ❌ Resultados não apresentam qualquer diferença estatisticamente significativa entre modelos
* ❌ Experimento não responde a nenhuma das questões de pesquisa formuladas

**NG4 – Problemas de Cronograma Irrecuperáveis:**

* ❌ Atraso > 4 semanas no cronograma sem possibilidade de recuperação
* ❌ Impossibilidade de conclusão dentro do prazo acadêmico institucional

**NG5 – Questões Éticas ou Legais:**

* ❌ Identificação de problemas éticos não previstos (privacidade, uso de dados)
* ❌ Restrições legais impedem uso do dataset ou publicação de resultados

---

#### **Procedimento de Avaliação de Sucesso:**

1. **Checkpoints de Progresso:**
   * Revisão quinzenal com orientador verificando atendimento progressivo aos critérios
   * Milestones intermediários alinhados aos critérios essenciais

2. **Avaliação Final:**
   * Checklist formal de atendimento aos critérios Must-Have antes da entrega
   * Autoavaliação documentada dos critérios Should-Have
   * Discussão de critérios de descontinuação na seção de limitações

3. **Decisões Go/No-Go:**
   * Reunião formal com orientador se qualquer critério No-Go for identificado
   * Plano de contingência deve ser ativado se houver risco de não atender critérios Must-Have
   * Possibilidade de ajuste de escopo se necessário, mantendo rigor científico

---

