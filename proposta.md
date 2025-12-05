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
# 2. Contexto e Problema

## 2.1 Descrição do Problema / Oportunidade

A pecuária extensiva brasileira ocupa cerca de 159 milhões de hectares e depende diretamente da qualidade das pastagens, que influenciam produtividade, sustentabilidade e custos operacionais. O monitoramento tradicional, baseado em inspeção visual, apresenta detecção tardia de anomalias como degradação, pragas, solo exposto ou plantas invasoras, além de exigir altos custos pelo deslocamento de equipes técnicas, apresentar subjetividade nas avaliações e oferecer cobertura limitada em grandes áreas. Diante disso, tecnologias com drones e visão computacional permitem automatizar o monitoramento, mas ainda não há clareza científica sobre quais modelos de Machine Learning são mais eficazes nesse contexto, especialmente considerando a heterogeneidade e variabilidade das pastagens brasileiras.

## 2.2 Contexto Organizacional e Técnico

O estudo ocorre em um contexto acadêmico, com foco em pesquisa aplicada em Engenharia de Software voltada à agricultura de precisão. O ambiente técnico considera o uso de Python, frameworks como PyTorch ou TensorFlow para Machine Learning, OpenCV para processamento de imagens e ferramentas como scikit-learn, SciPy e pandas para análises estatísticas. O dataset consiste em imagens aéreas RGB de pastagens brasileiras capturadas por drones, com resolução adequada para identificar anomalias e rotuladas previamente por especialistas, além de apresentar condições variadas de iluminação.

## 2.3 Trabalhos e Evidências Prévias (Internos e Externos)

Na literatura, há ampla utilização de CNNs em agricultura, especialmente em tarefas de detecção de doenças em plantas, com modelos como ResNet, VGG, YOLO e U-Net demonstrando alto desempenho; além disso, o uso de transfer learning reduz a necessidade de grandes volumes de dados. Em sensoriamento remoto, drones vêm sendo utilizados no monitoramento de safras, sendo comum o uso de NDVI, e imagens multiespectrais oferecem maior precisão, ainda que com custo elevado. Estudos comparativos apontam que não há um modelo universalmente superior, pois o desempenho depende do contexto. Internamente, o projeto ainda não possui experimentos prévios, mas conta com conhecimento intermediário do pesquisador em Python e ML e orientação especializada. Lacunas incluem a escassez de estudos focados em pastagens brasileiras, ausência de datasets públicos rotulados e necessidade de avaliação empírica rigorosa.

## 2.4 Referencial Teórico e Empírico Essencial

Machine Learning e Deep Learning fornecem fundamentos como aprendizagem supervisionada e CNNs, além do uso de transfer learning. A detecção de anomalias envolve classificação, segmentação e métodos baseados em reconstrução. A avaliação de modelos utiliza métricas como acurácia, precisão, recall e F1-score, bem como validação cruzada e análise de trade-offs. Os modelos candidatos incluem ResNet, MobileNet, EfficientNet, U-Net e YOLO. A literatura sugere que modelos mais profundos tendem a ter maior acurácia, transfer learning reduz custo computacional e modelos leves apresentam relação custo–benefício superior. Benchmarks indicam que CNNs têm desempenho consistente, com F1-scores entre 0,75 e 0,95, e datasets de 1000–5000 imagens são suficientes para transfer learning.


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

**O3 – Determinar a viabilidade prática de implementação**  
Avaliar a relação custo-benefício entre desempenho e requisitos técnicos de cada modelo, considerando cenários reais de aplicação (dispositivos embarcados em drones, processamento em nuvem, sistemas edge computing).

**O4 – Identificar padrões de erro e limitações**  
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


#### Relacionadas ao Objetivo O3 (Viabilidade Prática):

**Q3.1:** Qual modelo oferece a melhor relação entre acurácia e custo computacional para implementação em dispositivos com recursos limitados?

**Q3.2:** Qual modelo apresenta melhor custo-benefício considerando tempo de desenvolvimento, treinamento e implantação?

**Q3.3:** Quais modelos são tecnicamente viáveis para processamento embarcado em drones de médio porte?

#### Relacionadas ao Objetivo O4 (Padrões de Erro):

**Q4.1:** Quais tipos de anomalias geram mais erros de classificação em cada modelo?

**Q4.2:** Qual modelo produz menor taxa de falsos positivos em áreas de pastagem saudável?

**Q4.3:** Qual modelo apresenta melhor desempenho na detecção de anomalias em estágios iniciais de degradação?

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
| **O3 – Determinar a viabilidade prática** | Q3.1: Qual modelo oferece melhor relação acurácia/custo? | M17: Índice de Eficiência (F1-Score / Tempo Inferência)<br>M13: Número de Parâmetros |
| | Q3.2: Qual modelo apresenta melhor custo-benefício total? | M8: Tempo Total de Treinamento<br>M18: Custo Computacional Estimado |
| | Q3.3: Quais modelos são viáveis para processamento embarcado? | M12: Consumo Médio de Memória GPU<br>M11: Taxa de Processamento (FPS) |
| **O4 – Identificar padrões de erro e limitações** | Q4.1: Quais tipos de anomalias geram mais erros? | M19: Taxa de Erro por Classe<br>M2: Acurácia por Classe |
| | Q4.2: Qual modelo produz menor taxa de falsos positivos? | M20: Taxa de Falsos Positivos (FPR)<br>M4: Precisão Média |
| | Q4.3: Qual modelo detecta melhor anomalias iniciais? | M5: Recall Médio (para classe "degradação leve")<br>M3: F1-Score (para classe "degradação leve") |

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

## 4.2 Contexto do Estudo

O estudo ocorre em uma universidade, no âmbito de um TCC experimental com relevância prática e caráter científico. A criticidade é média-alta, com potencial impacto econômico, embora sem riscos diretos à vida. O pesquisador possui experiência intermediária em ML e Python; o orientador traz expertise em experimentação; e especialistas agrônomos contribuem com a validação das rotulações. Os recursos são limitados, com dependência de ferramentas gratuitas e infraestrutura simples.

## 4.3 Premissas

As premissas incluem a existência de um dataset adequado com pelo menos 2000 imagens, rotulações confiáveis com concordância elevada, acesso a GPU durante todo o experimento, estabilidade das bibliotecas usadas, tempo disponível de cerca de 20 horas semanais por quatro meses, orientação regular, conhecimento técnico suficiente do pesquisador, capacidade de controlar variações com técnicas como data augmentation, poder estatístico suficiente e representatividade das imagens capturadas.

## 4.4 Restrições

As restrições envolvem limitações de tempo, como o prazo máximo de quatro meses e sessões limitadas do Google Colab; restrições orçamentárias que impedem o uso de hardware ou serviços pagos; limitações técnicas devido à capacidade de GPU e impossibilidade de testar hardware especializado; limitações de dados, já que não é possível coletar novas imagens; limitações de escopo, pois o estudo ocorre apenas em ambiente acadêmico; restrições organizacionais, incluindo requisitos institucionais; e restrições de acesso a propriedades rurais e especialistas.

## 4.5 Limitações Previstas

As limitações previstas incluem a possibilidade de resultados com validade externa restrita devido a fatores geográficos, como dataset limitado a biomas brasileiros; representatividade insuficiente do dataset, que pode conter vieses ou distribuição desbalanceada; limitações tecnológicas relacionadas ao hardware usado e ao uso de modelos pré-treinados; limitações metodológicas decorrentes de validação restrita e ausência de avaliação com usuários reais; limitações temporais e risco de obsolescência; limitações para generalização em ambientes produtivos; e limitações de validação devido a rotulações humanas e ausência de ground-truth absoluto.


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

## 5.3 Impactos Potenciais no Processo / Produto

Durante a execução, o estudo pode gerar competências importantes para o pesquisador, conhecimento científico relevante, oportunidades de colaboração e visibilidade acadêmica, embora apresente desafios como carga de trabalho elevada, riscos técnicos e dependência de recursos. Após a conclusão, o experimento pode gerar contribuições científicas, servir como prova de conceito para tecnologias futuras, orientar desenvolvedores, apoiar a adoção de soluções no setor agropecuário e fomentar sustentabilidade, ao mesmo tempo em que envolve riscos como interpretações indevidas, expectativas irreais ou rápida obsolescência tecnológica.


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

## 6.2 Critérios de Sucesso Globais (Go / No-Go)

Os critérios essenciais determinam que todos os modelos devem ser treinados e avaliados com sucesso, seguindo o protocolo experimental completo; todas as métricas essenciais devem ser coletadas; análises estatísticas adequadas devem ser conduzidas; o relatório final deve seguir padrões acadêmicos e permitir replicação; e as conclusões precisam estar fundamentadas em evidências e responder às questões de pesquisa. Os critérios desejáveis incluem a obtenção de diferenças estatisticamente significativas entre modelos, robustez comprovada em condições adversas, viabilidade prática para dispositivos com poucos recursos, potencial de publicação científica e disponibilização do código em repositório público. Os critérios de descontinuação envolvem problemas críticos de dados, inviabilidade técnica, falta de significância estatística, atrasos irrecuperáveis e questões éticas.

---
### 6.3 Critérios de Parada Antecipada (Pré-execução)

Antes do início da execução do experimento, alguns critérios podem justificar seu adiamento ou cancelamento. O primeiro deles refere-se à indisponibilidade de recursos computacionais essenciais, como GPU, memória ou ambiente de execução adequado. Se o pesquisador não tiver acesso mínimo aos recursos necessários, a continuidade do experimento torna-se inviável. Outro critério importante está relacionado à integridade do dataset: caso seja identificada baixa qualidade das imagens, rotulações inconsistentes, distribuição extremamente desbalanceada entre classes ou uma quantidade insuficiente de dados, o experimento deve ser interrompido até que tais problemas sejam tratados.

Também deve ser considerado o aspecto ético. Se houver qualquer dúvida sobre a necessidade de aprovação em comitê de ética, ou se algum impedimento legal impedir o uso do dataset, a execução deve ser suspensa. Além disso, mudanças no contexto acadêmico — como alterações no escopo do TCC, no cronograma institucional ou na orientação — podem demandar replanejamento antes do início. Por fim, se testes preliminares demonstrarem que um ou mais modelos não conseguem ser treinados de maneira estável ou dentro do tempo disponível, o experimento deve ser adiado para ajustes ou redefinição metodológica.

# 7. Modelo Conceitual e Hipóteses

## 7.1 Modelo Conceitual do Experimento

O modelo conceitual adotado neste trabalho considera que diferentes arquiteturas de Machine Learning influenciam diretamente o desempenho na detecção de anomalias em pastagens. Modelos mais profundos tendem a apresentar maior capacidade de extração de características, o que pode resultar em maior acurácia e robustez, ainda que com maior demanda computacional. Em contrapartida, modelos mais leves normalmente produzem inferências mais rápidas, porém com menor precisão. Assim, parte-se da premissa de que a escolha da arquitetura impacta tanto a qualidade das predições quanto a viabilidade prática do uso em cenários reais. A relação entre modelo, desempenho e custo computacional constitui o núcleo conceitual deste estudo.

## 7.2 Hipóteses Formais (H0 e H1)

As hipóteses experimentais foram estabelecidas com base no modelo conceitual. A primeira refere-se à acurácia: a hipótese nula estabelece que não há diferença significativa entre os modelos, enquanto a hipótese alternativa assume que ao menos um deles apresenta desempenho superior. Em relação ao F1-score, a hipótese nula também pressupõe igualdade entre os modelos, mentre a alternativa indica diferença significativa. Para o tempo de inferência, presume-se inicialmente que não há variação relevante entre os modelos, mas a hipótese alternativa sugere que modelos mais leves apresentam desempenho significativamente mais rápido. Quanto à robustez, a hipótese nula assume que todos os modelos possuem variabilidade semelhante em condições adversas, enquanto a hipótese alternativa sugere que pelo menos um deles é mais estável. Finalmente, em relação à eficiência custo-benefício, a hipótese alternativa pressupõe a superioridade de algum modelo em termos de equilíbrio entre desempenho e consumo de recursos.

## 7.3 Nível de Significância e Considerações de Poder

Este estudo adota o nível de significância α = 0,05, conforme prática comum em experimentação científica. Espera-se um poder estatístico mínimo de 80%, o que requer um número suficientemente grande de imagens no conjunto experimental. A validação cruzada contribui para aumentar a confiabilidade dos resultados e diminuir o impacto de partições específicas do dataset. Caso seja identificado desbalanceamento entre classes ou alta variabilidade nas métricas, técnicas auxiliares como reamostragem e análise estratificada poderão ser empregadas.

# 8. Variáveis, Fatores, Tratamentos e Objetos de Estudo

## 8.1 Objetos de Estudo

Os objetos de estudo deste experimento são as imagens aéreas de pastagens capturadas por drones, previamente rotuladas quanto à presença ou ausência de diferentes tipos de anomalias, e os modelos de Machine Learning aplicados sobre esse conjunto de dados. A comparação entre as predições geradas por cada modelo permitirá avaliar sua eficácia, eficiência e robustez diante do problema de detecção automática de anomalias em pastagens.

## 8.2 Sujeitos / Participantes

O experimento não envolve participantes humanos como sujeitos experimentais. A intervenção humana limita-se ao pesquisador, responsável pela execução técnica, e aos especialistas agrônomos que atuaram no processo de rotulação das imagens. Não há interação direta com usuários finais durante a execução deste estudo, o que reduz riscos éticos e simplifica o escopo da pesquisa.

---

## 8.3 Variáveis Independentes, Dependentes, de Controle e de Confusão

Nesta seção, são descritas as variáveis consideradas no experimento. A variável independente principal é a arquitetura do modelo de Machine Learning. As variáveis dependentes representam as métricas de desempenho avaliadas. As variáveis de controle são fatores mantidos constantes para garantir validade interna. Por fim, as variáveis de confusão são fatores externos que podem influenciar os resultados e devem ser monitorados.

A seguir, apresenta-se uma tabela consolidada com as principais variáveis e suas descrições:

### **Tabela 1 – Variáveis do Experimento**

| Tipo de Variável | Nome | Descrição |
|------------------|------|-----------|
| Independente | Arquitetura do modelo | Tipo de modelo de ML utilizado (ResNet, MobileNet, EfficientNet, U-Net, YOLO). |
| Dependente | Acurácia | Percentual de predições corretas realizadas pelo modelo. |
| Dependente | F1-Score | Média harmônica entre precisão e recall, medindo equilíbrio entre acertos positivos. |
| Dependente | Precisão | Proporção de verdadeiros positivos entre todas as detecções positivas. |
| Dependente | Recall | Capacidade de identificar corretamente todos os casos positivos. |
| Dependente | Tempo de inferência | Tempo médio necessário para gerar uma predição para uma imagem. |
| Dependente | Tempo de treinamento | Duração total do processo de treinamento do modelo. |
| Dependente | Consumo de memória | Quantidade de memória utilizada durante a inferência. |
| Dependente | FPS | Número de imagens processadas por segundo. |
| Controle | Hardware utilizado | GPU, CPU, memória e ambiente de execução padronizado para todos os modelos. |
| Controle | Pré-processamento das imagens | Procedimentos fixos de normalização, redimensionamento e augmentations. |
| Controle | Divisão do dataset | Dataset idêntico para todos os modelos, usando mesma estratégia de K-fold. |
| Controle | Parâmetros de treinamento | Número de épocas, taxa de aprendizado e batch size fixados. |
| Confusão | Desbalanceamento de classes | Diferenças na quantidade de exemplos por classe podem influenciar o treinamento. |
| Confusão | Iluminação e ruído | Variações naturais nas condições de captura das imagens aéreas. |
| Confusão | Erros de rotulação | Possíveis inconsistências no processo de anotação humana. |
| Confusão | Variação sazonal da pastagem | Diferenças naturais entre épocas do ano podem alterar padrões visuais. |

---

## 8.4 Fatores, Tratamentos e Combinações

O principal fator manipulado no experimento é a arquitetura do modelo de Machine Learning. Cada modelo representa um tratamento distinto, com características próprias de profundidade, número de parâmetros, velocidade de inferência e tipo de tarefa predominante (classificação, detecção ou segmentação).

A seguir, apresenta-se a tabela consolidada dos fatores, tratamentos e suas combinações experimentais.

### **Tabela 2 – Fatores, Tratamentos e Combinações**

| Fator | Tratamento | Descrição | Tipo de Tarefa |
|-------|------------|-----------|----------------|
| Arquitetura do Modelo | T1 – ResNet-50 | Modelo profundo com camadas residuais, alta capacidade de extração de padrões. | Classificação |
| Arquitetura do Modelo | T2 – MobileNetV2 | Modelo leve, otimizado para dispositivos móveis, com baixo custo computacional. | Classificação |
| Arquitetura do Modelo | T3 – EfficientNet-B0 | Modelo com equilíbrio entre profundidade e eficiência, escalonamento composto. | Classificação |
| Arquitetura do Modelo | T4 – U-Net | Modelo especializado em segmentação semântica, útil para delinear áreas anômalas. | Segmentação |
| Arquitetura do Modelo | T5 – YOLOv8 | Modelo de detecção rápida, capaz de localizar regiões anômalas em tempo real. | Detecção |

### **Combinações experimentais**

Como o experimento envolve apenas um fator (a arquitetura), as combinações correspondem diretamente aos cinco tratamentos definidos. A avaliação é realizada em cinco folds de validação cruzada, resultando em:

**5 modelos × 5 folds = 25 combinações experimentais executadas.**

Cada combinação consiste em:

- um modelo específico;  
- treinado em um fold específico;  
- avaliado no conjunto complementar daquele fold;  
- produzindo métricas completas para comparação.


## 8.5 Variáveis Dependentes

As tabelas acima estruturam os elementos fundamentais do experimento. No entanto, sua interpretação dentro do texto é igualmente importante. A escolha da arquitetura como fator principal deve-se ao objetivo central do estudo, que é comparar a efetividade de modelos distintos no contexto de detecção de anomalias em pastagens. Cada arquitetura representa um paradigma diferente — redes profundas, modelos leves, segmentadores ou detectores — e, portanto, oferece impactos distintos sobre as métricas dependentes.

As variáveis dependentes foram selecionadas para permitir uma avaliação multidimensional: não apenas se o modelo acerta, mas como ele acerta, quanto custa esse acerto em termos de tempo e recursos e quão estável é esse desempenho em múltiplas execuções. As variáveis de controle asseguram que apenas o fator arquitetural seja responsável pelas diferenças observadas, enquanto as variáveis de confusão representam ameaças que demandam monitoramento e discussão na análise final.


As variáveis dependentes são as métricas de desempenho obtidas durante o experimento. Entre elas destacam-se acurácia, F1-score, precisão, recall, tempo de inferência, tempo total de treinamento, consumo de memória e taxa de processamento em frames por segundo. Todas essas medidas permitem avaliar de forma abrangente a performance de cada modelo.

## 8.6 Variáveis de Controle

Diversos fatores serão mantidos fixos para assegurar validade interna, incluindo hardware utilizado, ambiente de execução, pré-processamento das imagens, técnica de data augmentation, divisão do dataset e parâmetros de treinamento. O uso de um conjunto de dados comum a todos os modelos garante condições equivalentes de avaliação.

## 8.7 Variáveis de Confusão

Alguns fatores podem influenciar os resultados e serão cuidadosamente monitorados, como desbalanceamento entre classes, diferenças de iluminação, ruídos naturais introduzidos pela captura aérea, variações sazonais e eventuais erros de rotulação humana. Esses fatores serão discutidos tanto na análise quanto nas ameaças à validade.

# 9. Desenho Experimental

## 9.1 Tipo de Desenho Experimental

O estudo utilizará um desenho completamente randomizado com validação cruzada K-fold (K = 5). Esse método é adequado a experimentos computacionais, permitindo comparar diretamente diferentes tratamentos sob condições equivalentes e reduzindo vieses de amostragem.

## 9.2 Randomização e Alocação

A randomização será aplicada na divisão estratificada do dataset, garantindo que cada fold mantenha proporções semelhantes entre as classes. A ordem das imagens durante o treinamento será embaralhada automaticamente, e a ordem de execução dos modelos poderá ser alternada para reduzir interferências externas, como variações temporais de processamento da GPU.

## 9.3 Balanceamento e Contrabalanço

O balanceamento entre as classes será realizado por meio da estratificação do dataset. Para evitar efeitos de ordem, serão aplicadas técnicas de contrabalanço, como embaralhamento das sequências de imagens em cada época de treinamento e uso uniforme de data augmentation entre todos os modelos. Tais práticas asseguram igualdade de condições entre os tratamentos.

## 9.4 Número de Grupos e Sessões

O experimento contará com cinco grupos principais, cada um correspondente a um dos modelos avaliados. Cada grupo será submetido a cinco execuções distintas, uma para cada fold da validação cruzada, totalizando vinte e cinco sessões experimentais. Esse número é suficiente para estimar a variabilidade e garantir rigor estatístico nas comparações.

# 10. População, Sujeitos e Amostragem

## 10.1 População-alvo
A população-alvo deste experimento corresponde ao conjunto de modelos de Machine Learning aplicáveis ao problema de detecção automática de anomalias em pastagens utilizando imagens aéreas capturadas por drones. Embora o estudo não envolva participantes humanos, considera-se que a “população real” representada é composta por arquiteturas modernas de visão computacional — como CNNs profundas, modelos leves e arquiteturas híbridas — que potencialmente poderiam ser utilizadas em sistemas reais de monitoramento agropecuário. Assim, o experimento busca representar o comportamento esperado dessas arquiteturas diante de um dataset realista de pastagens brasileiras.

## 10.2 Critérios de inclusão de sujeitos
Os “sujeitos” do experimento são os modelos selecionados. Para serem incluídos, precisam atender a critérios mínimos: possuir implementação pública estável; ser compatíveis com frameworks amplamente utilizados (PyTorch ou TensorFlow); permitir ajustes para classificação, detecção ou segmentação de imagens; serem citados na literatura científica como métodos consolidados; e serem executáveis no hardware disponível. Além disso, cada modelo deve ser tecnicamente adequado ao problema e representar uma abordagem distinta para garantir diversidade metodológica no estudo.

## 10.3 Critérios de exclusão de sujeitos
Modelos proprietários ou que exigem licenças pagas são excluídos. Arquiteturas experimentais sem documentação adequada ou com comportamento imprevisível também não são consideradas. Modelos muito pesados, que excedam a capacidade computacional disponível, são automaticamente descartados. Métodos não relacionados ao processamento de imagens, como modelos textuais ou redes especializadas em áudio, também são excluídos por não atenderem às necessidades da tarefa.

## 10.4 Tamanho da amostra planejado (por grupo)
A amostra é definida pelos modelos e pelas execuções realizadas. Cada arquitetura será avaliada utilizando validação cruzada K-fold (K = 5), o que resulta em cinco execuções por modelo. Assim, cada grupo possui cinco observações independentes, totalizando 25 execuções experimentais. Esse tamanho de amostra é suficiente para estimar variações de desempenho e suportar a aplicação dos testes estatísticos definidos previamente.

## 10.5 Método de seleção / recrutamento
A seleção dos modelos segue uma estratégia de amostragem intencional, considerando relevância científica, maturidade tecnológica, disponibilidade de implementações confiáveis e alinhamento com o objetivo da pesquisa. Assim, os modelos foram “recrutados” com base em sua representatividade dentro do domínio de visão computacional aplicada à agricultura. O processo considera tanto desempenho teórico quanto viabilidade de execução no ambiente experimental.

## 10.6 Treinamento e preparação dos sujeitos
Cada modelo receberá preparação uniforme, incluindo carregamento de pesos pré-treinados, normalização das imagens, ajustes das camadas finais e definição de hiperparâmetros padronizados. O objetivo é garantir equidade entre os tratamentos. Guias de configuração, scripts de automação e instruções técnicas serão utilizados para assegurar consistência e reduzir vieses decorrentes de configurações distintas ou inadequadas.

---

# 11. Instrumentação e Protocolo Operacional

## 11.1 Instrumentos de coleta (questionários, logs, planilhas, etc.)
A coleta de dados será realizada de forma automatizada. Os modelos gerarão logs contendo métricas de treinamento e validação, registrados em arquivos de texto ou JSON. Scripts específicos computarão tempo de inferência, consumo de memória e demais métricas de interesse. Planilhas no formato CSV consolidarão os resultados de cada fold, permitindo importação posterior em ferramentas de análise estatística. Ferramentas como TensorBoard poderão ser utilizadas para visualização dinâmica dos gráficos de desempenho durante o processo.

## 11.2 Materiais de suporte (instruções, guias)
O experimento contará com um conjunto de materiais de suporte, incluindo um guia detalhado de execução, instruções para configuração do ambiente virtual, documentação da estrutura dos scripts e uma descrição completa dos parâmetros utilizados. Esses materiais garantem reprodutibilidade e servem como referência para execução consistente do pipeline experimental. Slide decks poderão ser usados para apresentação do progresso ao orientador e para organizar revisões periódicas.

## 11.3 Procedimento experimental (protocolo – visão passo a passo)
O protocolo inicia-se com a preparação do ambiente, instalação das dependências e validação das versões das bibliotecas. Em seguida, o dataset é carregado e dividido em cinco folds estratificados. Cada modelo é configurado com parâmetros previamente definidos e submetido ao treinamento no primeiro fold. Após o treinamento, o modelo é avaliado no conjunto correspondente, e todas as métricas são registradas.

Este processo se repete para todos os folds e para todos os modelos, respeitando a mesma estrutura de execução. Finalizadas as execuções, os resultados são consolidados em planilhas para inspeção inicial. Por fim, realiza-se a análise estatística, geração de gráficos, interpretação dos resultados e documentação das conclusões.

## 11.4 Plano de piloto (se haverá piloto, escopo e critérios de ajuste)
Um piloto será realizado utilizando apenas uma pequena fração do dataset e um modelo leve (MobileNetV2). O objetivo é validar o funcionamento do pipeline e a compatibilidade das bibliotecas. Caso sejam identificados problemas como falhas de execução, incompatibilidade de versões ou tempos excessivos de processamento, ajustes serão feitos antes da execução completa. O protocolo só será iniciado integralmente após o piloto demonstrar estabilidade e reprodutibilidade.

---

# 12. Plano de Análise de Dados (Pré-execução)

## 12.1 Estratégia geral de análise
A análise será conduzida de forma sistemática, observando tanto métricas de desempenho quanto métricas de eficiência computacional. Para cada modelo, serão calculadas médias e desvios-padrão das métricas ao longo dos folds, permitindo identificar padrões e tendências. Gráficos comparativos facilitarão a visualização das diferenças entre os modelos. Os resultados serão utilizados diretamente para responder cada questão de pesquisa, garantindo alinhamento entre coleta, análise e objetivos do estudo.

## 12.2 Métodos estatísticos planejados
A análise estatística incluirá testes de hipótese adequados para medidas repetidas, como ANOVA de medidas repetidas ou, se necessário, a versão não paramétrica (teste de Friedman). Também poderão ser aplicados testes post-hoc, como Tukey ou Nemenyi, caso diferenças significativas sejam identificadas. Testes de normalidade serão realizados previamente para orientar a escolha dos métodos. Adicionalmente, intervalos de confiança serão calculados para fornecer maior robustez interpretativa.

## 12.3 Tratamento de dados faltantes e outliers
Dados faltantes decorrentes de falhas de execução serão registrados, mas não serão interpolados. Caso a ausência comprometa alguma análise estatística, a execução correspondente poderá ser repetida. Valores considerados outliers serão investigados individualmente; se forem resultado de erros técnicos, serão descartados. Caso representem variabilidade natural dos modelos, serão mantidos, desde que não comprometam a consistência das conclusões.

## 12.4 Plano de análise para dados qualitativos (se houver)
Embora predominem dados quantitativos, algumas observações qualitativas poderão ser analisadas, especialmente no que diz respeito aos padrões de erro, comportamentos inesperados dos modelos e dificuldades identificadas durante o treinamento. A análise qualitativa será conduzida por meio de categorização simples, permitindo identificar classes mais problemáticas ou limitações específicas de cada arquitetura. Essas informações complementam os resultados numéricos e enriquecem a discussão final.

## 13. Avaliação de validade (ameaças e mitigação)

### 13.1 Validade de conclusão
As principais ameaças envolvem baixo poder estatístico, violação de suposições dos testes e erros de medida decorrentes de coleta inadequada. Esses fatores podem comprometer a robustez das conclusões sobre o desempenho comparativo dos modelos.  
Para mitigar, o estudo adota K-fold adequado, testes de normalidade com uso de métodos não paramétricos quando necessário e padronização dos scripts de coleta de métricas, reduzindo inconsistências.

### 13.2 Validade interna
Há risco de efeitos observados serem influenciados por variações externas, como ordem de execução, carga da GPU ou diferenças de pré-processamento. Além disso, viés de seleção pode surgir se a divisão dos folds não preservar a distribuição das classes.  
A mitigação ocorre com padronização do pipeline, seeds fixas, alternância na ordem de execução, estratificação dos folds e registro detalhado de metadados de execução.

### 13.3 Validade de constructo
Ameaças surgem quando as métricas não representam plenamente o conceito de “qualidade de detecção” ou quando há ambiguidade de rotulação, distorcendo interpretações.  
Para contornar, são utilizadas métricas complementares, validação das rotulações por especialistas e diretrizes claras para anotação das classes, reduzindo ambiguidades.

### 13.4 Validade externa
A generalização pode ser limitada por características específicas do dataset, como bioma, resolução e condições de captura, o que restringe a aplicabilidade dos resultados a outros cenários.  
A mitigação inclui documentar o domínio, testar pequenas variações de contexto e recomendar validações futuras com novos ambientes e sensores.

### 13.5 Resumo das ameaças e mitigação
As ameaças mais críticas envolvem baixo poder estatístico, erros de rotulação, variações operacionais e limitações de generalização. Elas são mitigadas com K-fold consistente, revisão de rotulações, padronização do ambiente e documentação explícita das condições do experimento.  
Essas ações reforçam a confiabilidade das conclusões e deixam claros os limites do estudo para replicações futuras.

---

## 14. Ética, privacidade e conformidade

### 14.1 Questões éticas
O estudo não envolve participantes humanos, mas requer atenção ao uso de especialistas na rotulação e ao tratamento de imagens que possam conter informações sensíveis.  
A mitigação consiste em transparência sobre objetivos, ausência de pressão sobre colaboradores e anonimização de quaisquer dados que identifiquem propriedades ou locais.

### 14.2 Consentimento informado
Os especialistas envolvidos receberão explicações sobre objetivos, riscos e uso dos dados, formalizando consentimento eletrônico simples.  
Esse registro garante conformidade institucional e oferece rastreabilidade para revisões e auditorias.

### 14.3 Privacidade e proteção de dados
Embora haja mínima coleta de dados pessoais, metadados sensíveis serão removidos ou pseudoanonimizados. Repositórios privados e controle de acesso serão utilizados para proteger o material.  
Os dados serão mantidos somente pelo período necessário à reprodutibilidade acadêmica.

### 14.4 Aprovações necessárias
Apesar de provavelmente dispensar aprovação de comitê de ética, recomenda-se confirmação institucional.  
Caso existam dados sensíveis ou intenção de divulgação pública, o jurídico ou DPO deverá ser consultado previamente.

---

## 15. Recursos, infraestrutura e orçamento

### 15.1 Recursos humanos e papéis
O pesquisador realiza a execução técnica e análise; o orientador atua como aprovador metodológico; especialistas contribuem com validação de rótulos.  
Essa distribuição cobre execução, supervisão e avaliação técnica.

### 15.2 Infraestrutura técnica necessária
O experimento requer ambiente Python com bibliotecas de visão computacional, GPU local ou na nuvem, repositório GitHub e ferramentas para versionamento e logs.  
Essa infraestrutura assegura rastreabilidade, desempenho e reprodutibilidade.

### 15.3 Materiais e insumos
São necessários o dataset rotulado, dependências computacionais, dispositivos de armazenamento e eventuais licenças auxiliares.  
Todos os materiais serão organizados para facilitar rastreamento e auditoria.

### 15.4 Orçamento e custos
Os custos envolvem horas de trabalho, possível uso de GPU paga e compensação para especialistas, quando aplicável.  
O financiamento pode ser pessoal ou institucional, mantendo o projeto economicamente viável.

---

## 16. Cronograma, marcos e riscos operacionais

### 16.1 Macrocronograma
O cronograma inclui finalização do plano, piloto, execução dos experimentos, análise estatística e consolidação dos resultados.  
Cada fase depende da validação da anterior, garantindo fluxo progressivo e controlado.

### 16.2 Dependências entre atividades
O treinamento completo depende do sucesso do piloto, e a publicação de resultados depende de validações ética e técnica.  
Essas dependências reduzem riscos de retrabalho e falhas estruturais.

### 16.3 Riscos operacionais
Os principais riscos são indisponibilidade de GPU, interrupções em serviços de nuvem e falhas no treinamento.  
As contingências incluem checkpoints frequentes, alternância de plataformas e ajustes de hiperparâmetros mais leves.

---

## 17. Governança do experimento

### 17.1 Papéis e responsabilidades formais
A governança define o orientador como aprovador final, o pesquisador como executor principal e os especialistas como revisores técnicos.  
Esse arranjo delimita autoridade e reduz ambiguidades de responsabilidade.

### 17.2 Ritos de acompanhamento pré-execução
Serão realizadas reuniões periódicas com o orientador e checkpoints formais antes do início da execução plena.  
Esses ritos permitem ajustes antecipados e validação contínua do plano.

### 17.3 Processo de controle de mudanças
Alterações serão registradas via issues e commits, com aprovação explícita do orientador para mudanças metodológicas.  
Isso garante rastreabilidade e mantém integridade do experimento.

---

## 18. Documentação e reprodutibilidade

### 18.1 Repositórios e convenções de nomeação
Scripts, dados e relatórios serão organizados em repositórios com convenções claras de pastas e nomes, reforçando clareza e navegação.  
Esse padrão contribui para padronização e facilidade de manutenção.

### 18.2 Templates e artefatos padrão
Serão utilizados templates para relatórios, scripts, arquivos de configuração e checklists, garantindo consistência documental.  
Esses artefatos evitam divergências de estilo e reduzem erros.

### 18.3 Plano de empacotamento para replicação futura
A replicação futura será facilitada com disponibilização de scripts, seeds, instruções de ambiente e documentação completa do processo.  
Dessa forma, a reprodução do estudo demanda mínima adaptação externa.

---

## 19. Plano de comunicação

### 19.1 Públicos e mensagens-chave pré-execução
O público inclui orientador, especialistas e coordenação, que receberão informações sobre escopo, datas e requisitos do experimento.  
Essa comunicação inicial alinha expectativas e reduz riscos de desencontro.

### 19.2 Canais e frequência de comunicação
Serão utilizadas reuniões periódicas, e-mails e comunicação assíncrona por plataformas como Slack ou Teams.  
Cada canal atende necessidades diferentes de urgência e formalidade.

### 19.3 Pontos de comunicação obrigatórios
Aprovação do plano, conclusão do piloto, alterações relevantes e entrega final exigem comunicações formais.  
Esses pontos garantem que todos os envolvidos estejam atualizados e sincronizados.

---

## 20. Critérios de prontidão (Definition of Ready)

### 20.1 Checklist de prontidão
Antes da execução devem estar concluídos: validação do plano, dataset verificado, piloto bem-sucedido, recursos técnicos disponíveis e conformidade ética assegurada.  
Esse checklist evita início prematuro e reduz riscos metodológicos.

### 20.2 Aprovações finais para iniciar a operação
O orientador fornecerá o aceite formal, e a equipe garantirá a disponibilidade técnica completa.  
Somente após essas confirmações o experimento poderá iniciar sua execução integral.




