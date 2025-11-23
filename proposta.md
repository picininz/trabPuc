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
