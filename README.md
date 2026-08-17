# SLD — Scientific Literature Decoder

**SLD — Scientific Literature Decoder** é uma aplicação em Python e Streamlit destinada à recuperação semântica, classificação supervisionada multilabel e extração conceitual por modelo de linguagem (LLM) sobre literatura científica.

O sistema foi estruturado metodologicamente para apoio a pesquisas acadêmicas e teses de doutorado, priorizando **rigor estatístico**, **reprodutibilidade**, **rastreabilidade por `run_id`** e **transparência dos parâmetros**.

---

## 📐 Arquitetura do Pipeline Metodológico

```text
PDF / ZIP com PDFs
       ↓
Extração e Conversão para Markdown (Prevenção de ZIP Slip, SHA-256)
       ↓
Sentence Transformer (E(Pi) em R^d, modelo padrão: nomic-embed-text)
       ↓
Similaridade Semântica por Cosseno (Recuperação High Recall)
       ↓
Regressão Logística Multilabel (One-vs-Rest, Regra da Classe 0, Group Split por Doc)
       ↓
Classificação Conceitual (Probabilidades P(y_k|X_i), Thresholds θ_k)
       ↓
Corpus Final (Relevant_i = Or P(y_k|X_i) >= θ_k)
       ↓
Gemma 3 1B via Ollama Local (Extração Estruturada JSON, Validação de Evidência)
       ↓
Relatório Metodológico Completo (run_id, Equações LaTeX, Funil de Dados)
```

> [!IMPORTANT]
> **Papel Restrito do LLM:** O LLM **NÃO** participa da filtragem inicial dos milhares de parágrafos. Ele atua **EXCLUSIVAMENTE** sobre o `CORPUS FINAL` previamente filtrado pela busca semântica e classificado pela Regressão Logística Supervisionada ($C_{\text{final}} = \{ P_i \mid \text{Relevant}(P_i) = 1 \}$).

---

## 🏛️ Estrutura das 8 Abas Streamlit

1. **1. ETL — Extração, Transformação e Carregamento:** Coleta de PDFs/ZIPs (Extração), remoção de referências e limpeza (Transformação), salvamento de um **arquivo `.md` individual por artigo** no repositório de corpus local (Carregamento) e medição da porcentagem de redução de volume armazenado.
2. **2. Embeddings Vetoriais:** Leitura dos arquivos `.md` individuais, fragmentação por parágrafos concisos (`segment_markdown_paragraphs` max 500 caracteres) e cálculo dos Sentence Embeddings vetoriais via `nomic-embed-text` (ou `all-MiniLM-L6-v2`).
3. **3. Análise Exploratória do Corpus:** Painel bibliométrico, ranking de termos mais frequentes, WordCloud interativa, Matriz de Co-ocorrência de Termos $C_{ij} = \sum_d I(i \in d)I(j \in d)$ e Grafo de rede interativo.
4. **4. Similaridade Semântica:** Recuperação semântica baseada em múltiplas sentenças-âncora, cálculo de Similaridade do Cosseno $S(P_i, Q)$, distribuição de scores e indicador de Taxa de Retenção.
5. **5. Treinamento Supervisionado e Anotação:** Ambiente genérico para qualquer conceito de pesquisa acadêmica, reorganizado em 7 sub-abas:
   - **📊 Visão Geral:** Métricas do acervo, legenda permanente das classes 0-5 e distribuição por classe.
   - **✍️ Anotar no SLD:** Ferramenta interna de anotação sequencial sem viés prévio.
   - **📤 Exportar para Anotação:** Geração de conjuntos em Markdown (`.md`), `.csv` e `.jsonl` com amostragem aleatória, por faixa de similaridade, por incerteza e modo cego.
   - **📥 Importar Anotações:** Upload de arquivos com validação de versão (`sld_annotation_format: 1`), integridade por hash SHA-256 e pré-visualização tabular.
   - **🏆 Gold Standard:** Painel de consolidação e adjudicação de conflitos entre múltiplos anotadores.
   - **🧠 Treinar Modelo:** Treinamento da Regressão Logística Multilabel sobre dados válidos do Gold Standard.
   - **📈 Avaliação e Concordância:** Métricas do modelo (PR/F1) e cálculo do **Cohen's Kappa ($\kappa$)** quando houver múltiplos anotadores.
6. **6. Classificação Conceitual:** Predição de probabilidades por classe $P(y_k \mid X_i)$, aplicação dos thresholds de decisão $\theta_1..\theta_5$, análise da matriz de co-ocorrência entre classes conceituais $C_{ij} = \sum_p y_{pi}y_{pj}$ e heatmap.
7. **7. Corpus Final e Análise LLM:** Execução estritamente local de modelos de linguagem via **Ollama** selecionados dinamicamente pelo usuário, controle de taxa por requisições por minuto (Rate Limiter RPM), saída JSON em Schema Canônico, métricas de validação de evidência textual (EVR e PVR), checkpointing incremental (`llm_results.jsonl`), cache determinístico por hash e exportação do Corpus Refinado (`.csv` e `.md`).
8. **8. Relatório Metodológico:** Geração automática do relatório completo de execução para a pasta `output/<run_id>/reports/` contendo identificação de hardware/OS, tabela do funil de dados, parâmetros do modelo LLM selecionado, estatísticas de RPM, equações LaTeX renderizadas e exportação em Markdown, JSON e CSV.

---

## 🦙 Configuração do Ollama e Modelos LLM Locais

Para utilizar a etapa final de extração conceitual estruturada, o servidor **Ollama** deve estar em execução localmente no seu computador.

### 1. Baixar um Modelo Recomendado (ex: Qwen 2.5 7B ou Llama 3.1 8B)

```bash
ollama pull qwen2.5:7b
```

### 2. Testar o Modelo via Terminal (Opcional)

```bash
ollama run gemma3:1b
```

---

## 🚀 Instalação e Execução

### 1. Criar e Ativar o Ambiente Virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Instalar as Dependências

```bash
pip install -r requirements.txt
```

### 3. Executar a Aplicação Streamlit

```bash
streamlit run app.py
```

Acesse a interface no navegador em `http://localhost:8501`.

---

## 🧪 Execução dos Testes Automatizados

Para executar a suíte completa de **31 testes unitários automatizados** (incluindo testes de LLM, validação de evidência, cache, checkpointing, anti-Data Leakage e Regra da Classe 0):

```bash
python3 -m pytest tests/ -v
```

---

## 📂 Estrutura do Diretório de Saída (`output/<run_id>/`)

Cada execução gera uma pasta isolada garantindo total reprodutibilidade:

```text
output/run_2026_08_11_001/
├── config/             # run_config.json e environment.json
├── markdown/           # Arquivos .md individuais
├── metadata/           # Parquet de metadados
├── paragraphs/         # Parquet de parágrafos extraídos
├── embeddings/         # Matriz .npy e índice vetorial
├── semantic_search/    # Resultados da busca semântica por cosseno
├── annotations/        # Gold Standard de anotações humanas (.jsonl)
├── models/             # Regressão Logística (.joblib), thresholds.json e model_metadata.json
├── classifications/    # Predições conceituais
├── corpus_final/       # Parquet, CSV e JSONL do Corpus Final
├── llm/                # llm_results.jsonl, llm_results.parquet, llm_statistics.json, llm_errors.csv
├── reports/            # methodology_report.md, statistics.json, pipeline_counts.csv
└── logs/               # Log de execução da aplicação
```

---

## 🏛️ Interface e Documentação Metodológica

A interface do **SLD** foi refatorada e desenhada estritamente como um **software científico para análise sistemática de literatura**, priorizando sobriedade visual, consistência acadêmica e total transparência metodológica para apresentação em bancas de doutorado e artigos.

### Padrão Visual e Estrutura das Abas
1. **Cabeçalho Institucional:** Apresentação limpa com o `logo.png` e identificação oficial da aplicação.
2. **"Sobre esta Etapa":** Descrição metodológica sintética (2 a 4 linhas) acompanhada por um expander com a fundamentação matemática, objetivos e interpretações acadêmicas.
3. **Indicadores do Funil:** Apresentação transparente das contagens de entrada ($N_{\text{in}}$), saída ($N_{\text{out}}$), removidos e taxa de retenção (%) com denominadores visíveis.
4. **Representações Matemáticas com Legendas:** Todas as equações estatísticas (Similaridade do Cosseno, Limiar Semântico, Regressão Logística Multilabel, Métricas PR/F1, Funil de Retenção) são renderizadas via LaTeX acompanhadas por **legendas obrigatórias dos símbolos** e caixas de orientação de interpretação.
5. **Divisão de Parâmetros:** Configurações separadas entre **Parâmetros Metodológicos** (com impacto direto na pesquisa) e **Parâmetros de Desempenho/Ambiente** (técnicos), todos equipados com caixas de ajuda (`help=`).
6. **Estatísticas Descritivas:** Tabelas padronizadas com estatísticas de $N$, Média ($\bar{x}$), Mediana, Desvio Padrão ($s$), Mínimo, Máximo, P25 e P75.
7. **Orientação nos Gráficos:** Caixas do tipo "Como interpretar" integradas abaixo dos gráficos Plotly para esclarecer o significado acadêmico das distribuições sem ilações de causalidade não sustentadas.
8. **Relatório Metodológico Consolidado:** Compilação automática de 16 seções completas e glossário científico para inclusão direta em teses e dissertações.

---

## 🎨 Design e Experiência da Aplicação

A arquitetura de UI do **SLD** foi estruturada em um **Design System centralizado** ([src/sld/ui/styles.py](file:///Users/jairo/Desktop/Mesa%20-%20MacBook%20Air%20de%20Jairo/Sem%20T%C3%ADtulo/src/sld/ui/styles.py)) com componentes reutilizáveis ([src/sld/ui/components.py](file:///Users/jairo/Desktop/Mesa%20-%20MacBook%20Air%20de%20Jairo/Sem%20T%C3%ADtulo/src/sld/ui/components.py)) e gerenciamento de progresso em lotes ([src/sld/ui/tracker.py](file:///Users/jairo/Desktop/Mesa%20-%20MacBook%20Air%20de%20Jairo/Sem%20T%C3%ADtulo/src/sld/ui/tracker.py)).

### Destaques do Design e UX
- **Design System Centralizado:** Paleta de cores sóbria em tons Slate (`#0f172a`, `#334155`, `#e2e8f0`), tipografia limpa, bordas de 6px e cards com sombra imperceptível para leitura acadêmica confortável.
- **Rastreamento de Progresso sem Overhead (`ProgressTracker`):** Rastreia tarefas de longa duração atualizando a interface em lotes (intervalos de 10/25 itens), fornecendo velocidade (itens/min), decorrido e tempo estimado restante sem travar a interface Streamlit.
- **Feedback Discreto por Toasts (`st.toast`):** Notificações curtas de sucesso ao concluir etapas principais sem poluição visual ou uso de animações como balões.
- **Painéis de Conclusão e Estados Vazios (`render_completion_panel` e `render_empty_state`):** Cartões elegantes de estado informando quando uma etapa foi finalizada ou quando aguarda a execução da etapa anterior.
- **Pipeline Stepper Compacto (`render_pipeline_stepper`):** Barra visual indicando o status das 8 etapas do pipeline (`✓ concluído`, `● atual`, `○ pendente`).
- **Hierarquia de Botões:** Botão primário escuro exclusivo para a ação principal da etapa; exportações e filtros utilizam botões secundários; operações destrutivas ficam abrigadas na *Zona de Manutenção*.

## Transparência e Uso de IA

Este projeto foi desenvolvido combinando programação tradicional e auxílio de ferramentas de Inteligência Artificial:

* **Desenvolvimento de Código**: Utilizado o auxílio de IA (Antigravity Vibe Code) para suporte no desenvolvimento de trechos de código e estruturação de partes do sistema. **Todo o código gerado por IA passou por criteriosa análise, revisão, refatoração e validação técnica manual do desenvolvedor.**
* **Identidade Visual**: As artes e imagens visuais da marca e interface foram geradas utilizando a ferramenta **nano banana**.

---

## Autor e Créditos

Desenvolvido por **Jairo Almeida** — [github.com/jairoalmeid](https://github.com/jairoalmeid)


