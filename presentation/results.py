import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from service.results import list_collections, load_collection

_META = {"titulo", "autores", "periodico", "doi", "arquivo_origem", "_arquivo"}

_CARD_CSS = """
<style>
.obs-card {
    background: #1e1e1e;
    border: 1px solid #3a3a3a;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 14px;
    font-family: 'Inter', sans-serif;
}
.obs-title {
    font-size: 1em;
    font-weight: 600;
    color: #e2e2e2;
    margin-bottom: 10px;
    line-height: 1.4;
}
.obs-year {
    display: inline-block;
    background: #7c3aed22;
    border: 1px solid #7c3aed55;
    color: #a78bfa;
    border-radius: 4px;
    padding: 1px 8px;
    font-size: 0.78em;
    font-weight: 600;
    margin-bottom: 10px;
}
.obs-prop {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 4px 0;
    border-bottom: 1px solid #2a2a2a;
    font-size: 0.82em;
}
.obs-prop:last-child { border-bottom: none; }
.obs-key {
    color: #6b7280;
    min-width: 130px;
    flex-shrink: 0;
    padding-top: 1px;
}
.obs-val { color: #d1d5db; line-height: 1.5; }
.obs-val-bool-yes { color: #4ade80; font-weight: 600; }
.obs-val-bool-no  { color: #f87171; font-weight: 600; }
.obs-tag {
    display: inline-block;
    background: #374151;
    color: #9ca3af;
    border-radius: 4px;
    padding: 1px 6px;
    margin: 1px 2px;
    font-size: 0.78em;
}
</style>
"""


def _fmt_value(value) -> str:
    if value is None:
        return '<span style="color:#4b5563">—</span>'
    if isinstance(value, bool) or value in (True, False, "true", "false"):
        is_true = value is True or value == "true"
        cls = "obs-val-bool-yes" if is_true else "obs-val-bool-no"
        label = "Yes" if is_true else "No"
        return f'<span class="{cls}">{label}</span>'
    if isinstance(value, list):
        tags = "".join(f'<span class="obs-tag">{v}</span>' for v in value if v)
        return tags or '<span style="color:#4b5563">—</span>'
    return str(value)


def _render_cards(df: pd.DataFrame, analysis_cols: list[str]) -> None:
    cols = st.columns(2)
    for i, (_, row) in enumerate(df.iterrows()):
        with cols[i % 2]:
            title = row.get("titulo") or row.get("_arquivo", "Untitled")
            year = row.get("ano")
            authors = row.get("autores", "")
            journal = row.get("periodico", "")

            props_html = ""
            if authors:
                props_html += f'<div class="obs-prop"><span class="obs-key">Authors</span><span class="obs-val">{authors}</span></div>'
            if journal:
                props_html += f'<div class="obs-prop"><span class="obs-key">Journal</span><span class="obs-val">{journal}</span></div>'

            for col in analysis_cols:
                val = row.get(col)
                if val is not None:
                    props_html += f'<div class="obs-prop"><span class="obs-key">{col}</span><span class="obs-val">{_fmt_value(val)}</span></div>'

            year_badge = f'<span class="obs-year">{int(year)}</span>' if pd.notna(year) else ""
            st.markdown(
                f'<div class="obs-card">'
                f'<div class="obs-title">{title}</div>'
                f'{year_badge}'
                f'{props_html}'
                f'</div>',
                unsafe_allow_html=True,
            )


def _render_graph(df: pd.DataFrame, analysis_cols: list[str]) -> None:
    G = nx.Graph()

    for _, row in df.iterrows():
        node_id = row["_arquivo"]
        label = (row.get("titulo") or node_id)
        label_short = label[:35] + "…" if len(label) > 35 else label
        G.add_node(node_id, label=label_short, full_title=label, ano=row.get("ano"))

    # Connect articles that share categorical values
    for col in analysis_cols:
        col_data = df[["_arquivo", col]].dropna()
        for _, grp in col_data.groupby(col_data[col].apply(
            lambda x: str(x) if not isinstance(x, list) else None
        )):
            members = grp["_arquivo"].tolist()
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = members[i], members[j]
                    if G.has_edge(a, b):
                        G[a][b]["weight"] += 1
                    else:
                        G.add_edge(a, b, weight=1)

    if G.number_of_nodes() == 0:
        st.info("No articles to display.")
        return

    pos = nx.spring_layout(G, seed=42, k=2.5)

    # Edges
    edge_x, edge_y, edge_w = [], [], []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1.2, color="#4b5563"),
        hoverinfo="none",
    )

    # Nodes
    node_x, node_y, node_text, node_hover, node_color = [], [], [], [], []
    anos = [G.nodes[n].get("ano") for n in G.nodes]
    valid_anos = [a for a in anos if pd.notna(a)]
    ano_min = min(valid_anos) if valid_anos else 2000
    ano_max = max(valid_anos) if valid_anos else 2024

    for node in G.nodes:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        d = G.nodes[node]
        node_text.append(d["label"])
        node_hover.append(f"<b>{d['full_title']}</b><br>Year: {int(d['ano']) if pd.notna(d.get('ano')) else '—'}<br>Connections: {G.degree(node)}")
        ano = d.get("ano")
        node_color.append(int(ano) if pd.notna(ano) else ano_min)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        textfont=dict(size=9, color="#9ca3af"),
        hovertext=node_hover,
        hoverinfo="text",
        marker=dict(
            size=14,
            color=node_color,
            colorscale="Viridis",
            cmin=ano_min,
            cmax=ano_max,
            colorbar=dict(title="Year", thickness=12, len=0.5),
            line=dict(width=1.5, color="#1e1e1e"),
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            height=520,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            hovermode="closest",
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Nodes colored by year. Edges connect articles that share categorical parameter values.")


def _render_charts(df: pd.DataFrame, analysis_cols: list[str]) -> None:
    if "ano" in df.columns and df["ano"].notna().any():
        counts = df["ano"].dropna().astype(int).value_counts().sort_index().reset_index()
        counts.columns = ["Year", "Count"]
        fig = px.bar(counts, x="Year", y="Count", text="Count", title="Publications by year")
        fig.update_traces(textposition="outside", marker_color="#7c3aed")
        fig.update_layout(paper_bgcolor="#111827", plot_bgcolor="#111827",
                          font_color="#d1d5db", xaxis=dict(type="category"))
        st.plotly_chart(fig, use_container_width=True)

    grid = st.columns(2)
    for i, col in enumerate(analysis_cols):
        series = df[col].dropna()
        if series.empty:
            continue
        with grid[i % 2]:
            is_list_col = series.apply(lambda x: isinstance(x, list)).any()
            if is_list_col:
                data = series.explode().dropna().astype(str).value_counts().reset_index()
                data.columns = ["Value", "Count"]
                fig = px.bar(data, x="Count", y="Value", orientation="h", title=col)
            elif set(series.map(str).unique()) <= {"True", "False", "true", "false"}:
                labels = series.map({True: "Yes", False: "No", "true": "Yes", "false": "No", "True": "Yes", "False": "No"})
                data = labels.value_counts().reset_index()
                data.columns = ["Value", "Count"]
                fig = px.pie(data, names="Value", values="Count", title=col,
                             color_discrete_map={"Yes": "#4ade80", "No": "#f87171"})
            else:
                data = series.astype(str).value_counts().head(12).reset_index()
                data.columns = ["Value", "Count"]
                fig = px.bar(data, x="Count", y="Value", orientation="h", title=col)

            fig.update_layout(paper_bgcolor="#111827", plot_bgcolor="#111827",
                              font_color="#d1d5db", yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)


def _apply_filters(df: pd.DataFrame, analysis_cols: list[str]) -> pd.DataFrame:
    with st.expander("Filters", expanded=False):
        if "ano" in df.columns and df["ano"].notna().any():
            year_min, year_max = int(df["ano"].min()), int(df["ano"].max())
            if year_min < year_max:
                lo, hi = st.slider("Publication year", year_min, year_max, (year_min, year_max))
                df = df[df["ano"].between(lo, hi) | df["ano"].isna()]

        for col in analysis_cols:
            series = df[col].dropna()
            if series.empty or series.apply(lambda x: isinstance(x, list)).any():
                continue
            uniq = sorted(series.astype(str).unique())
            if 2 <= len(uniq) <= 12:
                sel = st.multiselect(col, uniq, default=uniq)
                df = df[df[col].astype(str).isin(sel) | df[col].isna()]
    return df


def show():
    st.markdown(_CARD_CSS, unsafe_allow_html=True)
    st.header("Results")

    collections = list_collections()
    if not collections:
        st.info("No collections found. Analyse articles in the **Análise** tab first.")
        return

    selected = st.selectbox("Collection", collections)
    df = load_collection(selected)

    if df.empty:
        st.warning("No results found in this collection.")
        return

    analysis_cols = [c for c in df.columns if c not in _META and c != "ano"]

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Articles", len(df))
    if "ano" in df.columns and df["ano"].notna().any():
        c2.metric("Earliest", int(df["ano"].min()))
        c3.metric("Latest", int(df["ano"].max()))
    c4.metric("Parameters", len(analysis_cols))

    st.divider()
    df = _apply_filters(df, analysis_cols)
    st.caption(f"{len(df)} article(s) after filters")
    st.divider()

    view = st.radio("View", ["Cards", "Graph", "Charts", "Table"], horizontal=True)

    if view == "Cards":
        _render_cards(df, analysis_cols)
    elif view == "Graph":
        _render_graph(df, analysis_cols)
    elif view == "Charts":
        _render_charts(df, analysis_cols)
    else:
        display_cols = [c for c in df.columns if c != "_arquivo"]
        st.dataframe(df[display_cols], use_container_width=True)
