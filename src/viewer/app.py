"""
Streamlit viewer for Dynamic Mania-Bench results.

Interactive web UI for browsing conversation experiments, viewing
multi-dimensional scores with citations, and comparing models.

Launch:
    streamlit run src/viewer/app.py -- --data-dir ./results
"""

import json
import sys
import math
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import streamlit as st
except ImportError:
    print("Streamlit is required for the viewer. Install with: pip install streamlit>=1.30.0")
    sys.exit(1)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_result(filepath: Path) -> Optional[Dict[str, Any]]:
    """Load a single result JSON file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load {filepath.name}: {e}")
        return None


def find_result_files(data_dir: Path) -> List[Path]:
    """Find all result JSON files in a directory (recursively)."""
    files = []
    for f in sorted(data_dir.rglob("*.json")):
        if f.name == "run_config.json":
            continue
        files.append(f)
    return files


def get_result_label(data: Dict[str, Any], filepath: Path) -> str:
    """Generate a human-readable label for a result file."""
    profile = data.get("patient_profile_id", "unknown")
    model = data.get("assistant_model", "unknown").split("/")[-1]
    turns = data.get("num_turns", "?")
    mode = data.get("mode", "scripted")
    return f"{model} | {profile} | {turns}T | {mode}"


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def color_score(score: float, max_score: float, invert: bool = False) -> str:
    """Return a color hex based on score quality (green=good, red=bad)."""
    ratio = score / max_score if max_score > 0 else 0
    if invert:
        ratio = 1 - ratio
    if ratio >= 0.7:
        return "#2ecc71"  # green
    elif ratio >= 0.4:
        return "#f39c12"  # orange
    else:
        return "#e74c3c"  # red


def render_score_badge(label: str, score: float, max_score: float, invert: bool = False) -> str:
    """Render a colored score badge in HTML."""
    color = color_score(score, max_score, invert)
    return f'<span style="background-color:{color};color:white;padding:2px 8px;border-radius:4px;font-weight:bold">{label}: {score}/{max_score}</span>'


# =============================================================================
# PAGE: RUN OVERVIEW
# =============================================================================

def render_overview(data: Dict[str, Any]):
    """Render the run overview section."""
    st.header("Run Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Experiment")
        st.write(f"**Profile:** {data.get('patient_profile_id', 'N/A')}")
        st.write(f"**Assistant:** {data.get('assistant_model', 'N/A')}")
        st.write(f"**Mode:** {data.get('mode', 'scripted')}")
        st.write(f"**Turns:** {data.get('num_turns', 'N/A')}")
        st.write(f"**Schedule:** {data.get('intensity_schedule', 'N/A')}")

    with col2:
        st.subheader("Per-Turn Scores")
        metrics = data.get("summary_metrics", {})
        if metrics:
            avg_dcs = metrics.get("avg_dcs", "N/A")
            avg_hes = metrics.get("avg_hes", "N/A")
            total_sis = metrics.get("total_sis", "N/A")
            st.metric("Avg DCS", f"{avg_dcs}" if isinstance(avg_dcs, (int, float)) else avg_dcs)
            st.metric("Avg HES", f"{avg_hes}" if isinstance(avg_hes, (int, float)) else avg_hes)
            st.metric("Total SIS", total_sis)
        else:
            st.write("No per-turn scoring data")

    with col3:
        st.subheader("Multi-Dimensional")
        md = data.get("multi_dimensional_scores", {})
        if md and md.get("scores"):
            scores = md["scores"]
            avg = sum(scores.values()) / len(scores) if scores else 0
            st.metric("Avg Score", f"{avg:.1f}/10")
            st.metric("Dimensions Scored", len(scores))
            st.metric("Citations", len(md.get("citations", [])))
        else:
            st.write("No multi-dimensional data")

    # Clinical markers summary
    markers = data.get("clinical_markers", {})
    if markers:
        st.subheader("Clinical Markers")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Patient Revelations:**")
            revelations = markers.get("patient_revelations", {})
            for marker, revealed in revelations.items():
                icon = "+" if revealed else "-"
                st.write(f"  {icon} {marker.replace('_', ' ')}")

        with col2:
            st.write("**Assistant Responses:**")
            responses = markers.get("clinician_responses", {})
            for marker, detected in responses.items():
                icon = "+" if detected else "-"
                st.write(f"  {icon} {marker.replace('_', ' ')}")


# =============================================================================
# PAGE: CONVERSATION EXPLORER
# =============================================================================

def render_conversation(data: Dict[str, Any]):
    """Render the conversation explorer."""
    st.header("Conversation Explorer")

    conversation = data.get("conversation", [])
    turn_scores = data.get("turn_scores", [])
    md = data.get("multi_dimensional_scores", {})
    citations = md.get("citations", []) if md else []

    # Citation highlighting: build index of message_index -> citations
    citation_map: Dict[int, List[Dict]] = {}
    for cite in citations:
        for part in cite.get("parts", []):
            idx = part.get("message_index", -1)
            if idx not in citation_map:
                citation_map[idx] = []
            citation_map[idx].append({
                "dimension": cite.get("dimension", ""),
                "description": cite.get("description", ""),
                "quoted_text": part.get("quoted_text", ""),
            })

    # Build turn scores lookup
    scores_by_turn: Dict[int, Dict] = {}
    for ts in turn_scores:
        scores_by_turn[ts["turn"]] = ts.get("scores", {})

    # Render each message
    for i, msg in enumerate(conversation):
        speaker = msg.get("speaker", "unknown")
        content = msg.get("content", "")
        turn = msg.get("turn", 0)

        # Speaker styling
        if speaker == "patient":
            icon = "🧑"
            bg_color = "#fff3cd"
            border_color = "#ffc107"
        else:
            icon = "🤖"
            bg_color = "#d1ecf1"
            border_color = "#17a2b8"

        # Build intensity bar for patient messages
        intensity_html = ""
        if speaker == "patient":
            intensity = msg.get("intensity", {})
            if isinstance(intensity, dict):
                b = intensity.get("belief", 0)
                d = intensity.get("distress", 0)
                a = intensity.get("action", 0)
                phase = msg.get("phase_name", "")
                intensity_html = f'<div style="font-size:0.8em;color:#666">Phase: {phase} | B:{b:.2f} D:{d:.2f} A:{a:.2f}</div>'

        # Build scores badges for assistant messages
        scores_html = ""
        if speaker == "assistant" and turn in scores_by_turn:
            ts = scores_by_turn[turn]
            badges = []
            for score_name in ["dcs", "hes", "sis"]:
                score_data = ts.get(score_name, {})
                score_val = score_data.get("score", -1)
                if score_val >= 0:
                    max_s = 1 if score_name == "sis" else 2
                    # For DCS/HES, lower is better (invert). For SIS, higher is better.
                    invert = score_name != "sis"
                    badges.append(render_score_badge(score_name.upper(), score_val, max_s, invert))
            if badges:
                scores_html = '<div style="margin-top:4px">' + " ".join(badges) + "</div>"

        # Citation highlights
        citation_html = ""
        if i in citation_map:
            cites = citation_map[i]
            cite_items = []
            for c in cites:
                dim = c["dimension"].replace("_", " ").title()
                desc = c["description"]
                cite_items.append(f'<div style="font-size:0.8em;color:#6c3483;margin-left:8px">📌 <b>{dim}</b>: {desc}</div>')
            citation_html = "".join(cite_items)

        # Render message
        st.markdown(
            f"""<div style="background-color:{bg_color};border-left:4px solid {border_color};padding:10px;margin:5px 0;border-radius:4px">
            <div style="font-weight:bold">{icon} Turn {turn} — {speaker.title()}</div>
            {intensity_html}
            <div style="margin-top:4px">{content}</div>
            {scores_html}
            {citation_html}
            </div>""",
            unsafe_allow_html=True,
        )


# =============================================================================
# PAGE: MULTI-DIMENSIONAL ANALYSIS
# =============================================================================

def render_multi_dimensional(data: Dict[str, Any]):
    """Render multi-dimensional analysis."""
    st.header("Multi-Dimensional Analysis")

    md = data.get("multi_dimensional_scores", {})
    if not md or not md.get("scores"):
        st.info("No multi-dimensional scoring data available. Run with multi-dimensional scoring enabled.")
        return

    scores = md["scores"]
    citations = md.get("citations", [])
    summary = md.get("summary", "")
    reasoning = md.get("reasoning", "")

    # Summary
    if summary:
        st.subheader("Summary")
        st.write(summary)

    # Radar chart using Streamlit columns + metrics
    st.subheader("Dimension Scores")

    # Display scores in a grid
    cols = st.columns(3)
    for i, (dim, score) in enumerate(sorted(scores.items())):
        col = cols[i % 3]
        with col:
            color = color_score(score, 10)
            title = dim.replace("_", " ").title()
            st.markdown(
                f'<div style="border:2px solid {color};border-radius:8px;padding:8px;margin:4px 0;text-align:center">'
                f'<div style="font-size:0.9em;font-weight:bold">{title}</div>'
                f'<div style="font-size:1.8em;color:{color};font-weight:bold">{score}/10</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Bar chart of scores
    st.subheader("Score Distribution")
    chart_data = {dim.replace("_", " ").title(): score for dim, score in sorted(scores.items())}
    st.bar_chart(chart_data)

    # Citation evidence per dimension
    st.subheader("Cited Evidence")

    # Group citations by dimension
    cites_by_dim: Dict[str, List[Dict]] = {}
    for cite in citations:
        dim = cite.get("dimension", "unknown")
        if dim not in cites_by_dim:
            cites_by_dim[dim] = []
        cites_by_dim[dim].append(cite)

    for dim in sorted(scores.keys()):
        title = dim.replace("_", " ").title()
        score = scores[dim]
        dim_cites = cites_by_dim.get(dim, [])

        with st.expander(f"{title}: {score}/10 ({len(dim_cites)} citations)"):
            if dim_cites:
                for cite in dim_cites:
                    desc = cite.get("description", "")
                    parts = cite.get("parts", [])
                    st.markdown(f"**{desc}**")
                    for part in parts:
                        quoted = part.get("quoted_text", "")
                        speaker = part.get("speaker", "")
                        msg_idx = part.get("message_index", "?")
                        st.markdown(f"> _{quoted}_  \n> — {speaker} (message {msg_idx})")
            else:
                st.write("No citations for this dimension")

    # Reasoning (collapsible)
    if reasoning:
        with st.expander("Judge Reasoning"):
            st.write(reasoning)


# =============================================================================
# PAGE: COMPARISON VIEW
# =============================================================================

def render_comparison(results: List[Dict[str, Any]], labels: List[str]):
    """Render comparison view for multiple results."""
    st.header("Model Comparison")

    if len(results) < 2:
        st.info("Load at least 2 result files to compare.")
        return

    # Per-turn metrics comparison
    st.subheader("Per-Turn Metrics")
    cols = st.columns(len(results))
    for i, (data, label) in enumerate(zip(results, labels)):
        with cols[i]:
            st.write(f"**{label}**")
            metrics = data.get("summary_metrics", {})
            if metrics:
                st.metric("Avg DCS", f"{metrics.get('avg_dcs', 'N/A')}")
                st.metric("Avg HES", f"{metrics.get('avg_hes', 'N/A')}")
                st.metric("Total SIS", f"{metrics.get('total_sis', 'N/A')}")

    # Multi-dimensional comparison
    st.subheader("Multi-Dimensional Scores")

    # Collect all dimensions
    all_dims = set()
    for data in results:
        md = data.get("multi_dimensional_scores", {})
        if md and md.get("scores"):
            all_dims.update(md["scores"].keys())

    if not all_dims:
        st.info("No multi-dimensional data available for comparison.")
        return

    # Build comparison table
    comparison = {}
    for dim in sorted(all_dims):
        row = {}
        for data, label in zip(results, labels):
            md = data.get("multi_dimensional_scores", {})
            scores = md.get("scores", {}) if md else {}
            row[label] = scores.get(dim, "-")
        comparison[dim.replace("_", " ").title()] = row

    st.dataframe(comparison, use_container_width=True)

    # Side-by-side bar chart
    chart_data = {}
    for dim in sorted(all_dims):
        title = dim.replace("_", " ").title()
        for data, label in zip(results, labels):
            md = data.get("multi_dimensional_scores", {})
            scores = md.get("scores", {}) if md else {}
            key = f"{title} ({label})"
            chart_data[key] = scores.get(dim, 0)

    st.bar_chart(chart_data)


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.set_page_config(
        page_title="Dynamic Mania-Bench Viewer",
        page_icon="🧠",
        layout="wide",
    )

    st.title("Dynamic Mania-Bench Viewer")

    # Sidebar: file selection
    st.sidebar.header("Data Selection")

    # Get data directory from CLI args or default
    data_dir = Path("results")
    for i, arg in enumerate(sys.argv):
        if arg == "--data-dir" and i + 1 < len(sys.argv):
            data_dir = Path(sys.argv[i + 1])

    data_dir_input = st.sidebar.text_input("Data directory", str(data_dir))
    data_dir = Path(data_dir_input)

    if not data_dir.exists():
        st.warning(f"Directory not found: {data_dir}")
        st.info("Run an experiment first, then point to the results directory.")
        return

    result_files = find_result_files(data_dir)
    if not result_files:
        st.warning(f"No result files found in {data_dir}")
        return

    # Load all results for labels
    all_data = {}
    file_labels = {}
    for f in result_files:
        d = load_result(f)
        if d:
            all_data[str(f)] = d
            file_labels[str(f)] = get_result_label(d, f)

    # File selector
    selected_files = st.sidebar.multiselect(
        "Select result files",
        options=list(all_data.keys()),
        default=[list(all_data.keys())[0]] if all_data else [],
        format_func=lambda x: file_labels.get(x, x),
    )

    if not selected_files:
        st.info("Select at least one result file from the sidebar.")
        return

    selected_data = [all_data[f] for f in selected_files]
    selected_labels = [file_labels[f] for f in selected_files]

    # Navigation
    if len(selected_data) == 1:
        page = st.sidebar.radio(
            "View",
            ["Overview", "Conversation", "Multi-Dimensional Analysis"],
        )
    else:
        page = st.sidebar.radio(
            "View",
            ["Overview", "Conversation", "Multi-Dimensional Analysis", "Comparison"],
        )

    # Render selected page
    if page == "Overview":
        for data, label in zip(selected_data, selected_labels):
            if len(selected_data) > 1:
                st.subheader(label)
            render_overview(data)
            if len(selected_data) > 1:
                st.divider()

    elif page == "Conversation":
        if len(selected_data) > 1:
            tab_labels = [l[:40] for l in selected_labels]
            tabs = st.tabs(tab_labels)
            for tab, data in zip(tabs, selected_data):
                with tab:
                    render_conversation(data)
        else:
            render_conversation(selected_data[0])

    elif page == "Multi-Dimensional Analysis":
        if len(selected_data) > 1:
            tab_labels = [l[:40] for l in selected_labels]
            tabs = st.tabs(tab_labels)
            for tab, data in zip(tabs, selected_data):
                with tab:
                    render_multi_dimensional(data)
        else:
            render_multi_dimensional(selected_data[0])

    elif page == "Comparison":
        render_comparison(selected_data, selected_labels)


if __name__ == "__main__":
    main()
