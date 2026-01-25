import markdown
from xhtml2pdf import pisa
from io import BytesIO
import datetime

def generate_markdown_report(state: dict) -> str:
    """
    Converts DiscoveryState dict into a formatted Markdown report.
    Feature-rich export for grants/auditability.
    """
    import datetime
    
    query = state.get("user_query", "Unknown Query")
    hypotheses = state.get("hypotheses", [])
    decision = state.get("decision_summary", {})
    literature = state.get("literature", {})
    papers = literature.get("papers", []) if literature else state.get("documents", []) # fallback
    graph = state.get("concept_graph", {})
    edges = graph.get("edges", []) if graph else []
    
    # Extract Decision fields safely
    if isinstance(decision, dict):
        system_confidence = decision.get("system_confidence", "Moderate")
        key_risks = decision.get("key_risks", [])
        next_steps = decision.get("recommended_next_steps", [])
        primary_reason = decision.get("primary_hypothesis_reason", "")
    else:
        system_confidence = getattr(decision, "system_confidence", "Moderate")
        key_risks = getattr(decision, "key_risks", [])
        next_steps = getattr(decision, "recommended_next_steps", [])
        primary_reason = getattr(decision, "primary_hypothesis_reason", "")

    md = []
    
    # =========================================================================
    # 1. RUN SUMMARY (Ask #1)
    # =========================================================================
    md.append(f"# SciNets Discovery Report")
    md.append(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    md.append(f"**Query:** {query}")
    md.append("")
    
    md.append("## Run Summary")
    md.append(f"- **Papers Analyzed:** {len(papers)}")
    md.append(f"- **Hypotheses Generated:** {len(hypotheses)}")
    md.append(f"- **Graph Connections:** {len(edges)}")
    md.append(f"- **System Confidence:** {system_confidence}")
    md.append(f"- **Mode:** {state.get('hypothesis_mode', 'standard').capitalize()}")
    md.append("")
    md.append("---")
    md.append("")

    # =========================================================================
    # 2. EXECUTIVE SYNTHESIS
    # =========================================================================
    if primary_reason:
        md.append(f"## Executive Synthesis")
        md.append(primary_reason)
        md.append("")
    
    # =========================================================================
    # 3. HYPOTHESIS ANALYSIS
    # =========================================================================
    md.append("## Hypotheses Analysis")
    
    for i, h in enumerate(hypotheses):
        # Guard against malformed hypothesis
        if isinstance(h, str):
            md.append(f"### Hypothesis {i+1}")
            md.append(f"> {h}")
            md.append("")
            continue
            
        # Object access helpers
        is_obj = hasattr(h, "text")
        text = h.text if is_obj else h.get("text", "")
        # stability_class = h.stability_class if is_obj else h.get("stability_class", "speculative")
        
        # Determine status label (Ask #9)
        # We derive vaguely from evidence content if 'stability_class' is generic
        # but let's stick to the mapped variable first.
        
        rationale_gap = h.rationale_gap if is_obj else h.get("rationale_gap", {})
        causal_chain = h.causal_chain if is_obj else h.get("causal_chain", {})
        evidence_list = h.evidence if is_obj else h.get("evidence", [])
        roadmap = h.confidence_roadmap if is_obj else h.get("confidence_roadmap", [])
        
        # Calculate stance counts (Ask #5)
        supports = [e for e in evidence_list if (e.stance if hasattr(e, "stance") else e.get("stance")) == "support"]
        contradicts = [e for e in evidence_list if (e.stance if hasattr(e, "stance") else e.get("stance")) == "contradict"]
        neutrals = [e for e in evidence_list if (e.stance if hasattr(e, "stance") else e.get("stance")) == "neutral"]
        
        # Improved Veridct Handling
        if len(supports) > len(contradicts) and len(contradicts) == 0:
            status_label = "Supported"
        elif len(contradicts) > 0:
            status_label = "Mixed / Contested"
        elif len(supports) == 0:
            status_label = "Speculative"
        else:
            status_label = "Partially Supported"

        md.append(f"### Hypothesis {i+1}: {status_label}")
        md.append(f"> {text}")
        md.append("")
        
        # A) Structural Gap (Ask #2)
        if rationale_gap:
            # Handle if it's an object
            rg_dict = rationale_gap.dict() if hasattr(rationale_gap, "dict") else rationale_gap
            if rg_dict:
                md.append("#### Structural Gap Analysis")
                if rg_dict.get("missing_link"):
                    md.append(f"**Missing Link:** {rg_dict.get('missing_link')}")
                if rg_dict.get("disconnected_clusters"):
                    clusters = ", ".join(rg_dict.get("disconnected_clusters", []))
                    md.append(f"**Disconnected Clusters:** {clusters}")
                if rg_dict.get("structural_reason"):
                    md.append(f"**Why Existing Models Fail:** {rg_dict.get('structural_reason')}")
                md.append("")

        # B) Mechanistic Chain (Ask #3)
        if causal_chain:
            cc_dict = causal_chain.dict() if hasattr(causal_chain, "dict") else causal_chain
            nodes = cc_dict.get("nodes", [])
            if nodes:
                md.append("#### Mechanistic Chain")
                chain_str = " -> ".join(nodes)
                md.append(f"**{chain_str}**")
                md.append("")

        # C) Evidence Table (Ask #4)
        if evidence_list:
            md.append("#### Evidence Table")
            # Markdown Table Header
            md.append("| Title | Stance | Strength | Year |")
            md.append("|---|---|---|---|")
            for e in evidence_list:
                e_obj = e if isinstance(e, dict) else e.dict() # Handle object
                title = e_obj.get("title", "Unknown")[:60] + "..." if len(e_obj.get("title", "")) > 60 else e_obj.get("title", "Unknown")
                stance = e_obj.get("stance", "neutral").capitalize()
                strength = e_obj.get("strength", 1)
                year = e_obj.get("year") or "-"
                md.append(f"| {title} | {stance} | {strength}/5 | {year} |")
            md.append("")
        
        # D) Grounding Status (Ask #5)
        md.append("#### Grounding Status")
        md.append(f"**VERDICT:** {status_label}")
        md.append(f"- **Supporting:** {len(supports)}")
        md.append(f"- **Contradicting:** {len(contradicts)}")
        md.append(f"- **Neutral:** {len(neutrals)}")
        if is_obj and h.evidence_summary:
            md.append(f"\n{h.evidence_summary}")
        elif isinstance(h, dict) and h.get("evidence_summary"):
             md.append(f"\n{h.get('evidence_summary')}")
        md.append("")

        # E) What would increase confidence? (Ask #6)
        if roadmap:
            md.append("#### What would increase confidence?")
            for item in roadmap:
                md.append(f"- {item}")
            md.append("")

        md.append("---")
        md.append("")

    # =========================================================================
    # 4. STRATEGIC RISKS (Ask #7)
    # =========================================================================
    if key_risks:
        md.append("## Strategic Risks")
        for risk in key_risks:
            md.append(f"- {risk}")
        md.append("")
        md.append("---")
        md.append("")

    # =========================================================================
    # 5. NEXT STEPS
    # =========================================================================
    if next_steps:
        md.append("## Recommended Next Steps")
        for step in next_steps:
            md.append(f"- {step}")
        md.append("")
    
    # =========================================================================
    # 6. FOOTER (Ask #8)
    # =========================================================================
    md.append("")
    md.append(f"**System Confidence:** {system_confidence}")
    md.append("_Generated by SciNets Discovery Engine_")

    return "\n".join(md)

def render_html_report(markdown_content: str) -> bytes:
    """
    Renders Markdown content to a standalone HTML report.
    Fallback for PDF generation issues on slim containers.
    """
    # 1. Convert Markdown to HTML
    html_content = markdown.markdown(markdown_content)
    
    # 2. Wrap in simple CSS (Print optimized)
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>SciNets Discovery Report</title>
        <style>
            @page {{
                size: A4;
                margin: 2.5cm;
            }}
            body {{
                font-family: Helvetica, Arial, sans-serif;
                font-size: 11pt;
                line-height: 1.6;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
                padding: 40px 20px;
            }}
            h1 {{ color: #1a1a1a; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
            h2 {{ color: #2c3e50; margin-top: 30px; border-bottom: 1px solid #eee; }}
            h3 {{ color: #16a085; margin-top: 25px; }}
            blockquote {{
                background: #f9f9f9;
                border-left: 5px solid #ccc;
                margin: 1.5em 10px;
                padding: 0.5em 10px;
            }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            /* Print Specifics */
            @media print {{
                body {{ max-width: 100%; padding: 0; }}
                a {{ text-decoration: none; color: #000; }}
            }}
        </style>
    </head>
    <body>
        {html_content}
        <script>
            // Auto-trigger print dialog for convenience
            // window.print();
        </script>
    </body>
    </html>
    """
    
    return full_html.encode('utf-8')
