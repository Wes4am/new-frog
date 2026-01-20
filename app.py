import json
import re
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag
import streamlit.components.v1 as components
import requests
import streamlit as st
from bs4 import BeautifulSoup
from pyvis.network import Network
import tempfile
import os

UA = "SimpleFlowCrawler/0.1"

def normalize_url(base: str, href: str) -> str | None:
    if href is None:
        return None
    href = href.strip()

    if href.startswith(("mailto:", "tel:", "javascript:", "data:")):
        return None

    abs_url = urljoin(base, href)
    abs_url, _ = urldefrag(abs_url)

    p = urlparse(abs_url)
    if p.scheme not in ("http", "https"):
        return None

    netloc = p.netloc
    if netloc.endswith(":80") and p.scheme == "http":
        netloc = netloc[:-3]
    if netloc.endswith(":443") and p.scheme == "https":
        netloc = netloc[:-4]

    path = re.sub(r"/{2,}", "/", p.path or "/")
    return p._replace(netloc=netloc, path=path).geturl()

def is_internal(url: str, root_netloc: str) -> bool:
    try:
        return urlparse(url).netloc == root_netloc
    except Exception:
        return False

def fetch_html(url: str, timeout: int = 15):
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout, allow_redirects=True)
        ct = (r.headers.get("Content-Type") or "").lower()
        if "text/html" not in ct:
            return r.status_code, None
        return r.status_code, r.text
    except Exception:
        return None, None

def extract_links(page_url: str, html: str) -> set[str]:
    soup = BeautifulSoup(html, "html.parser")
    out = set()
    for a in soup.find_all("a", href=True):
        u = normalize_url(page_url, a.get("href"))
        if u:
            out.add(u)
    return out

def get_url_label(url: str, max_length: int = 40) -> str:
    """Create a readable label from URL"""
    parsed = urlparse(url)
    path = parsed.path.rstrip('/') or '/'
    
    # If it's just the homepage
    if path == '/':
        return parsed.netloc
    
    # Show the last part of the path
    parts = [p for p in path.split('/') if p]
    if parts:
        label = '/' + '/'.join(parts[-2:]) if len(parts) > 1 else '/' + parts[-1]
    else:
        label = path
    
    # Truncate if too long
    if len(label) > max_length:
        label = '...' + label[-(max_length-3):]
    
    return label

def build_network_graph(nodes: list[str], edges: list[tuple[str, str]], status_by_url: dict, seed_url: str, layout: str = "hierarchical"):
    """Build an interactive network graph using pyvis"""
    net = Network(
        height="800px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#000000",
        directed=True
    )
    
    # Different layout configurations
    if layout == "hierarchical":
        net.set_options("""
        {
            "layout": {
                "hierarchical": {
                    "enabled": true,
                    "levelSeparation": 150,
                    "nodeSpacing": 200,
                    "treeSpacing": 250,
                    "direction": "UD",
                    "sortMethod": "directed"
                }
            },
            "physics": {
                "enabled": false
            },
            "nodes": {
                "font": {
                    "size": 12,
                    "face": "arial"
                },
                "borderWidth": 2
            },
            "edges": {
                "smooth": {
                    "type": "cubicBezier",
                    "forceDirection": "vertical",
                    "roundness": 0.4
                },
                "arrows": {
                    "to": {
                        "enabled": true,
                        "scaleFactor": 0.5
                    }
                },
                "color": {
                    "color": "#848484",
                    "highlight": "#FF0000",
                    "opacity": 0.6
                }
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 100,
                "navigationButtons": true,
                "keyboard": true
            }
        }
        """)
    else:  # physics-based layout
        net.set_options("""
        {
            "physics": {
                "enabled": true,
                "stabilization": {
                    "enabled": true,
                    "iterations": 500
                },
                "barnesHut": {
                    "gravitationalConstant": -30000,
                    "centralGravity": 0.8,
                    "springLength": 250,
                    "springConstant": 0.001,
                    "damping": 0.3,
                    "avoidOverlap": 0.8
                }
            },
            "nodes": {
                "font": {
                    "size": 11,
                    "face": "arial"
                },
                "borderWidth": 2
            },
            "edges": {
                "smooth": {
                    "type": "continuous"
                },
                "arrows": {
                    "to": {
                        "enabled": true,
                        "scaleFactor": 0.5
                    }
                },
                "color": {
                    "color": "#848484",
                    "opacity": 0.5
                }
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 100,
                "hideEdgesOnDrag": true,
                "hideEdgesOnZoom": true,
                "navigationButtons": true
            }
        }
        """)
    
    # Calculate node levels/depths for sizing
    node_depths = {}
    node_set = set(nodes)
    
    # BFS to calculate depths
    from collections import deque
    q = deque([(seed_url, 0)])
    visited_depth = {seed_url: 0}
    
    while q:
        url, depth = q.popleft()
        node_depths[url] = depth
        for source, target in edges:
            if source == url and target in node_set and target not in visited_depth:
                visited_depth[target] = depth + 1
                q.append((target, depth + 1))
    
    # Add nodes with colors based on status
    for url in nodes:
        status = status_by_url.get(url)
        label = get_url_label(url)
        depth = node_depths.get(url, 0)
        
        # Color based on HTTP status
        if status == 200:
            color = "#4CAF50"  # Green for success
        elif status and 300 <= status < 400:
            color = "#FF9800"  # Orange for redirects
        elif status and 400 <= status < 500:
            color = "#F44336"  # Red for client errors
        elif status and 500 <= status < 600:
            color = "#9C27B0"  # Purple for server errors
        else:
            color = "#9E9E9E"  # Gray for unknown
        
        # Seed URL is larger
        size = 35 if url == seed_url else max(15, 25 - depth * 2)
        
        title = f"{url}\nStatus: {status if status else 'N/A'}\nDepth: {depth}"
        
        net.add_node(
            url,
            label=label,
            title=title,
            color=color,
            size=size,
            level=depth if layout == "hierarchical" else None
        )
    
    # Add edges (only if both nodes exist)
    for source, target in edges:
        if source in node_set and target in node_set:
            net.add_edge(source, target)
    
    return net

def crawl(seed: str, max_pages: int, max_depth: int):
    seed = normalize_url(seed, "") or seed
    root_netloc = urlparse(seed).netloc

    q = deque([(seed, 0)])
    visited = set()
    edges = set()
    status_by_url = {}

    progress_bar = st.progress(0)
    status_text = st.empty()

    while q and len(visited) < max_pages:
        url, depth = q.popleft()
        if url in visited:
            continue
        visited.add(url)
        
        # Update progress
        progress = len(visited) / max_pages
        progress_bar.progress(min(progress, 1.0))
        status_text.text(f"Crawling: {len(visited)} pages found...")

        status, html = fetch_html(url)
        status_by_url[url] = status

        if not html or depth >= max_depth:
            continue

        for link in extract_links(url, html):
            if not is_internal(link, root_netloc):
                continue
            edges.add((url, link))
            if link not in visited:
                q.append((link, depth + 1))

    progress_bar.empty()
    status_text.empty()

    nodes = sorted(visited)
    edges_list = sorted(edges)
    data = {
        "seed": seed,
        "nodes": [{"url": u, "status": status_by_url.get(u)} for u in nodes],
        "edges": [{"from": a, "to": b} for a, b in edges_list],
    }
    
    return data, nodes, edges_list, status_by_url

def render_network(net: Network):
    """Render the pyvis network in Streamlit"""
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as f:
        net.save_graph(f.name)
        with open(f.name, 'r', encoding='utf-8') as file:
            html_content = file.read()
        os.unlink(f.name)
    
    # Display in Streamlit
    components.html(html_content, height=750, scrolling=False)

# Streamlit UI
st.set_page_config(page_title="Website Flow Crawler", layout="wide")

st.title("🕷️ Website Flow Crawler")
st.caption("Enter a website → Crawl internal pages → Interactive network visualization")

col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
with col1:
    seed = st.text_input("Seed URL", value="https://example.com")
with col2:
    max_pages = st.number_input("Max pages", min_value=1, max_value=5000, value=200, step=50)
with col3:
    max_depth = st.number_input("Max depth", min_value=0, max_value=20, value=3, step=1)
with col4:
    layout_type = st.selectbox("Layout", ["hierarchical", "physics"], index=0)

run = st.button("🚀 Crawl & Build Network Graph", use_container_width=True)

if run:
    if not seed.startswith(("http://", "https://")):
        st.error("Please include http:// or https:// in the URL.")
        st.stop()

    with st.spinner("Crawling website..."):
        data, nodes, edges_list, status_by_url = crawl(seed, int(max_pages), int(max_depth))

    # Summary section
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📄 Pages Found", len(data['nodes']))
    with col2:
        st.metric("🔗 Links Found", len(data['edges']))
    with col3:
        success_pages = sum(1 for n in data['nodes'] if n.get('status') == 200)
        st.metric("✅ Successful (200)", success_pages)
    with col4:
        error_pages = sum(1 for n in data['nodes'] if n.get('status', 0) >= 400)
        st.metric("❌ Errors (4xx/5xx)", error_pages)

    # Download section
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Download crawl.json",
            data=json.dumps(data, indent=2, ensure_ascii=False),
            file_name="crawl.json",
            mime="application/json",
            use_container_width=True,
        )
    with col2:
        # Create CSV export
        csv_data = "URL,Status,Type\n"
        for node in data['nodes']:
            csv_data += f"\"{node['url']}\",{node.get('status', 'N/A')},page\n"
        
        st.download_button(
            "⬇️ Download crawl.csv",
            data=csv_data,
            file_name="crawl.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Network visualization
    st.divider()
    st.subheader("📊 Interactive Site Flow Network")
    
    # Add legend
    st.markdown("""
    **Legend:** 
    🟢 Green = Success (200) | 
    🟠 Orange = Redirect (3xx) | 
    🔴 Red = Client Error (4xx) | 
    🟣 Purple = Server Error (5xx) | 
    ⚪ Gray = Unknown
    """)
    
    if layout_type == "hierarchical":
        st.info("💡 **Hierarchical Layout:** Top-down tree structure. Pages flow from seed URL downward. No jiggling!")
    else:
        st.info("💡 **Physics Layout:** Force-directed graph. Click and drag nodes to reorganize. Graph will stabilize after a few seconds.")
    
    with st.spinner("Building network graph..."):
        net = build_network_graph(nodes, edges_list, status_by_url, seed, layout_type)
        render_network(net)

    # Show page list in expander
    with st.expander("📋 View All Pages"):
        for node in data['nodes']:
            status_emoji = "✅" if node.get('status') == 200 else "❌" if node.get('status', 0) >= 400 else "⚠️"
            st.text(f"{status_emoji} [{node.get('status', 'N/A')}] {node['url']}")
