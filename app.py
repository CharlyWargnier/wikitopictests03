import streamlit as st
import pandas as pd
import numpy as np

#import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

from bs4 import BeautifulSoup
#import pywikibot
import math
import os
import re
import requests
import tempfile
import validators

from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
from google.cloud import language_v1
from google.cloud.language_v1 import enums


import streamlit as st
import streamlit.components.v1 as components

# import json
#import networkx as nx
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt
from pyvis import network as net

# from IPython.core.display import display, HTML
import got

st.set_page_config(
    page_title="Wiki Topic Grapher",
    page_icon="✨",
)

#######################

c30, c32 = st.beta_columns(2)

with c30:
    # st.image("logo.jpg", width=275)
    st.text("")
    st.image("Wikilogo.png", width=475)
    st.header("")


with c32:
    st.header("")
    # st.text("")
    st.text("")
    st.text("")
    # st.header("")
    st.markdown(
        "###### Original script by JR Oakes - Ported to [![this is an image link](https://i.imgur.com/iIOA6kU.png)](https://www.streamlit.io/)&nbsp, with :heart: by [@DataChaz](https://twitter.com/DataChaz) &nbsp [![this is an image link](https://i.imgur.com/thJhzOO.png)](https://www.buymeacoffee.com/cwar05)"
    )


with st.beta_expander("ℹ️ - About this app ", expanded=True):

    st.write(
        """  
    
-   This data app uses _________________ to ___________________________.
-   Remove spinners - cache function i.e. Running recurse_entities
-   This app is free. If it's useful to you, you can [buy me a coffee](https://www.buymeacoffee.com/cwar05) to support my work! 🙏

	    """
    )


with st.beta_expander("ℹ️ - Todo - roadmap", expanded=True):

    st.write(
        """  

-   Deploy on S4!!!
-   Add Joahnnes' app
-   Fix issue with PyVis format - see screenshot - https://imgur.com/a/lnskLcI
-   Add Black formatting
-   Rule: if exception doesn't work - Do a catch-all (see below!)
-   https://calendar.google.com/calendar/u/0/r/eventedit/MHJmcmY1dmtjdmpramRrZmVnamR2NnZ2M21fMjAyMTAyMTZUMTUwMDAwWiBjd2FyMDVAbQ
-   Add wikipedia http check - warning if 404
-   add default minumum and maximum depth + limit (watch table (Exce))
-   add default depth as 2 as a param
-   Check issue with canvas size e.g. limit = 2 and depth = 2

	    """
    )

with st.beta_expander("ℹ️ - Errors to fix ", expanded=True):

    st.write(
        """  
-   CREDENTIALS - CREDENTIALS - CREDENTIALS - CREDENTIALS - CREDENTIALS - 
-   KeyError: 'salience' - Means Error with language API:  403 This API method requires billing to be enabled
-   Create a "st.warning" if wrong (404/non existing) URL (Currently: TypeError: object of type 'NoneType' has no len())
-   Add "add credentials! - IndexError: list index out of range
-   Can't display credentials in app, tried: GADf = pd.read_json('nlp-colab.json') GADf = pd.read_json(fp.name)
-   Remove "Error: search: searchstring cannot be empty"
	    """
    )


with st.beta_expander("ℹ️ - Fixed ", expanded=False):

    st.write(
        """  

-   Add title tag + emoji  
-   Find name and add logo (#1 WikiKGChecker!)
-   Add rules for URLs
-   Add timer from https://streamea-entity-analyzer.herokuapp.com/
-   Add %age from https://streamea-entity-analyzer.herokuapp.com/-   fix index in table
-   Remove spinners from cache functions
-   Add all fields
-   Deploy 01 - Deploy on S4
-   Cached seemed OK as toggle doesn't remove the chart
-   issue with table (doesn't display)
-   display dataframe 
-   add columns/layout
-   Add download button formdataframe
-   Add toggle + URLs RULES

	    """
    )

with st.beta_expander("ℹ️ - Text/Copy  ", expanded=False):

    st.write(
        """  

-   https://www.oncrawl.com/technical-seo/topic-graph-wikipedia/
-   USE CASES USE CASES USE CASES USE CASES USE CASES 
-   #1 - map out associations with products and brands ()
-   SPEND SPEND SPEND SPEND SPEND SPEND SPEND SPEND SPEND 
-   Update: The pricing is based on units of 1,000 unicode characters sent to the API and is free up to 5k units. Since Wikipedia articles can get long, you want to watch your spend

	    """
    )


with st.beta_expander("ℹ️ - Later", expanded=True):

    st.write(
        """  

-   English only - Add mutlilingual display dataframe 
-   Add mulitlanguages via usernames['wikipedia']['en'] = 'test
	    """
    )


with st.beta_expander("ℹ️ - Parked ", expanded=True):

    st.write(
        """  

-   Add gif when functions are running

	    """
    )


st.markdown("## **① Upload your Google NLP credentials 🗝️**")  #########
# st.text('')
with st.beta_expander("ℹ️ - How to create your credentials?", expanded=False):

    st.write(
        """
	          
      - In the [Cloud Console](https://console.cloud.google.com/), go to the _'Create Service Account Key'_  page
      - From the *Service account list*, select  _'New service account'_
      - In the *Service account name* field, enter a name
      - From the *Role list*, select  _'Project > Owner'_
      - Click create, then download your JSON key
      - Upload it (or drag and drop it) in the grey box above ☝️

	    """
    )

# region Layout size ####################################################################################


def _max_width_():
    max_width_str = f"max-width: 1200px;"
    # max_width_str = f"max-width: 1550px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


_max_width_()

# endregion Layout size ####################################################################################

# hv.extension('bokeh')
# defaults = dict(width=400, height=400)
# hv.opts.defaults(
#    opts.EdgePaths(**defaults), opts.Graph(**defaults), opts.Nodes(**defaults))
#
#
## Declare abstract edges
# N = 8
# node_indices = np.arange(N, dtype=np.int32)
# source = np.zeros(N, dtype=np.int32)
# target = node_indices
#
#
# simple_graph = hv.Graph(((source, target),))
##simple_graph
##st.bokeh_chart(hv.render(simple_graph, backend='bokeh')) # will work
#
# G = nx.karate_club_graph()
# karate_club_graph = hv.Graph.from_networkx(G, nx.layout.circular_layout).opts(tools=['hover'])
# st.bokeh_chart(hv.render(karate_club_graph, backend='bokeh')) # will work


# Pywikibot needs a config file
pywikibot_config = r"""# -*- coding: utf-8  -*-
mylang = 'en'
family = 'wikipedia'
usernames['wikipedia']['en'] = 'test'"""

with open("user-config.py", "w", encoding="utf-8") as f:
    f.write(pywikibot_config)

# endregion imports

# region key

c3, c4 = st.beta_columns(2)

with c3:
    try:
        uploaded_file = st.file_uploader("", type="json")
        with tempfile.NamedTemporaryFile(delete=False) as fp:
            fp.write(uploaded_file.getvalue())
        try:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = fp.name
            # Tweak - unhash if other line doesn't work or is not retained in the state
            # with open(fp.name,'wb') as a:
            # source: https://stackoverflow.com/questions/5512811/builtins-typeerror-must-be-str-not-bytes
            with open(fp.name, "rb") as a:
                # Tweak - unhash if other line doesn't work or is not retained in the state
                # with open(fp.name) as a:
                # st.write(a.read())
                # client = language.LanguageServiceClient()

                # client = language_v1.LanguageServiceClient()
                client = language.LanguageServiceClient.from_service_account_json(
                    fp.name
                )

        finally:
            if os.path.isfile(fp.name):
                os.unlink(fp.name)

    except AttributeError:

        print("wait")


with c4:
    st.markdown("###")
    c = st.beta_container()
    if uploaded_file:
        st.success("✅ Nice! Your credentials are uploaded!")

# uploaded_file


# region functions


def google_nlp_entities(
    input,
    input_type="html",
    result_type="all",
    limit=10,
    invalid_types=["OTHER", "NUMBER", "DATE"],
):

    """
    Loads HTML or text from a URL and passes to the Google NLP API
    Parameters:
        * input: HTML or Plain Text to send to the Google Language API
        * input_type: Either `html` or `text` (string)
        * result_type: Either `all`(pull all entities) or `wikipedia` (only pull entities with Wikipedia pages)
        * limit: Limits the number of results to this number sorted, decending, by salience.
        * invalid_types: A list of entity types to exclude.
    Returns:
        List of entities in format [{'name':<name>,'type':<type>,'salience':<salience>, 'wikipedia': <wikipedia url - optional>}]
    """

    # client = language.LanguageServiceClient.from_service_account_json('fp.name')

    def get_type(type):
        return client.enums.Entity.Type(d.type).name

    if not input:
        print("No input content found.")
        return None

    if input_type == "html":
        doc_type = language.enums.Document.Type.HTML
    else:
        doc_type = language.enums.Document.Type.PLAIN_TEXT

    document = types.Document(content=input, type=doc_type)

    features = {"extract_entities": True}

    try:
        response = client.annotate_text(
            document=document, features=features, timeout=20
        )
    except Exception as e:
        print("Error with language API: ", re.sub(r"\(.*$", "", str(e)))
        return []

    used = []
    results = []
    for d in response.entities:

        if limit and len(results) >= limit:
            break

        if get_type(d.type) not in invalid_types and d.name not in used:

            data = {
                "name": d.name,
                "type": client.enums.Entity.Type(d.type).name,
                "salience": d.salience,
            }
            if result_type is "wikipedia":
                if "wikipedia_url" in d.metadata:
                    data["wikipedia"] = d.metadata["wikipedia_url"]
                    results.append(data)
            else:
                results.append(data)

            used.append(d.name)

    return results


# @st.cache(allow_output_mutation=True,show_spinner=False,show_spinner=False)
def load_page_title(url):
    """
    Returns the <title> given a URL.
    Parameters:
        * url: URL (string)
    Returns:
       Inner text of <title> (string)
    """
    soup = BeautifulSoup(requests.get(url).text)
    return soup.title.text


@st.cache(allow_output_mutation=True, show_spinner=False)
# @st.cache(allow_output_mutation=True,show_spinner=False,show_spinner=False)
def html_to_text(html, target_elements=None):
    """
    Transforms HTML to clean text
    Parameters:
        * html: HTML from a web page (str)
        * target_elements: Elements like `div` or `p` to target pulling text from. (optional) (string)
    Returns:
        Text (string)
    """
    soup = BeautifulSoup(html)

    for script in soup(
        ["script", "style"]
    ):  # remove all javascript and stylesheet code
        script.extract()

    targets = []

    if target_elements:
        targets = soup.find_all(target_elements)

    if target_elements and len(targets) > 3:
        text = " ".join([t.text for t in targets])
    else:
        text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = "\n".join(chunk for chunk in chunks if chunk)

    return text


# @st.cache(allow_output_mutation=True,show_spinner=False,show_spinner=False)
@st.cache(allow_output_mutation=True, show_spinner=False)
def load_text_from_url(url, **data):

    """
    Loads html from a URL
    Parameters:
        * url: url of page to load (str)
        * timeout: request timeout in seconds (int) default: 20
    Returns:
        HTML (str)
    """

    timeout = data.get("timeout", 20)

    results = []

    try:

        # print("Extracting HTML from: {}".format(url))
        response = requests.get(
            url,
            headers={
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:68.0) Gecko/20100101 Firefox/68.0"
            },
            timeout=timeout,
        )

        text = response.text
        status = response.status_code

        if status == 200 and len(text) > 0:
            return text
        else:
            print("Incorrect status returned: ", status)

        return None

    except Exception as e:
        print("Problem with url: {0}.".format(url))
        return None


# @st.cache(allow_output_mutation=True,show_spinner=False,show_spinner=False)


@st.cache(allow_output_mutation=True, show_spinner=False)
def get_wikipedia_url(query):
    """
    Finds the closest matching Wikipedia page for a given string.
    Parameters:
        * query: Query to search Wikipedia with. (string)
    Returns:
       The top matching URL for the query.  Follows redirects (string)
    """
    sitew = pywikibot.Site("en", "wikipedia")
    result = None
    print("looking up:", query)
    search = sitew.search(
        query, where="title", get_redirects=True, total=1, content=False, namespaces="0"
    )
    for page in search:
        if page.isRedirectPage():
            page = page.getRedirectTarget()
        result = page.full_url()
        break

    return result


# @st.cache(allow_output_mutation=True,show_spinner=False,show_spinner=False)


@st.cache(allow_output_mutation=True, show_spinner=False)
def recurse_entities(
    input_data, entity_results=[], G=nx.Graph(), current_depth=0, depth=2, limit=3
):
    """
    Recursively finds entities of connected Wikipedia topics by taking the top entities
    for each page and following those entities up to the specified depth
    Parameters:
        * input_data: A topic or URL.  If topic, finds the closes matching Wikipedia start page.
                      If URL, starts with the top enetities of that page. (string)
        * depth: Max recursion depth (integer)
        * limit: The max number of entities to pull for each page. (integer)
    Returns:
       A tuple of:
        * entity_results: List of dictionaries of found entities.
        * G: Networkx graph of entities.
    """
    if isinstance(input_data, str):
        # Starting fresh.  Make sure variables are fresh.
        entity_results = []
        G = nx.Graph()
        current_depth = 0
        if not validators.url(input_data):
            input_data = get_wikipedia_url(input_data)
            if not input_data:
                print("No Wikipedia URL Found.")
                return None, None
            else:
                print("Wikipedia URL: ", input_data)
            name = load_page_title(input_data).split("-")[0].strip()
        else:
            name = load_page_title(input_data)
        input_data = (
            [
                {
                    "name": name.title(),
                    "type": "START",
                    "salience": 0.0,
                    "wikipedia": input_data,
                }
            ]
            if input_data
            else []
        )

    # Regex for wikipedia terms to not bias entities returned
    subs = r"(wikipedia|wikimedia|wikitext|mediawiki|wikibase)"

    for d in input_data:
        url = d["wikipedia"]
        name = d["name"]

        print(
            "   " * current_depth + "Level: {0} Name: {1}".format(current_depth, name)
        )

        html = load_text_from_url(url)

        # html_to_text will default to all text if < 4 `p` elements found.
        if "wikipedia.org" in url:
            html = html_to_text(html, target_elements="p")
        else:
            html = html_to_text(html)

        # Kill brutally wikipedia terms.
        html = re.sub(subs, "", html, flags=re.IGNORECASE)

        results = [
            r
            for r in google_nlp_entities(
                html, input_type="text", limit=None, result_type="wikipedia"
            )
            if "wiki" not in r["name"].lower() and not G.has_node(r["name"])
        ][:limit]
        _ = [G.add_edge(name, r["name"]) for r in results]
        entity_results.extend(results)

        new_depth = int(current_depth + 1)
        if results and new_depth <= depth:
            recurse_entities(results, entity_results, G, new_depth, depth, limit)

    if current_depth == 0:
        return entity_results, G


# @st.cache(allow_output_mutation=True,show_spinner=False,show_spinner=False)


@st.cache(allow_output_mutation=True, show_spinner=False)
def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):

    """
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    """
    if not nx.is_tree(G):
        raise TypeError("cannot use hierarchy_pos on a graph that is not a tree")

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(
                iter(nx.topological_sort(G))
            )  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(
        G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None
    ):
        """
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        """

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(
                    G,
                    child,
                    width=dx,
                    vert_gap=vert_gap,
                    vert_loc=vert_loc - vert_gap,
                    xcenter=nextx,
                    pos=pos,
                    parent=root,
                )
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


# @st.cache(allow_output_mutation=True,show_spinner=False)
# @st.cache(allow_output_mutation=True,show_spinner=False,show_spinner=False)
def plot_entity_branches(G, w=10, h=10, c=1, font_size=14, filename=None):
    """
    Given a networkx graph, builds a recursive tree graph

    Parameters:
        * G: Networkx graph of entities.
        * w: Width of output plot
        * h: height of output plot
        * c: Circle percentage (float) 0.5 is a semi-circle. Range: 0.1-1.0
        * font_size: Font Size of labels (integer)
        * filename: Filename for the saved plot.  Optional (string)
    Returns:
       Nothing. Plots a graph

    """
    start = list(G.nodes)[0]
    G = nx.bfs_tree(G, start)
    plt.figure(figsize=(w, h))
    pos = hierarchy_pos(G, start, width=float(2 * c) * math.pi, xcenter=0)
    new_pos = {
        u: (r * math.sin(theta), r * math.cos(theta)) for u, (theta, r) in pos.items()
    }
    nx.draw(
        G,
        pos=new_pos,
        alpha=0.8,
        node_size=25,
        with_labels=True,
        font_size=font_size,
        edge_color="gray",
    )
    nx.draw_networkx_nodes(
        G, pos=new_pos, nodelist=[start], node_color="blue", node_size=500
    )

    if filename:
        plt.savefig("{0}/{1}".format("images", filename))


st.set_option("deprecation.showPyplotGlobalUse", False)

# url = "https://en.wikipedia.org/wiki/Entity%E2%80%93relationship_model"
# url = "https://en.wikipedia.org/wiki/Michael_Jordan"
# html = load_text_from_url(url)
# data = google_nlp_entities(html, result_type="wikipedia")
# df_wiki_entities = pd.DataFrame(data)
# df_wiki_entities
#

# url = "https://en.wikipedia.org/wiki/Michael_Jordan"
# html = load_text_from_url(url)
# data2 = google_nlp_entities(html, result_type="all")
# df_wiki_entities2 = pd.DataFrame(data2)
# df_wiki_entities2

# st.markdown("---")
# st.markdown('## **② Set things up!**')
st.markdown("## **② Type URL or keyword 📝**")

# st.subheader('https://en.wikipedia.org/wiki/Diet_(nutrition)')

with st.beta_expander("ℹ️ - Details about depth, limits etc. ", expanded=False):

    st.write(
        """
    - https://en.wikipedia.org/wiki/Diet_(nutrition)
    -   add text: we recommend depth 1 or 2 as default - depth 3+ works but legilibilty issues may occur
    - 'Depth - 2 levels max for now, try 3')
    - 'Limit - 2 levels max for now, anything above  doesnt work')
    -  Add text: def recurse_entities(input_data, entity_results=[], G=nx.Graph(), current_depth=0, depth=2, limit=3):
    '''
    Recursively finds entities of connected Wikipedia topics by taking the top entities 
    for each page and following those entities up to the specified depth
    Parameters:
        * input_data: A topic or URL.  If topic, finds the closes matching Wikipedia start page.  
                      If URL, starts with the top enetities of that page. (string)
        * depth: Max recursion depth (integer)
        * limit: The max number of entities to pull for each page. (integer)
    Returns:
       A tuple of:
        * entity_results: List of dictionaries of found entities.
        * G: Networkx graph of entities.


	    """
    )

# st.subheader('aaaaaaaaaaaa')

# c200 = st.beta_container()

st.text("")

c10, c0, c8, c1, c2, c3, c4, c5, c6 = st.beta_columns(
    [0.10, 0.50, 0.10, 8, 0.10, 1.5, 0.10, 1.5, 0.10]
)


with c0:
    st.text("")
    toggle = st.select_slider("", options=("URL", "KW"))

with c1:

    from re import search

    substring = "http://|https://"

    if toggle == "KW":
        keyword = st.text_input("Enter a keyword", key=1)
        if keyword:
            if search(substring, keyword):
                st.warning(
                    "⚠️ Seems like you&#39re trying to paste a URL, switch to &#39URL&#39 mode!"
                )
            else:
                st.markdown('Keyword is "' + str(keyword) + '"')

    elif toggle == "URL":

        keyword = st.text_input(
            "Enter a Wikipedia URL",
            "https://en.wikipedia.org/wiki/Diet_(nutrition)",
            key=2,
        )

        if keyword:
            if search(substring, keyword):
                st.markdown('URL is "' + str(keyword) + '"')
            else:
                st.warning("⚠️ URL Format is invalid, please add http:// or https://")

with c3:
    depth = st.number_input("Depth", step=1, min_value=1, max_value=3, key=1)

with c5:
    limit = st.number_input("Limit", step=1, min_value=1, max_value=5, key=2)

c3, c4 = st.beta_columns(2)

with c3:
    st.text("")
    st.text("")
    c20 = st.beta_container()

with c4:
    st.text("")
    c30 = st.beta_container()

button1 = c20.button("✨ Happy with costs, get me the data!")

#st.stop()

######################

if not button1 and not uploaded_file:
    st.stop()
elif not button1 and uploaded_file:
    # c30.warning('◀️ Press button')
    st.stop()
elif button1 and not uploaded_file:
    # c30.warning('Press button')
    c.error("◀️ Add credentials 1st")
    st.stop()
else:
    pass

if button1:

    import time

    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
        latest_iteration.markdown(f"Sending your request ({i+1} % Completed)")
        bar.progress(i + 1)
        time.sleep(0.05)

data, G = recurse_entities(keyword, depth=depth, limit=limit)

# st.markdown('## **② Choose URL or keyword **')

st.markdown("## **③ Check results! 🙌  **")

# with st.beta_expander("Collapse/Expand table ", expanded=True):
with st.beta_expander("Hide/Show table  ", expanded=True):

    st.text("")

    c30, c31, c32 = st.beta_columns(3)

    with c30:
        c1 = st.beta_container()
    with c31:
        c2 = st.beta_container()

    cm = sns.light_palette("green", as_cmap=True)
    # df = pd.DataFrame(data)
    df = pd.DataFrame(data).sort_values(by="salience", ascending=False)

    df = df.reset_index()

    df.index += 1
    df = df.drop(["index"], axis=1)
    format_dictionary = {
        "salience": "{:.1%}",
    }
    dfStyled = df.style.background_gradient(cmap=cm)
    # dfStyled = df['salience'].style.background_gradient(cmap=cm)

    # three = c2.multiselect('Multiselect', ['PERSON','ORGANIZATION'],['PERSON','ORGANIZATION'])
    # dfStyled = df[df['type'].isin(three)]
    # dfStyled = dfStyled.style.background_gradient(cmap=cm)

    dfStyled2 = dfStyled.format(format_dictionary)

    st.table(dfStyled2)

    try:
        import base64

        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        # st.markdown('## **▼ List view**')
        # st.subheader("")
        href = f'<a href="data:file/csv;base64,{b64}" download="listViewExport.csv">** - Download data to CSV 🎁 **</a>'
        c1.markdown(href, unsafe_allow_html=True)
    except NameError:
        print("wait")


# Pyplot - Matplotlib feature
# fig = plot_entity_branches(G, w=15, h=15, c=1, font_size=16)
# st.pyplot(fig)

# st.header('test')

# def read_json_file(filename):
#    with open(filename) as f:
#        js_graph = json.load(f)
#    return json_graph.node_link_graph(js_graph)

##G = read_json_file('/content/data.json')
# G = read_json_file('FootballJan2021.json')
# st.write(G)


g4 = net.Network(
    directed=True, height="1000px", width="1200px", notebook=True, heading="Football"
)
g4.from_nx(G)

# g4.show_buttons(filter_=['physics'])
g4.show("wikiOutput.html")
# display(HTML('karate.html'))

HtmlFile = open("karate.html", "r")
source_code = HtmlFile.read()
components.html(source_code, height=1000, width=800)

st.stop()


###### bokeh experiments  bokeh experiments  bokeh experiments  bokeh experiments
###### bokeh experiments  bokeh experiments  bokeh experiments  bokeh experiments

from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes

WikiNetworkx = hv.Graph.from_networkx(fig, nx.layout.circular_layout).opts(
    tools=["hover"]
)
st.bokeh_chart(hv.render(WikiNetworkx, backend="bokeh"))  # will work
karate_club_graph = hv.Graph.from_networkx(G, nx.layout.circular_layout).opts(
    tools=["hover"]
)
from bokeh.plotting import figure, from_networkx

st.stop()

G = nx.grid_2d_graph(5, 5)  # 5x5 grid

# print the adjacency list
for line in nx.generate_adjlist(G):
    print(line)
# write edgelist to grid.edgelist
nx.write_edgelist(G, path="grid.edgelist", delimiter=":")
# read edgelist from grid.edgelist
H = nx.read_edgelist(path="grid.edgelist", delimiter=":")

fig = nx.draw(H)
plt.show()
