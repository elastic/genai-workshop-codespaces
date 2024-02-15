import os
import streamlit as st
import openai
import tiktoken
import secrets
from openai import OpenAI
from elasticsearch import Elasticsearch
import elasticapm
import base64
from elasticsearch_llm_cache.elasticsearch_llm_cache import ElasticsearchLLMCache
import time
import json
import textwrap

######################################
# Streamlit Configuration
st.set_page_config(layout="wide")


# wrap text when printing, because colab scrolls output to the right too much
def wrap_text(text, width):
    wrapped_text = textwrap.wrap(text, width)
    return '\n'.join(wrapped_text)


@st.cache_data()
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


set_background('images/background-dark2.jpeg')


######################################

######################################
# Sidebar Options
def sidebar_bg(side_bg):
    side_bg_ext = 'png'
    st.markdown(
        f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
      </style>
      """,
        unsafe_allow_html=True,
    )


side_bg = './images/sidebar2-dark.png'
sidebar_bg(side_bg)

# sidebar logo
st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)

with st.sidebar:
    st.image("images/elastic_logo_transp_100.png")

######################################
# expander markdown
st.markdown(
    '''
    <style>
    .streamlit-expanderHeader {
        background-color: gray;
        color: black; # Adjust this for expander header color
    }
    .streamlit-expanderContent {
        background-color: white;
        color: black; # Expander content color
    }
    </style>
    ''',
    unsafe_allow_html=True
)

######################################

@st.cache_resource
def initOpenAI():
    #if using the Elastic AI proxy, then generate the correct API key
    if os.environ['ELASTIC_PROXY'] == "True":
        #generate and share "your" unique hash
        os.environ['USER_HASH'] = secrets.token_hex(nbytes=6)
        print(f"Your unique user hash is: {os.environ['USER_HASH']}")
        #get the current API key and combine with your hash
        os.environ['OPENAI_API_KEY'] = f"{os.environ['OPENAI_API_KEY']} {os.environ['USER_HASH']}"
    else:
        openai.api_type = os.environ['OPENAI_API_TYPE']
        openai.api_version = os.environ['OPENAI_API_VERSION']

    # Configure OpenAI client
    openai.api_key = os.environ['OPENAI_API_KEY']
    openai.api_base = os.environ['OPENAI_API_BASE']
    openai.default_model = os.environ['OPENAI_API_ENGINE']
    openai.verify_ssl_certs = False
    client = OpenAI(api_key=openai.api_key, base_url=openai.api_base)
    return client

openAIClient = initOpenAI()

# Initialize Elasticsearch and APM clients
# Configure APM and Elasticsearch clients
@st.cache_resource
def initElastic():
    os.environ['ELASTIC_APM_SERVICE_NAME'] = "genai_workshop_v2_lab_2-2"
    apmclient = elasticapm.Client()
    elasticapm.instrument()

    if 'ELASTIC_CLOUD_ID_W' in os.environ:
        es = Elasticsearch(
            cloud_id=os.environ['ELASTIC_CLOUD_ID_W'],
            api_key=(os.environ['ELASTIC_APIKEY_ID_W']),
            request_timeout=30
        )
    else:
        es = Elasticsearch(
            os.environ['ELASTIC_URL'],
            basic_auth=(os.environ['ELASTIC_USER'], os.environ['ELASTIC_PASSWORD']),
            request_timeout=30
        )

    return apmclient, es


apmclient, es = initElastic()

# Set our data index
index = os.environ['ELASTIC_INDEX_DOCS_W']

###############################################################
# Similarity Cache functions
# move to env if time
cache_index = "wikipedia-cache"


def clear_es_cache(es):
    print('clearing cache')
    match_all_query = {"query": {"match_all": {}}}
    clear_response = es.delete_by_query(index=cache_index, body=match_all_query)
    return clear_response


@elasticapm.capture_span("cache_search")
def cache_query(cache, prompt_text, similarity_threshold=0.5):
    hit = cache.query(prompt_text=prompt_text, similarity_threshold=similarity_threshold)

    if hit:
        st.sidebar.markdown('`Cache Match Found`')
    else:
        st.sidebar.markdown('`Cache Miss`')

    return hit


@elasticapm.capture_span("add_to_cache")
def add_to_cache(cache, prompt, response):
    st.sidebar.markdown('`Adding response to cache`')
    print('adding to cache')
    print(prompt)
    print(response)
    resp = cache.add(prompt=prompt, response=response)
    st.markdown(resp)
    return resp


def init_elastic_cache():
    # Init Elasticsearch Cache
    # Only want to attempt to create the index on first run
    cache = ElasticsearchLLMCache(es_client=es,
                                  index_name=cache_index,
                                  create_index=False  # setting only because of Streamlit behavior
                                  )
    st.sidebar.markdown('`creating Elasticsearch Cache`')

    if "index_created" not in st.session_state:

        st.sidebar.markdown('`running create_index`')
        cache.create_index(768)

        # Set the flag so it doesn't run every time
        st.session_state.index_created = True
    else:
        st.sidebar.markdown('`index already created, skipping`')

    return cache


def calc_similarity(score, func_type='dot_product'):
    if func_type == 'dot_product':
        return (score + 1) / 2
    elif func_type == 'cosine':
        return (1 + score) / 2
    elif func_type == 'l2_norm':
        return 1 / (1 + score ^ 2)
    else:
        return score


###############################################################


def get_bm25_query(query_text, augment_method):
    if augment_method == "Full Text":
        return {
            "match": {
                "text": query_text
            }
        }
    elif augment_method == "Matching Chunk":
        return {
            "nested": {
                "path": "passages",
                "query": {
                    "bool": {
                        "must": [
                            {
                                "match": {
                                    "passages.text": query_text
                                }
                            }
                        ]
                    }
                },
                "inner_hits": {
                    "_source": False,
                    "fields": [
                        "passages.text"
                    ]
                }

            }
        }


# Run an Elasticsearch query using BM25 relevance scoring
@elasticapm.capture_span("bm25_search")
def search_bm25(query_text,
                es,
                size=1,
                augment_method="Full Text",
                use_hybrid=False  # always false - use semantic opt for hybrid
                ):
    fields = [
        "text",
        "title",
    ]

    resp = es.search(index=index,
                     query=get_bm25_query(query_text, augment_method),
                     fields=fields,
                     size=size,
                     source=False)
    # print(resp)
    body = resp
    url = 'nothing'

    return body, url


@elasticapm.capture_span("knn_search")
def search_knn(query_text,
               es,
               size=1,
               augment_method="Full Text",
               use_hybrid=False
               ):
    fields = [
        "title",
        "text"
    ]

    knn = {
        "inner_hits": {
            "_source": False,
            "fields": [
                "passages.text"
            ]
        },
        "field": "passages.embeddings",
        "k": size,
        "num_candidates": 100,
        "query_vector_builder": {
            "text_embedding": {
                "model_id": "sentence-transformers__all-distilroberta-v1",
                "model_text": query_text
            }
        }
    }

    rank = {"rrf": {}} if use_hybrid else None

    # need to get the bm25 query if we are using hybrid
    if use_hybrid:
        print('using hybrid with augment method %s' % augment_method)
        query = get_bm25_query(query_text, augment_method)
        print(query)
        if augment_method == "Matching Chunk":
            del query['nested']['inner_hits']
    else:
        print('not using hybrid')
        query = None

    print(query)
    print(knn)

    resp = es.search(index=index,
                     knn=knn,
                     query=query,
                     fields=fields,
                     size=size,
                     rank=rank,
                     source=False)

    return resp, None


def truncate_text(text, max_tokens):
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text

    return ' '.join(tokens[:max_tokens])


def build_text_obj(resp, aug_method):

    tobj = {}

    for hit in resp['hits']['hits']:
        # tobj[hit['fields']['title'][0]] = []
        title = hit['fields']['title'][0]
        tobj.setdefault(title, [])

        if aug_method == "Matching Chunk":
            print('hit')
            print(hit)
            # tobj['passages'] = []
            for ihit in hit['inner_hits']['passages']['hits']['hits']:
                tobj[title].append(
                    {'passage': ihit['fields']['passages'][0]['text'][0],
                     '_score': ihit['_score']}
                )
        elif aug_method == "Full Text":
            tobj[title].append(
                hit['fields']
            )

    return tobj


def generate_response(query,
                      es,
                      search_method,
                      custom_prompt,
                      negative_response,
                      show_prompt, size=1,
                      augment_method="Full Text",
                      use_hybrid=False,
                      show_es_response=True,
                      show_es_augment=True,
                      ):

    # Perform the search based on the specified method
    search_functions = {
        'bm25': {'method': search_bm25, 'display': 'Lexical Search'},
        'knn': {'method': search_knn, 'display': 'Semantic Search'}
    }
    search_func = search_functions.get(search_method)['method']
    if not search_func:
        raise ValueError(f"Invalid search method: {search_method}")

    # Perform the search and format the docs
    response, url = search_func(query, es, size, augment_method, use_hybrid)
    es_time = time.time()
    augment_text = build_text_obj(response, augment_method)

    res_col1, res_col2 = st.columns(2)
    # Display the search results from ES
    with res_col2:
        st.header(':rainbow[Elasticsearch Response]')
        st.subheader(':orange[Search Settings]')
        st.write(':gray[Search Method:] :blue[%s]' % search_functions.get(search_method)['display'])
        st.write(':gray[Size Setting:] :blue[%s]' % size)
        st.write(':gray[Augment Setting:] :blue[%s]' % augment_method)
        st.write(':gray[Using Hybrid:] :blue[%s]' % (
            'Not Applicable with Lexical' if search_method == 'bm25' else use_hybrid))

        st.subheader(':green[Augment Chunk(s) from Elasticsearch]')
        if show_es_augment:
            st.json(dict(augment_text))
        else:
            st.write(':blue[Show Augment Disabled]')

        st.subheader(':violet[Elasticsearch Response]')
        if show_es_response:
            st.json(dict(response))
        else:
            st.write(':blue[Response Received]')

    formatted_prompt = custom_prompt.replace("$query", query).replace("$response", str(augment_text)).replace(
        "$negResponse", negative_response)

    with res_col1:
        st.header(':orange[GenAI Response]')

        chat_response = chat_gpt(formatted_prompt, system_prompt="You are a helpful assistant.")

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for chunk in chat_response.split():
                full_response += chunk + " "
                time.sleep(0.02)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

    # Display results
    if show_prompt:
        st.text("Full prompt sent to ChatGPT:")
        st.text(wrap_text(formatted_prompt, 70))

    if negative_response not in chat_response:
        pass
    else:
        chat_response = None

    return es_time, chat_response

def count_tokens(messages, model="gpt-35-turbo"):
    if "gpt-3.5-turbo" in model or "gpt-35-turbo" in model:
        model = "gpt-3.5-turbo-0613"
    elif "gpt-4" in model:
        model="gpt-4-0613"

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using gpt-3.5-turbo-0613 encoding.")
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0613")

    if isinstance(messages, str):
        return len(encoding.encode(messages))
    else:
        tokens_per_message = 3
        tokens_per_name = 1

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

def chat_gpt(user_prompt, system_prompt):
    """
    Generates a response from ChatGPT based on the given user and system prompts.
    """
    max_tokens = 1024
    max_context_tokens = 4000
    safety_margin = 5

    # Truncate the prompt content to fit within the model's context length
    truncated_prompt = truncate_text(user_prompt, max_context_tokens - max_tokens - safety_margin)

    # Prepare the messages for the ChatGPT API
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": truncated_prompt}]

    full_response = ""
    for response in openAIClient.chat.completions.create(
        model=openai.default_model,
        temperature=0,
        messages=messages,
        stream=True
    ):
        full_response += (response.choices[0].delta.content or "")

    # APM: add metadata labels of data we want to capture
    elasticapm.label(model = openai.default_model)
    elasticapm.label(prompt = user_prompt)
    elasticapm.label(prompt_tokens = count_tokens(messages, model=openai.default_model))
    elasticapm.label(response_tokens = count_tokens(full_response, model=openai.default_model))
    elasticapm.label(total_tokens = count_tokens(messages, model=openai.default_model) + count_tokens(full_response, model=openai.default_model))
    if 'USER_HASH' in os.environ: elasticapm.label(user = os.environ['USER_HASH'])

    return full_response


# Main chat form
st.title("Wikipedia RAG Demo Platform")

# Define the default prompt and negative response
default_prompt_intro = "Answer this question:"
default_response_instructions = ("using only the information from the wikipedia documents included and nothing "
                                 "else.\nwikipedia_docs: $response\n")
default_negative_response = ("If the answer is not provided in the included documentation. You are to ONLY reply with "
                             "'I'm unable to answer the question based on the information I have from wikipedia' and "
                             "nothing else.")

with st.form("chat_form"):
    query = st.text_input("Ask the Elastic Documentation a question:",
                          placeholder='Who is Batman?')

    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        with st.expander("Customize Prompt Template"):
            prompt_intro = st.text_area("Introduction/context of the prompt:", value=default_prompt_intro)
            prompt_query_placeholder = st.text_area("Placeholder for the user's query:", value="$query")
            prompt_response_placeholder = st.text_area("Placeholder for the Elasticsearch response:",
                                                       value=default_response_instructions)
            prompt_negative_response = st.text_area("Negative response placeholder:", value=default_negative_response)
            prompt_closing = st.text_area("Closing remarks of the prompt:",
                                          value="Format the answer in complete markdown code format.")

            combined_prompt = f"{prompt_intro}\n{prompt_query_placeholder}\n{prompt_response_placeholder}\n{prompt_negative_response}\n{prompt_closing}"
            st.text_area("Preview of your custom prompt:", value=combined_prompt, disabled=True)

    with opt_col2:
        with st.expander("Retrieval Search and Display Options"):
            st.subheader("Retrieval Options")
            ret_1, ret_2 = st.columns(2)
            with ret_1:
                search_method = st.radio("Search Method", ("Semantic Search", "Lexical Search"))
                augment_method = st.radio("Augment Method", ("Full Text", "Matching Chunk"))
            with ret_2:
                # TODO this should update the title based on the augment_method
                doc_count_title = "Number of docs or chunks to Augment with" if augment_method == "Full Text" else "Number of Matching Chunks to Retrieve"
                doc_count = st.slider(doc_count_title, min_value=1, max_value=5, value=1)

                use_hybrid = st.checkbox('Use Hybrid Search')

            st.divider()

            st.subheader("Display Options")
            show_es_augment = st.checkbox('Show Elasticsearch Augment Text', value=True)
            show_es_response = st.checkbox('Show Elasticsearch Response', value=True)
            show_full_prompt = st.checkbox('Show Full Prompt Sent to LLM')

            st.divider()

            st.subheader("Caching Options")
            cache_1, cache_2 = st.columns(2)
            with cache_1:
                use_cache = st.checkbox('Use Similarity Cache')
                # Slider for adjusting similarity threshold
                similarity_threshold_selection = st.slider(
                    "Select Similarity Threshold (dot_product - Higher Similarity means closer)",
                    min_value=0.0, max_value=2.0,
                    value=0.5, step=0.01)

            with cache_2:
                clear_cache_butt = st.form_submit_button(':red[Clear Similarity Cache]')

    col1, col2 = st.columns(2)
    with col1:
        answer_button = st.form_submit_button("Find my answer!")

# Clear Cache Button
if clear_cache_butt:
    st.session_state.clear_cache_clicked = True

# Confirmation step
if st.session_state.get("clear_cache_clicked", False):
    apmclient.begin_transaction("clear_cache")
    elasticapm.label(action="clear_cache")

    # Start timing
    start_time = time.time()

    if st.button(":red[Confirm Clear Cache]"):
        print('clear cache clicked')
        # TODO if index doesn't exist, catch exception then create it
        response = clear_es_cache(es)
        st.success("Cache cleared successfully!", icon="🤯")
        st.session_state.clear_cache_clicked = False  # Reset the state

    apmclient.end_transaction("clear_cache", "success")

if answer_button:
    search_method = "knn" if search_method == "Semantic Search" else "bm25"

    apmclient.begin_transaction("query")
    elasticapm.label(search_method=search_method)
    elasticapm.label(query=query)

    # Start timing
    start_time = time.time()

    if use_cache:
        cache = init_elastic_cache()

        # check the llm cache first
        st.sidebar.markdown('`Checking ES Cache`')
        cache_check = cache_query(cache,
                                  prompt_text=query,
                                  similarity_threshold=similarity_threshold_selection
                                  )
        # st.markdown(cache_check)
    else:
        cache_check = None
        st.sidebar.markdown('`Skipping ES Cache`')

    try:

        if cache_check:
            es_time = time.time()
            st.sidebar.markdown('`cache match, using cached results`')
            st.subheader('Response from Cache')
            s_score = calc_similarity(cache_check['_score'], func_type='dot_product')
            st.code(f"Similarity Value: {s_score:.5f}")

            # Display response from LLM
            st.header('LLM Response')
            # st.markdown(cache_check['response'][0])
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for chunk in cache_check['response'][0].split():
                    full_response += chunk + " "
                    time.sleep(0.02)
                    # Add a blinking cursor to simulate typing
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

            llmAnswer = None  # no need to recache the answer

        else:
            # Use combined_prompt and show_full_prompt as arguments
            es_time, llmAnswer = generate_response(query,
                                                   es,
                                                   search_method,
                                                   combined_prompt,
                                                   prompt_negative_response,
                                                   show_full_prompt,
                                                   doc_count,
                                                   augment_method,
                                                   use_hybrid,
                                                   show_es_response,
                                                   show_es_augment,
                                                   )
        apmclient.end_transaction("query", "success")

        if use_cache and llmAnswer:
            if "I'm unable to answer the question" in llmAnswer:
                st.sidebar.markdown('`unable to answer, not adding to cache`')
            else:
                st.sidebar.markdown('`adding prompt and response to cache`')
                add_to_cache(cache, query, llmAnswer)

        # End timing and print the elapsed time
        elapsed_time = time.time() - start_time
        es_elapsed_time = es_time - start_time

        ct1, ct2 = st.columns(2)
        with ct1:
            st.subheader("GenAI Time taken: :red[%.2f seconds]" % elapsed_time)

        with ct2:
            st.subheader("ES Query Time taken: :green[%.2f seconds]" % es_elapsed_time)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        apmclient.end_transaction("query", "failure")
