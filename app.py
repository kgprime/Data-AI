import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(page_title="Gemini Database Bot", layout="wide")
st.title("📊 Gemini AI Database & Web Search Assistant")

# 1. Read hidden cloud database credentials from Streamlit Secrets
SUPABASE_URI = st.secrets["SUPABASE_URL"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# 2. Establish connection to your cloud Supabase database
@st.cache_resource
def initialize_database():
    return SQLDatabase.from_uri(SUPABASE_URI)

db = initialize_database()

# 3. Initialize the FREE Google Gemini Model
# We use gemini-2.5-flash as it is lightning fast and free
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=GEMINI_API_KEY,
    temperature=0
)

# 4. Initialize the free live Web Search tool
web_search_tool = DuckDuckGoSearchRun()

# 5. Assemble the LangChain SQL Agent and hand it the search tool
agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    extra_tools=[web_search_tool],
    verbose=True,
    agent_type="openai-tools" # Gemini supports this standardized agent format smoothly
)

# 6. Build the Chat History UI Mechanism
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Display all past conversational text bubbles
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 7. Listen for new user input questions
if user_prompt := st.chat_input("Ask about inventory records or global web data..."):
    # Display the user's question bubble
    st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)
        
    # Generate the AI response bubble
    with st.chat_message("assistant"):
        with st.spinner("Gemini is analyzing tools and calculating data..."):
            # Run the agent to fetch the answer
            ai_response = agent_executor.run(user_prompt)
            
            # Extract only the plain human-readable text if it is returned as an object/list
            clean_text = ""
            if isinstance(ai_response, list):
                for part in ai_response:
                    if hasattr(part, 'text'):
                        clean_text += part.text
                    elif isinstance(part, dict) and "text" in part:
                        clean_text += part["text"]
                    else:
                        clean_text += str(part)
            elif hasattr(ai_response, 'content'):
                clean_text = ai_response.content
            else:
                clean_text = str(ai_response)
                
            # Clean up any leftover code formatting bracket remnants
            clean_text = clean_text.strip().lstrip("[").rstrip("]").strip()
            
            # Display the beautiful, clean output to your user interface
            st.markdown(clean_text)
            
            # Save the clean message to memory
            st.session_state.chat_messages.append({"role": "assistant", "content": clean_text})
