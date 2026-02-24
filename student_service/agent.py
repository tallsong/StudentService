from google.adk.agents.llm_agent import Agent
from typing import Any, Dict, cast
from student_service.rag import WebsiteRAG

website_rag = WebsiteRAG(base_url="https://xu-university.com/")


def search_xu_university_knowledge(question: str) -> Dict[str, Any]:
    """
    Retrieve relevant information from xu-university.com knowledge base.
    
    The knowledge base indexes 170+ HTML pages and 30+ PDF documents including:
    - All bachelor and master program pages
    - Admissions requirements and application process
    - Program fees and structure
    - Student life, career services, and testimonials
    - Executive education offerings
    
    Returns JSON with status, sources (URL, title, content snippets), and guidance.
    """
    return cast(Dict[str, Any], website_rag.query(question))



root_agent = Agent(
    model='gemini-2.5-flash',
    name='root_agent',
    description="A helpful assistant for Students' questions.",
    instruction=(
        "You are a student support assistant with access to a comprehensive knowledge base "
        "covering all xu-university.com content (170+ pages, 30+ PDFs). "
        "\n\n"
        "For ANY question about XU University, ALWAYS call search_xu_university_knowledge first. "
        "Answer using only the retrieved sources—cite specific source_url values. "
        "If information is missing or you're uncertain, clearly state what's unknown. "
        "\n\n"
        "The knowledge base includes:\n"
        "- Bachelor & Master program details (curricula, specializations, career paths)\n"
        "- Admissions requirements, application process, enrollment steps\n"
        "- Program fees, payment plans, scholarships\n"
        "- Student life, housing, campus facilities\n"
        "- Career services and job placement\n"
        "- Executive education programs"
    ),
    tools=[search_xu_university_knowledge],
)
