from app.services.utils import ServerUtils

from .process import ProcessController

class QuestionContext:
    def __init__(self, index, documents, llm_chain, model_name, repo_name, github_url, conversation_history, file_type_counts, filenames):
        self.index = index
        self.documents = documents
        self.llm_chain = llm_chain
        self.model_name = model_name
        self.repo_name = repo_name
        self.github_url = github_url
        self.conversation_history = conversation_history
        self.file_type_counts = file_type_counts
        self.filenames = filenames
        
        
class QuestionController:
    def __init__(self):
        pass
    
    @staticmethod  
    async def ask_question(question, context: QuestionContext):
        print(question)
        relevant_docs = await ProcessController.search_documents(question, context.index, context.documents, n_results=5)

        numbered_documents = ServerUtils.format_documents(relevant_docs)
        question_context = f"This question is about the GitHub repository '{context.repo_name}' available at {context.github_url}. The most relevant documents are:\n\n{numbered_documents}"

        answer_with_sources = context.llm_chain.run(
            model=context.model_name,
            question=question,
            user_input=question,
            context=question_context,
            repo_name=context.repo_name,
            github_url=context.github_url,
            conversation_history=context.conversation_history,
            numbered_documents=numbered_documents,
            file_type_counts=context.file_type_counts,
            filenames=context.filenames
        )
        return answer_with_sources

