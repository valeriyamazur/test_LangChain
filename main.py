from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.document_loaders import AsyncChromiumLoader
from usp.tree import sitemap_tree_for_homepage

#Парсер карты сайта - собирает все url в список
def parse_sitemap(domain):
    list_pages = []
    domain = domain.strip()
    url = f'http://{domain}/'
    tree = sitemap_tree_for_homepage(url)
    for page in tree.all_pages():
        list_pages.append(page.url)
    return list_pages

#На всякий случай удаляем дубли
def get_unique_list_pages(list_pages):
    unique_list_pages = []
    for page in list_pages:
        if page in unique_list_pages:
            pass
        else:
            unique_list_pages.append(page)
    return unique_list_pages

#Собираем базу знаний
class KnowledgeBase():
    def __init__(self,
                 url: str,
                 chunk_size: int,
                 chunk_overlap: int):
        urls = get_unique_list_pages(parse_sitemap(domain=url))
        loader = AsyncChromiumLoader(urls)
        data = loader.load()
        doc_splitter = CharacterTextSplitter(
            chunk_size = chunk_size, chunk_overlap=chunk_overlap)
        docs = doc_splitter.split_documents(data)
        embeddings = OpenAIEmbeddings()
        docsearch = Chroma.from_documents(docs, embeddings)
        self.chain = RetrievalQAWithSourcesChain.from_chain_type(
            ChatOpenAI(openai_api_key="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"),
            chain_type='map_reduce',
            retriever=docsearch.as_retriever())

    #задаем боту вопрос, чтоб нашел его среди всех переданых ему url
    def Ask(self, query: str):
        return self.chain({'qestion': query}, return_only_outputs=True)




if __name__ == '__main__':
    kb = KnowledgeBase(
        url = 'rosexperts.ru',
        chunk_size = 8000,
        chunk_overlap = 3000)
    res = kb.Ask('Покажи мне список услуг')
