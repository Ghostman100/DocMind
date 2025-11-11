"""
Суммаризация и извлечение ключевых пунктов из документов через LangChain
"""

from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any, Literal
import logging
import langchain
import tiktoken
import math

from ..config import settings
from .vector_store import get_langchain_vector_store

logger = logging.getLogger(__name__)


def _get_llm():
    """
    Получить LLM для суммаризации

    Returns:
        LLM instance
    """
    if settings.llm_provider == "openai":
        llm_kwargs = {
            "model": settings.llm_model,
            "temperature": settings.llm_temperature,
            "api_key": settings.llm_api_key,
            "verbose": settings.langchain_debug  # Включить verbose режим если debug активен
        }

        # Добавить custom base_url если указан
        if settings.llm_base_url:
            llm_kwargs["base_url"] = settings.llm_base_url

        return ChatOpenAI(**llm_kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


def _create_summarization_chain(strategy: Literal["stuff", "map_reduce", "refine"]):
    """
    Создать цепочку суммаризации

    Args:
        strategy: Стратегия суммаризации

    Returns:
        Summarization chain
    """
    llm = _get_llm()

    # Промпты на русском языке
    if strategy == "stuff":
        # Для коротких документов - все в один промпт
        prompt_template = """Создайте подробное резюме следующего текста на русском языке:

{text}

РЕЗЮМЕ:"""
        prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
        chain = load_summarize_chain(
            llm=llm,
            chain_type="stuff",
            prompt=prompt,
            verbose=settings.langchain_debug
        )

    elif strategy == "map_reduce":
        # Для длинных документов - сначала суммаризация каждого чанка, потом объединение
        map_template = """Создайте краткое резюме следующего фрагмента текста на русском языке:

{text}

КРАТКОЕ РЕЗЮМЕ:"""
        map_prompt = PromptTemplate(template=map_template, input_variables=["text"])

        combine_template = """Объедините следующие резюме в одно связное итоговое резюме на русском языке.
Сохраните все важные детали и структуру информации:

{text}

ИТОГОВОЕ РЕЗЮМЕ:"""
        combine_prompt = PromptTemplate(template=combine_template, input_variables=["text"])

        chain = load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=settings.langchain_debug
        )

    elif strategy == "refine":
        # Итеративное уточнение резюме
        question_template = """Создайте резюме следующего текста на русском языке:

{text}

РЕЗЮМЕ:"""
        question_prompt = PromptTemplate(template=question_template, input_variables=["text"])

        refine_template = """У вас есть существующее резюме:
{existing_answer}

Уточните его, добавив информацию из следующего фрагмента текста:
{text}

Если новый фрагмент не добавляет полезной информации, просто верните существующее резюме.

УТОЧНЕННОЕ РЕЗЮМЕ:"""
        refine_prompt = PromptTemplate(
            template=refine_template,
            input_variables=["existing_answer", "text"]
        )

        chain = load_summarize_chain(
            llm=llm,
            chain_type="refine",
            question_prompt=question_prompt,
            refine_prompt=refine_prompt,
            verbose=settings.langchain_debug
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return chain


def summarize_document(
        document_id: str,
        strategy: Literal["stuff", "map_reduce", "refine"] = "map_reduce",
        max_chunks: int = 100
) -> Dict[str, Any]:
    """
    Суммаризовать документ по document_id

    Args:
        document_id: UUID документа
        strategy: Стратегия суммаризации
        max_chunks: Максимальное количество чанков для обработки

    Returns:
        Dict с summary и метаданными
    """
    logger.info(f"Summarizing document {document_id} with strategy {strategy}")

    # Получить все чанки документа
    vector_store_service = get_langchain_vector_store()
    chunks_data = vector_store_service.get_all_documents_by_id(
        document_id=document_id,
        max_chunks=max_chunks
    )

    if not chunks_data:
        raise ValueError(f"No chunks found for document_id: {document_id}")

    logger.info(f"Retrieved {len(chunks_data)} chunks")

    # Преобразовать в LangChain Documents
    documents = [
        Document(
            page_content=chunk["page_content"],
            metadata=chunk["metadata"]
        )
        for chunk in chunks_data
    ]

    # Создать цепочку суммаризации
    chain = _create_summarization_chain(strategy)

    # Выполнить суммаризацию
    try:
        summary = chain.invoke(documents)

        # Разные стратегии возвращают разные форматы
        if isinstance(summary, dict):
            summary_text = summary.get("output_text", str(summary))
        else:
            summary_text = str(summary)

        logger.info(f"Summary completed, length: {len(summary_text)} characters")

        return {
            "summary": summary_text,
            "chunks_processed": len(chunks_data),
            "strategy": strategy,
            "document_id": document_id
        }

    except Exception as e:
        logger.error(f"Error during summarization: {e}", exc_info=True)
        raise


def extract_points_from_document(
        document_id: str,
        topics: List[str],
        chunks_per_topic: int = 3,
        summarize: bool = False
) -> Dict[str, Any]:
    """
    Извлечь ключевые пункты из документа по заданным темам

    Args:
        document_id: UUID документа
        topics: Список тем/вопросов для извлечения
        chunks_per_topic: Количество чанков на каждую тему
        summarize: Суммаризовать ли найденные чанки

    Returns:
        Dict с извлеченными пунктами
    """
    logger.info(f"Extracting points from document {document_id} for {len(topics)} topics")

    vector_store_service = get_langchain_vector_store()
    extracted_points = []
    total_chunks = 0

    for topic in topics:
        logger.info(f"Searching for topic: {topic}")

        # Поиск релевантных чанков для каждой темы
        results = vector_store_service.search(
            document_id=document_id,
            query=topic,
            top_k=chunks_per_topic
        )

        if not results:
            logger.warning(f"No results found for topic: {topic}")
            extracted_points.append({
                "topic": topic,
                "relevant_chunks": [],
                "summary": None
            })
            continue

        # Извлечь тексты чанков
        relevant_chunks = [r["page_content"] for r in results]
        total_chunks += len(relevant_chunks)

        # Опционально суммаризовать найденные чанки
        summary_text = None
        if summarize and relevant_chunks:
            try:
                # Использовать "stuff" стратегию для небольшого количества чанков
                documents = [
                    Document(page_content=chunk)
                    for chunk in relevant_chunks
                ]

                chain = _create_summarization_chain("stuff")
                summary_result = chain.invoke(documents)

                if isinstance(summary_result, dict):
                    summary_text = summary_result.get("output_text", str(summary_result))
                else:
                    summary_text = str(summary_result)

                logger.info(f"Summarized {len(relevant_chunks)} chunks for topic: {topic}")

            except Exception as e:
                logger.error(f"Error summarizing topic '{topic}': {e}")
                summary_text = None

        extracted_points.append({
            "topic": topic,
            "relevant_chunks": relevant_chunks,
            "summary": summary_text
        })

    logger.info(f"Extraction completed. Total chunks retrieved: {total_chunks}")

    return {
        "document_id": document_id,
        "extracted_points": extracted_points,
        "total_chunks_retrieved": total_chunks
    }


def _count_tokens(text: str, model_name: str = "gpt-4") -> int:
    """
    Подсчитать количество токенов в тексте

    Args:
        text: Текст для подсчета
        model_name: Название модели для токенизатора

    Returns:
        Количество токенов
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Если модель не найдена, используем cl100k_base (для GPT-4, GPT-3.5-turbo)
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    return len(tokens)


def _split_text_into_parts(text: str, num_parts: int) -> List[str]:
    """
    Разделить текст на примерно равные части

    Args:
        text: Текст для разделения
        num_parts: Количество частей

    Returns:
        Список частей текста
    """
    # Разделить текст на абзацы
    paragraphs = text.split('\n\n')

    # Если абзацев мало, просто разделим по символам
    if len(paragraphs) < num_parts:
        chars_per_part = len(text) // num_parts
        parts = []
        for i in range(num_parts):
            start = i * chars_per_part
            end = start + chars_per_part if i < num_parts - 1 else len(text)
            parts.append(text[start:end])
        return parts

    # Разделить абзацы на части
    paragraphs_per_part = len(paragraphs) // num_parts
    parts = []

    for i in range(num_parts):
        start_idx = i * paragraphs_per_part
        end_idx = start_idx + paragraphs_per_part if i < num_parts - 1 else len(paragraphs)
        part_paragraphs = paragraphs[start_idx:end_idx]
        parts.append('\n\n'.join(part_paragraphs))

    return parts


def summarize_text(
        text: str,
        target_tokens: int = 70000
) -> Dict[str, Any]:
    """
    Суммаризовать текст с автоматическим выбором стратегии

    Если текст ≤ 80 000 токенов - использует "stuff" за один запрос.
    Если текст > 80 000 токенов - делит на 2-3 части и делает несколько "stuff" сводок,
    затем объединяет их.

    Args:
        text: Текст для суммаризации
        target_tokens: Целевое количество токенов в резюме (по умолчанию 70000)

    Returns:
        Dict с summary и метаданными
    """
    logger.info(f"Starting text summarization, target tokens: {target_tokens}")

    # Подсчитать токены в исходном тексте
    input_tokens = _count_tokens(text)
    logger.info(f"Input text has {input_tokens} tokens")

    # Определить стратегию
    TOKEN_THRESHOLD = 80000

    if input_tokens <= TOKEN_THRESHOLD:
        return {
            "summary": text,
            "input_tokens": input_tokens,
            "output_tokens": input_tokens,
            "parts_processed": 1,
            "strategy_used": "nothing"
        }

    else:
        # Текст большой - разделить на части
        # Определить количество частей (2-3)
        num_parts = 2 if input_tokens <= 150000 else 3
        logger.info(f"Using multi-part 'stuff' strategy with {num_parts} parts")

        # Разделить текст
        text_parts = _split_text_into_parts(text, num_parts)
        logger.info(f"Split text into {len(text_parts)} parts")

        # Создать сводки для каждой части
        part_summaries = []
        tokens_per_part_summary = target_tokens // (num_parts * 2)  # Используем половину для промежуточных сводок

        for i, part in enumerate(text_parts):
            part_tokens = _count_tokens(part)
            logger.info(f"Processing part {i+1}/{num_parts}, tokens: {part_tokens}")

            prompt_template = f"""Создайте подробное резюме следующего фрагмента текста на русском языке.
Резюме должно содержать примерно {tokens_per_part_summary} токенов и сохранять все важные детали и факты.

Фрагмент текста:
{{text}}

РЕЗЮМЕ:"""
            prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

            llm = _get_llm()
            chain = load_summarize_chain(
                llm=llm,
                chain_type="stuff",
                prompt=prompt,
                verbose=settings.langchain_debug
            )

            documents = [Document(page_content=part)]
            summary_result = chain.invoke(documents)

            if isinstance(summary_result, dict):
                part_summary = summary_result.get("output_text", str(summary_result))
            else:
                part_summary = str(summary_result)

            part_summaries.append(part_summary)
            logger.info(f"Part {i+1} summary completed, tokens: {_count_tokens(part_summary)}")

        # Объединить сводки через конкатенацию (без дополнительной суммаризации)
        logger.info("Concatenating part summaries")

        final_summary = "\n\n".join([f"ЧАСТЬ {i+1}:\n{summary}" for i, summary in enumerate(part_summaries)])
        output_tokens = _count_tokens(final_summary)

        logger.info(f"Final summary completed (concatenation), output tokens: {output_tokens}")

        return {
            "summary": final_summary,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "parts_processed": num_parts,
            "strategy_used": f"multi-part stuff ({num_parts} parts) with concatenation"
        }


def apply_prompt_to_text(text: str, prompt: str) -> Dict[str, Any]:
    """
    Применить произвольный промпт к тексту через LLM

    Args:
        text: Текст для обработки
        prompt: Промпт с инструкциями для LLM

    Returns:
        Dict с результатом обработки
    """
    logger.info(f"Applying prompt to text, text length: {len(text)} chars")

    # Включить debug режим для отображения промптов и ответов
    if settings.langchain_debug:
        langchain.debug = True

    try:
        llm = _get_llm()

        # Подсчет токенов
        input_text = f"{prompt}\n\nДокумент:\n{text}"
        input_tokens = _count_tokens(input_text)

        logger.info(f"Input tokens: {input_tokens}")

        # Создать промпт с текстом
        full_prompt = f"{prompt}\n\nДокумент:\n{text}"

        # Вызвать LLM
        response = llm.invoke(full_prompt)

        # Извлечь текст ответа
        if hasattr(response, 'content'):
            result_text = response.content
        else:
            result_text = str(response)

        output_tokens = _count_tokens(result_text)

        logger.info(f"Prompt application completed, output tokens: {output_tokens}")

        return {
            "result": result_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }

    finally:
        if settings.langchain_debug:
            langchain.debug = False
