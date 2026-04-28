# --- Data Processing ---
from .data_processing.text_preprocessor import TextPreprocessor

# --- Analysis Engine (Module A) ---
from .analysis.classifier import NewsClassifier
from .analysis.sentiment_analyzer import SentimentAnalyzer
from .analysis.topic_modeler import TopicDiscoveryEngine
from .analysis.ner_extractor import EntityRelationshipMapper

# --- Language Models (Module B) ---
from .language_models.summarizer import IntelligentSummarizer
from .language_models.semantic_search import SemanticSearchEngine

# --- Multilingual Intelligence (Module C) ---
from .multilingual.language_detector import NewsLanguageDetector
from .multilingual.translator import NewsTranslator
from .multilingual.cross_lingual_analyzer import CrossLingualAnalyzer

# --- Conversational Interface (Module D) ---
from .conversation.query_processor import QueryProcessor