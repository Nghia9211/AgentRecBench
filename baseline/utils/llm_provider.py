"""
utils/llm_provider.py
─────────────────────
Cung cấp LLM unified interface cho mọi file trong project.

Hỗ trợ:
  - ollama  : local server (ChatOllama / raw requests)
  - gpt     : OpenAI API (gpt-3.5-turbo, gpt-4o, ...)
  - groq    : Groq API (llama, mixtral, ...)

Cách dùng:
  # LangChain-based (dùng cho ARAGgcnRetrie, v.v.)
  from utils.llm_provider import get_langchain_llm
  llm = get_langchain_llm(provider="ollama", model="qwen2.5:7b-instruct-q5_K_M")

  # LLMBase-based (dùng cho WebSocietySimulator, v.v.)
  from utils.llm_provider import get_simulator_llm
  llm = get_simulator_llm(provider="gpt", model="gpt-3.5-turbo")

  # Từ argparse (tự động detect provider + model)
  from utils.llm_provider import add_llm_args, build_llm_from_args
  add_llm_args(parser)
  args = parser.parse_args()
  llm  = build_llm_from_args(args)          # trả về LangChain LLM
  llm  = build_llm_from_args(args, mode="simulator")  # trả về LLMBase
"""

import os
import logging
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ─── Default models theo provider ────────────────────────────────────────────
DEFAULT_MODELS = {
    "ollama": "qwen2.5:7b-instruct-q5_K_M",
    "gpt":    "gpt-3.5-turbo",
    "groq":   "llama-3.3-70b-versatile",
}

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# ═══════════════════════════════════════════════════════════════════════════════
#  A. LANGCHAIN LLMs  (dùng cho LangGraph / LangChain pipeline)
# ═══════════════════════════════════════════════════════════════════════════════

def get_langchain_llm(
    provider: str = "ollama",
    model: str = None,
    temperature: float = 0,
    **kwargs,
):
    """
    Trả về LangChain ChatModel tương ứng với provider.
    Tương thích với .invoke(), .batch(), .with_structured_output().
    """
    model = model or DEFAULT_MODELS[provider]

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model,
            base_url=kwargs.get("ollama_url", OLLAMA_BASE_URL),
            temperature=temperature,
            num_predict=kwargs.get("max_tokens", 2048),
            num_ctx=kwargs.get("num_ctx", 8192),
        )

    elif provider == "gpt":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=kwargs.get("api_key") or os.getenv("OPENAI_API_KEY"),
            temperature=temperature,
            max_tokens=kwargs.get("max_tokens", 2048),
        )

    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model,
            api_key=kwargs.get("api_key") or os.getenv("GROQ_API_KEY"),
            temperature=temperature,
        )

    else:
        raise ValueError(f"Unknown provider '{provider}'. Choose: ollama | gpt | groq")

def get_simulator_llm(
    provider: str = "ollama",
    model: str = None,
    **kwargs,
):
    """
    Trả về LLMBase instance tương ứng với provider.
    Tương thích với websocietysimulator.set_llm().
    """
    model = model or DEFAULT_MODELS[provider]

    if provider == "ollama":
        return _OllamaSimLLM(model=model, base_url=kwargs.get("ollama_url", OLLAMA_BASE_URL))
    elif provider == "gpt":
        return _GPTSimLLM(model=model, api_key=kwargs.get("api_key") or os.getenv("OPENAI_API_KEY"))
    elif provider == "groq":
        return _GroqSimLLM(model=model, api_key=kwargs.get("api_key") or os.getenv("GROQ_API_KEY"))
    else:
        raise ValueError(f"Unknown provider '{provider}'. Choose: ollama | gpt | groq")


class _OllamaSimLLM:
    """LLMBase-compatible wrapper cho Ollama."""
    def __init__(self, model: str, base_url: str):
        self.model    = model
        self.base_url = base_url.rstrip("/")

    def __call__(self, messages, temperature=0.1, max_tokens=1500, **kwargs) -> str:
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model":    self.model,
                    "messages": messages,
                    "stream":   False,
                    "options":  {"temperature": temperature, "num_predict": max_tokens},
                },
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except requests.exceptions.Timeout:
            logger.error("OllamaLLM: request timed out")
            return ""
        except Exception as e:
            logger.error(f"OllamaLLM error: {e}")
            return ""

class _GPTSimLLM:
    """LLMBase-compatible wrapper cho OpenAI GPT."""
    def __init__(self, model: str, api_key: str):
        from openai import OpenAI
        self.model  = model
        self.client = OpenAI(api_key=api_key)

    def __call__(self, messages, temperature=0.1, max_tokens=1500, **kwargs) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"GPTLLM error: {e}")
            return ""


class _GroqSimLLM:
    """LLMBase-compatible wrapper cho Groq."""
    def __init__(self, model: str, api_key: str):
        from groq import Groq
        self.model  = model
        self.client = Groq(api_key=api_key)

    def __call__(self, messages, temperature=0.1, max_tokens=1500, **kwargs) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"GroqLLM error: {e}")
            return ""

def add_llm_args(parser):
    """Thêm các argument LLM vào ArgumentParser có sẵn."""
    group = parser.add_argument_group("LLM options")
    group.add_argument("--provider",    default="ollama", choices=["ollama", "gpt", "groq"],
                       help="LLM provider")
    group.add_argument("--model",       default=None,
                       help="Model name (tự động chọn mặc định nếu bỏ trống)")
    group.add_argument("--ollama_url",  default=OLLAMA_BASE_URL,
                       help="Ollama server URL")
    return parser


def build_llm_from_args(args, mode: str = "langchain"):
    """
    Build LLM từ parsed args.
    mode = 'langchain'  → trả về ChatModel (cho LangChain pipeline)
    mode = 'simulator'  → trả về LLMBase   (cho WebSocietySimulator)
    """
    model = args.model or DEFAULT_MODELS[args.provider]
    kwargs = {"ollama_url": getattr(args, "ollama_url", OLLAMA_BASE_URL)}

    logger.info(f"✅ LLM provider={args.provider}, model={model}, mode={mode}")

    if mode == "langchain":
        return get_langchain_llm(provider=args.provider, model=model, **kwargs)
    elif mode == "simulator":
        return get_simulator_llm(provider=args.provider, model=model, **kwargs)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose: langchain | simulator")