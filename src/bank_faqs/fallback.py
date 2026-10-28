# Chuỗi LLM dự phòng: model chính lỗi/hết quota thì thử lần lượt các model sau, có ghi log và tạm bỏ qua
# model vừa bị giới hạn, để mỗi câu hỏi không phải gọi lại model đã hết quota rồi mới chuyển.
import logging
import time
from langchain.agents.middleware import AgentMiddleware
from langgraph.errors import GraphBubbleUp
from bank_faqs.llm import is_daily_limit, is_rate_limited, model_name

log = logging.getLogger('uvicorn.error')

MINUTE_COOLDOWN_S = 60     # vượt giới hạn theo phút: bỏ qua model 1 phút
DAILY_COOLDOWN_S = 3600    # hết quota ngày: bỏ qua 1 giờ rồi thử lại (không biết chính xác giờ reset của từng hãng)

class FallbackMiddleware(AgentMiddleware):
    def __init__(self, *models, clock=time.monotonic):
        super().__init__()
        self.models = list(models)
        self.clock = clock
        self.skip_until: dict[str, float] = {}  # tên model -> thời điểm được thử lại

    def _chain(self, primary) -> list:
        chain = [primary, *self.models]
        now = self.clock()
        ready = [m for m in chain if self.skip_until.get(model_name(m), 0) <= now]
        return ready or chain  # mọi model đều đang bị bỏ qua thì vẫn thử hết, còn hơn báo lỗi ngay

    def _on_error(self, model, e: Exception) -> None:
        name = model_name(model)
        if is_rate_limited(e):
            wait = DAILY_COOLDOWN_S if is_daily_limit(e) else MINUTE_COOLDOWN_S
            self.skip_until[name] = self.clock() + wait
            log.warning('LLM %s bị giới hạn (%s), tạm bỏ qua %d phút', name, type(e).__name__, wait // 60)
        else:
            log.warning('LLM %s lỗi (%s: %s), chuyển model dự phòng', name, type(e).__name__, str(e)[:200])

    def _on_success(self, primary, model) -> None:
        if model is not primary:
            log.info('Trả lời bằng model dự phòng %s', model_name(model))
        self.skip_until.pop(model_name(model), None)

    def call(self, primary, invoke):
        # invoke(model) gọi 1 model; dùng trực tiếp được ngoài create_agent (vd graph.py)
        last_error = None
        for model in self._chain(primary):
            try:
                result = invoke(model)
            except GraphBubbleUp:
                raise
            except Exception as e:
                last_error = e
                self._on_error(model, e)
                continue
            self._on_success(primary, model)
            return result
        raise last_error

    async def acall(self, primary, ainvoke):
        last_error = None
        for model in self._chain(primary):
            try:
                result = await ainvoke(model)
            except GraphBubbleUp:
                raise
            except Exception as e:
                last_error = e
                self._on_error(model, e)
                continue
            self._on_success(primary, model)
            return result
        raise last_error

    def wrap_model_call(self, request, handler):
        return self.call(request.model, lambda m: handler(request if m is request.model else request.override(model=m)))

    async def awrap_model_call(self, request, handler):
        return await self.acall(request.model,
                                lambda m: handler(request if m is request.model else request.override(model=m)))
