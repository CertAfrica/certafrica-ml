from __future__ import annotations

from app.ai.services.ai_layer_service import AILayerService
from app.ai.utils.demo_payload import demo_ml_payload


if __name__ == "__main__":
    service = AILayerService()
    result = service.interpret(demo_ml_payload())
    print(result.model_dump_json(indent=2))
