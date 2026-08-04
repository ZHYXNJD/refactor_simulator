"""Registry and factory for price-response models."""

from typing import Any, Dict, Mapping, Optional, Type

from .aggregate_elasticity import AggregateElasticityModel
from .base import PriceResponseModel
from .bounded_rational import BoundedRationalAgentModel
from .utility_choice import UtilityChoiceModel


MODEL_REGISTRY: Mapping[str, Type[PriceResponseModel]] = {
    AggregateElasticityModel.name: AggregateElasticityModel,
    UtilityChoiceModel.name: UtilityChoiceModel,
    BoundedRationalAgentModel.name: BoundedRationalAgentModel,
}

MODEL_ALIASES: Mapping[str, str] = {
    "aggregate": AggregateElasticityModel.name,
    "huang": AggregateElasticityModel.name,
    "utility": UtilityChoiceModel.name,
    "mo": UtilityChoiceModel.name,
    "bounded_rational": BoundedRationalAgentModel.name,
    "agent": BoundedRationalAgentModel.name,
    "xie": BoundedRationalAgentModel.name,
}


def create_price_response_model(
    model_name: str,
    parameters: Optional[Dict[str, Any]] = None,
    config: Optional[Any] = None,
) -> PriceResponseModel:
    """Create one standalone response model by canonical name or alias."""

    if parameters is not None and config is not None:
        raise TypeError("Use either parameters or config, not both")

    normalized_name = str(model_name).strip().lower()
    normalized_name = MODEL_ALIASES.get(normalized_name, normalized_name)
    if normalized_name not in MODEL_REGISTRY:
        valid_names = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown price response model '{model_name}'. Valid models: {valid_names}."
        )
    model_class = MODEL_REGISTRY[normalized_name]
    if config is not None:
        return model_class(config=config)
    return model_class(**(parameters or {}))


__all__ = ["MODEL_REGISTRY", "MODEL_ALIASES", "create_price_response_model"]
