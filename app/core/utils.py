from ..utils.logger import logger


def safe_lower(val):
    try:
        return str(val).lower().strip() if val is not None else ""
    except Exception as e:
        logger.error(
            "Failed to convert value to lowercase",
            error=str(e),
            exc_info=True,
            input_value={
                "value": repr(val),
                "type": type(val).__name__,
                "length": len(str(val)) if hasattr(val, "__len__") else None,
            },
            context={
                "function": "safe_lower",
                "purpose": "Convert input to lowercase string safely",
            },
        )
        return ""
