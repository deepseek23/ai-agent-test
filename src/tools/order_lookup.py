import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from src.config import (
    CANCELLATION_WINDOW_MINUTES,
    ORDERS_PATH,
    TERMINAL_NON_DELIVERY_STATUSES,
)


@dataclass
class OrderLookupResult:
    found: bool
    order_id: str
    data: dict | None = None
    error: str | None = None
    can_still_cancel: bool | None = None

    def to_tool_payload(self) -> dict:
        if not self.found:
            return {"found": False, "order_id": self.order_id, "error": self.error}

        d = self.data
        payload = {
            "found": True,
            "order_id": d["order_id"],
            "membership_tier": d["membership_tier"],
            "items": [
                {"name": i["name"], "quantity": i["quantity"], "final_sale": i["final_sale"]}
                for i in d["items"]
            ],
            "placed_at": d["placed_at"],
            "status": d["status"],
            "status_updated_at": d["status_updated_at"],
            "shipped_at": d["shipped_at"],
            "delivered_at": d["delivered_at"],
            "carrier": d["carrier"],
            "tracking_number": d["tracking_number"],
            "customer_safe_message": d["customer_safe_message"],
            "can_still_cancel": self.can_still_cancel,
        }

        if d["status"] in TERMINAL_NON_DELIVERY_STATUSES:
            payload["estimated_delivery"] = None
        else:
            payload["estimated_delivery"] = d["estimated_delivery"]

        return payload


class OrderLookupTool:
    def __init__(self, orders_path: str | Path):
        path = Path(orders_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Cannot find the orders database at: {path.absolute()}\n"
                "Check your folder structure."
            )

        raw = json.loads(path.read_text(encoding="utf-8"))
        self.snapshot_at = datetime.fromisoformat(raw["snapshot_at"].replace("Z", "+00:00"))
        self._orders_by_id = {o["order_id"]: o for o in raw["orders"]}

    @staticmethod
    def normalize_order_id(raw_id: str) -> str:
        return raw_id.strip().upper()

    def lookup(self, order_id: str | None) -> OrderLookupResult:
        if not order_id or not order_id.strip():
            return OrderLookupResult(found=False, order_id="", error="missing_id")

        normalized = self.normalize_order_id(order_id)
        order = self._orders_by_id.get(normalized)

        if order is None:
            return OrderLookupResult(found=False, order_id=normalized, error="not_found")

        return OrderLookupResult(
            found=True,
            order_id=normalized,
            data=order,
            can_still_cancel=self._can_still_cancel(order),
        )

    def _can_still_cancel(self, order: dict) -> bool:
        if order["status"] != "pending":
            return False
        placed_at = datetime.fromisoformat(order["placed_at"].replace("Z", "+00:00"))
        return (self.snapshot_at - placed_at) <= timedelta(minutes=CANCELLATION_WINDOW_MINUTES)


class OrderLookupInput(BaseModel):
    order_id: str = Field(
        description="The order ID, e.g. 'ORD-1007'. Case and surrounding whitespace do not matter."
    )


def make_order_lookup_tool(orders_path: str | Path = ORDERS_PATH) -> StructuredTool:
    order_lookup_tool = OrderLookupTool(orders_path)

    def _run_order_lookup(order_id: str) -> dict:
        result = order_lookup_tool.lookup(order_id)
        return json.dumps(result.to_tool_payload())

    return StructuredTool.from_function(
        func=_run_order_lookup,
        name="order_lookup",
        description=(
            "Look up the current status of a customer's order by order ID. "
            "Returns status, shipping info, and a customer-safe summary. "
            "Never returns customer PII or internal notes."
        ),
        args_schema=OrderLookupInput,
    )
