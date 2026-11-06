"""Method registry: service functions -> validated, Envelope-wrapped RPC methods.

A service function has the shape ``fn(ctx: ServiceContext, *, a: int, b: str | None = None)
-> SomeData``. ``Method.from_function`` derives a pydantic params model from the
keyword-only parameters, so the controller, MCP server and CLI all validate input the same
way. ``Method.call`` returns an ``Envelope`` dict:

* ``data``          – the returned model, minus ``coverage`` / ``observed_at`` /
                      ``warnings`` / ``next_cursor`` which are lifted to the envelope;
* ``summary``       – ``summarize(data)`` if the method has a summarizer, else the data's
                      ``summary`` field/method if present, else a generic line;
* ``ok=False``      – for any ``HpcLabError`` (typed ``error``), and for unexpected
                      exceptions as ``internal`` (details stay in the controller log).
"""

from __future__ import annotations

import inspect
import logging
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from hpclab.api.envelope import Coverage, Envelope, ErrorInfo
from hpclab.domain.clock import iso
from hpclab.domain.errors import ErrorCode, HpcLabError, InvalidArgument
from hpclab.services.context import ServiceContext

log = logging.getLogger("hpclab.registry")

_LIFTED = ("coverage", "observed_at", "warnings", "next_cursor")


def _params_model(fn: Callable[..., Any], name: str) -> type[BaseModel]:
    hints = typing.get_type_hints(fn)
    fields: dict[str, Any] = {}
    sig = inspect.signature(fn)
    for pname, p in sig.parameters.items():
        if pname == "ctx":
            continue
        if p.kind is not inspect.Parameter.KEYWORD_ONLY:
            raise TypeError(f"{fn.__qualname__}: parameter {pname!r} must be keyword-only")
        ann = hints.get(pname, Any)
        default = ... if p.default is inspect.Parameter.empty else p.default
        fields[pname] = (ann, default)
    model: type[BaseModel] = create_model(
        f"{name.title().replace('_', '').replace('.', '')}Params",
        __config__=ConfigDict(extra="forbid", arbitrary_types_allowed=True),
        **fields,
    )
    return model


def _return_model(fn: Callable[..., Any]) -> type[BaseModel] | None:
    ret = typing.get_type_hints(fn).get("return")
    if isinstance(ret, type) and issubclass(ret, BaseModel):
        return ret
    return None


@dataclass
class Method:
    name: str
    fn: Callable[..., Any]
    params_model: type[BaseModel]
    data_model: type[BaseModel] | None
    mutation: bool = False
    summarize: Callable[[Any], str] | None = None
    description: str = ""
    tags: tuple[str, ...] = ()

    @classmethod
    def from_function(
        cls,
        name: str,
        fn: Callable[..., Any],
        *,
        mutation: bool = False,
        summarize: Callable[[Any], str] | None = None,
        tags: tuple[str, ...] = (),
    ) -> Method:
        return cls(
            name=name,
            fn=fn,
            params_model=_params_model(fn, name),
            data_model=_return_model(fn),
            mutation=mutation,
            summarize=summarize,
            description=inspect.getdoc(fn) or "",
            tags=tags,
        )

    def validate(self, params: dict[str, Any] | None) -> dict[str, Any]:
        try:
            parsed = self.params_model.model_validate(params or {})
        except ValidationError as exc:
            raise InvalidArgument(
                f"invalid parameters for {self.name}: {exc.error_count()} error(s)",
                details={"errors": _compact_errors(exc)},
            ) from exc
        # keep pydantic models as models (services expect typed values)
        return {k: getattr(parsed, k) for k in parsed.model_fields_set | set(self.params_model.model_fields)}

    def call(self, ctx: ServiceContext, params: dict[str, Any] | None, *, request_id: str) -> dict[str, Any]:
        observed = iso(ctx.clock.now())
        try:
            kwargs = self.validate(params)
            result = self.fn(ctx, **kwargs)
            return envelope_from_result(
                result, request_id=request_id, summarize=self.summarize, observed_at=observed
            )
        except HpcLabError as exc:
            return error_envelope(exc, request_id=request_id)
        except Exception:
            log.exception("method %s failed (request %s)", self.name, request_id)
            return error_envelope(
                HpcLabError(
                    f"internal error in {self.name}; see the controller log (request {request_id})",
                    code=ErrorCode.INTERNAL,
                    details={"request_id": request_id},
                ),
                request_id=request_id,
            )


def _compact_errors(exc: ValidationError) -> list[dict[str, Any]]:
    out = []
    for e in exc.errors()[:20]:
        out.append(
            {"loc": ".".join(str(x) for x in e.get("loc", ())), "msg": e.get("msg"), "type": e.get("type")}
        )
    return out


def _generic_summary(data: Any) -> str:
    if isinstance(data, BaseModel):
        for attr in ("summary", "message"):
            v = getattr(data, attr, None)
            if callable(v):
                v = v()
            if isinstance(v, str) and v:
                return v
        return type(data).__name__.removesuffix("Data")
    return "ok"


def envelope_from_result(
    result: Any,
    *,
    request_id: str,
    summarize: Callable[[Any], str] | None = None,
    observed_at: str | None = None,
) -> dict[str, Any]:
    coverage: Coverage | None = None
    warnings: list[str] = []
    next_cursor: str | None = None
    data: Any = result
    if isinstance(result, BaseModel):
        coverage = getattr(result, "coverage", None) if "coverage" in type(result).model_fields else None
        if "observed_at" in type(result).model_fields and getattr(result, "observed_at", None):
            observed_at = result.observed_at  # type: ignore[attr-defined]
        if "warnings" in type(result).model_fields:
            warnings = list(getattr(result, "warnings", []) or [])
        if "next_cursor" in type(result).model_fields:
            next_cursor = getattr(result, "next_cursor", None)
        data = result.model_dump(mode="json", exclude=set(_LIFTED) & set(type(result).model_fields))
    try:
        summary = summarize(result) if summarize else _generic_summary(result)
    except Exception:  # a broken summarizer must never fail the call
        log.exception("summarizer failed")
        summary = _generic_summary(result)
    env = Envelope[Any](
        request_id=request_id,
        ok=True,
        data=data,
        coverage=coverage,
        observed_at=observed_at,
        next_cursor=next_cursor,
        warnings=warnings,
        summary=summary,
    )
    return env.model_dump(mode="json")


def error_envelope(exc: HpcLabError, *, request_id: str, data: Any = None) -> dict[str, Any]:
    env = Envelope[Any](
        request_id=request_id,
        ok=False,
        data=data,
        error=ErrorInfo.from_exc(exc),
        summary=f"{exc.code}: {exc.message}",
    )
    return env.model_dump(mode="json")


@dataclass
class Registry:
    methods: dict[str, Method] = field(default_factory=dict)

    def add(self, method: Method) -> None:
        if method.name in self.methods:
            raise ValueError(f"duplicate method {method.name}")
        self.methods[method.name] = method

    def register(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        mutation: bool = False,
        summarize: Callable[[Any], str] | None = None,
        tags: tuple[str, ...] = (),
    ) -> Method:
        m = Method.from_function(name, fn, mutation=mutation, summarize=summarize, tags=tags)
        self.add(m)
        return m

    def get(self, name: str) -> Method:
        try:
            return self.methods[name]
        except KeyError as exc:
            raise InvalidArgument(
                f"unknown method {name!r}", details={"known": sorted(self.methods)[:200]}
            ) from exc

    def dispatch(
        self, ctx: ServiceContext, name: str, params: dict[str, Any] | None, *, request_id: str
    ) -> dict[str, Any]:
        try:
            method = self.get(name)
        except HpcLabError as exc:
            return error_envelope(exc, request_id=request_id)
        return method.call(ctx, params, request_id=request_id)
