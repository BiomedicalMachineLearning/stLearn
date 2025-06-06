"""Logging and Profiling"""

import logging
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from functools import partial, update_wrapper
from logging import CRITICAL, DEBUG, ERROR, INFO, WARNING
from typing import Any, overload

import anndata.logging

HINT = (INFO + DEBUG) // 2
logging.addLevelName(HINT, "HINT")


class CustomLogRecord(logging.LogRecord):
    """Custom root logger that maintains compatibility with standard logging
    interface."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_passed: timedelta | None = None
        self.deep: str | None = None


class _RootLogger(logging.RootLogger):
    def __init__(self, level):
        super().__init__(level)
        self.propagate = False
        _RootLogger.manager = logging.Manager(self)

    def log_with_timing(
        self,
        level: int,
        msg: str,
        *,
        extra: dict | None = None,
        time: datetime | None = None,
        deep: str | None = None,
    ) -> datetime:
        from . import settings

        now = datetime.now(timezone.utc)
        time_passed: timedelta | None = None if time is None else now - time
        extra = {
            **(extra or {}),
            "deep": deep if settings.verbosity.level < level else None,
            "time_passed": time_passed,
        }
        super().log(level, msg, extra=extra)
        return now

    def _handle_enhanced_logging(
        self, level: int, msg, *args, **kwargs
    ) -> datetime | None:
        """Handle logging with enhanced features (timing, deep info) or fall back to
        standard logging."""
        if "time" in kwargs or "deep" in kwargs or "extra" in kwargs:
            # Extract enhanced arguments
            time_arg = kwargs.pop("time", None)
            deep_arg = kwargs.pop("deep", None)
            extra_arg = kwargs.pop("extra", None)

            # Format message if there are remaining args
            if args or kwargs:
                formatted_msg = msg % args if args else msg
            else:
                formatted_msg = msg

            return self.log_with_timing(
                level, formatted_msg, time=time_arg, deep=deep_arg, extra=extra_arg
            )
        else:
            super().log(level, msg, *args, **kwargs)
            return None

    def hint(
        self,
        msg,
        *,
        time: datetime | None = None,
        deep: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> datetime:
        return self.log_with_timing(HINT, msg, time=time, deep=deep, extra=extra)

    @overload
    def debug(
        self,
        msg: object,
        *args: object,
        exc_info: bool | tuple | BaseException | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
    ) -> None: ...

    @overload
    def debug(self, msg, *args, **kwargs): ...

    def debug(self, msg, *args, **kwargs) -> datetime | None:
        return self._handle_enhanced_logging(DEBUG, msg, *args, **kwargs)

    @overload
    def info(
        self,
        msg: object,
        *args: object,
        exc_info: bool | tuple | BaseException | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
    ) -> None: ...

    @overload
    def info(self, msg, *args, **kwargs): ...

    def info(self, msg, *args, **kwargs) -> datetime | None:
        return self._handle_enhanced_logging(INFO, msg, *args, **kwargs)

    @overload
    def warning(
        self,
        msg: object,
        *args: object,
        exc_info: bool | tuple | BaseException | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
    ) -> None: ...

    @overload
    def warning(self, msg, *args, **kwargs): ...

    def warning(self, msg, *args, **kwargs) -> datetime | None:
        return self._handle_enhanced_logging(WARNING, msg, *args, **kwargs)

    @overload
    def error(
        self,
        msg: object,
        *args: object,
        exc_info: bool | tuple | BaseException | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
    ) -> None: ...

    @overload
    def error(self, msg, *args, **kwargs): ...

    def error(self, msg, *args, **kwargs) -> datetime | None:
        return self._handle_enhanced_logging(ERROR, msg, *args, **kwargs)

    @overload
    def critical(
        self,
        msg: object,
        *args: object,
        exc_info: bool | tuple | BaseException | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
    ) -> None: ...

    @overload
    def critical(self, msg, *args, **kwargs): ...

    def critical(self, msg, *args, **kwargs) -> datetime | None:
        return self._handle_enhanced_logging(CRITICAL, msg, *args, **kwargs)


def _set_log_file(settings):
    file = settings.logfile
    name = settings.logpath
    root = settings._root_logger
    h = logging.StreamHandler(file) if name is None else logging.FileHandler(name)
    h.setFormatter(_LogFormatter())
    h.setLevel(root.level)
    if len(root.handlers) == 1:
        root.removeHandler(root.handlers[0])
    elif len(root.handlers) > 1:
        raise RuntimeError("Scanpy’s root logger somehow got more than one handler")
    root.addHandler(h)


def _set_log_level(settings, level: int):
    root = settings._root_logger
    root.setLevel(level)
    (h,) = root.handlers  # may only be 1
    h.setLevel(level)


class _LogFormatter(logging.Formatter):
    def __init__(
        self, fmt="{levelname}: {message}", datefmt="%Y-%m-%d %H:%M", style="{"
    ):
        super().__init__(fmt, datefmt, style)

    def format(self, record: logging.LogRecord):
        format_orig = self._style._fmt
        if record.levelno == INFO:
            self._style._fmt = "{message}"
        elif record.levelno == HINT:
            self._style._fmt = "--> {message}"
        elif record.levelno == DEBUG:
            self._style._fmt = "    {message}"

        # Handle time_passed if present (should be in extra)
        time_passed = getattr(record, "time_passed", None)
        if time_passed:
            # Strip microseconds
            if time_passed.microseconds:
                time_passed = timedelta(seconds=int(time_passed.total_seconds()))
            if "{time_passed}" in record.msg:
                record.msg = record.msg.replace("{time_passed}", str(time_passed))
            else:
                self._style._fmt += " ({time_passed})"
                # Add time_passed to record for formatting
                record.time_passed = time_passed

        deep = getattr(record, "deep", None)
        if deep:
            record.msg = f"{record.msg}: {deep}"

        result = logging.Formatter.format(self, record)
        self._style._fmt = format_orig
        return result


print_memory_usage = anndata.logging.print_memory_usage
get_memory_usage = anndata.logging.get_memory_usage

_DEPENDENCIES_NUMERICS = [
    "anndata",  # anndata actually shouldn't, but as long as it's in development
    "umap",
    "numpy",
    "scipy",
    "pandas",
    ("sklearn", "scikit-learn"),
    "statsmodels",
    ("igraph", "python-igraph"),
    "louvain",
]

_DEPENDENCIES_PLOTTING = ["matplotlib", "seaborn"]


def _versions_dependencies(dependencies):
    # this is not the same as the requirements!
    for mod in dependencies:
        mod_name, dist_name = mod if isinstance(mod, tuple) else (mod, mod)
        try:
            imp = __import__(mod_name)
            yield dist_name, imp.__version__
        except (ImportError, AttributeError):
            pass


def print_versions():
    """\
    Versions that might influence the numerical results.

    Matplotlib and Seaborn are excluded from this.
    """
    from ._settings import settings

    modules = ["scanpy"] + _DEPENDENCIES_NUMERICS
    print(
        " ".join(f"{mod}=={ver}" for mod, ver in _versions_dependencies(modules)),
        file=settings.logfile,
    )


def print_version_and_date():
    """\
    Useful for starting a notebook so you see when you started working.
    """
    from . import __version__
    from ._settings import settings

    print(
        f"Running Scanpy {__version__}, " f"on {datetime.now():%Y-%m-%d %H:%M}.",
        file=settings.logfile,
    )


def _copy_docs_and_signature(fn):
    """Copy documentation and signature from function."""
    return partial(update_wrapper, wrapped=fn, assigned=["__doc__", "__annotations__"])


def error(
    msg: str,
    *,
    time: datetime | None = None,
    deep: str | None = None,
    extra: dict[str, Any] | None = None,
) -> datetime:
    """\
    Log message with specific level and return current time.

    Parameters
    ==========
    msg
        Message to display.
    time
        A time in the past. If this is passed, the time difference from then
        to now is appended to `msg` as ` (HH:MM:SS)`.
        If `msg` contains `{time_passed}`, the time difference is instead
        inserted at that position.
    deep
        If the current verbosity is higher than the log function's level,
        this gets displayed as well
    extra
        Additional values you can specify in `msg` like `{time_passed}`.
    """
    from ._settings import settings

    result = settings._root_logger.error(msg, time=time, deep=deep, extra=extra)
    return result or datetime.now(timezone.utc)


@_copy_docs_and_signature(error)
def warning(
    msg: str,
    *,
    time: datetime | None = None,
    deep: str | None = None,
    extra: dict[str, Any] | None = None,
) -> datetime:
    from ._settings import settings

    result = settings._root_logger.warning(msg, time=time, deep=deep, extra=extra)
    return result or datetime.now(timezone.utc)


@_copy_docs_and_signature(error)
def info(
    msg: str,
    *,
    time: datetime | None = None,
    deep: str | None = None,
    extra: dict[str, Any] | None = None,
) -> datetime:
    from ._settings import settings

    result = settings._root_logger.info(msg, time=time, deep=deep, extra=extra)
    return result or datetime.now(timezone.utc)


@_copy_docs_and_signature(error)
def hint(
    msg: str,
    *,
    time: datetime | None = None,
    deep: str | None = None,
    extra: dict[str, Any] | None = None,
) -> datetime:
    from ._settings import settings

    return settings._root_logger.hint(msg, time=time, deep=deep, extra=extra)


@_copy_docs_and_signature(error)
def debug(
    msg: str,
    *,
    time: datetime | None = None,
    deep: str | None = None,
    extra: dict[str, Any] | None = None,
) -> datetime:
    from ._settings import settings

    result = settings._root_logger.debug(msg, time=time, deep=deep, extra=extra)
    return result or datetime.now(timezone.utc)
