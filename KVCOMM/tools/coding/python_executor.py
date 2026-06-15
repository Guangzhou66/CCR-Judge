import ast
try:
    import astunparse
except ImportError:  # Python 3.9+ provides ast.unparse in the standard library.
    astunparse = None
from typing import List

from KVCOMM.tools.coding.executor_utils import function_with_timeout
from KVCOMM.tools.coding.executor_types import ExecuteResult, Executor
import multiprocessing as mp
import textwrap
import traceback
import queue
import threading
import atexit
import contextlib
import io
import os
from typing import Any, Dict, Tuple


def get_call_str(assert_statement: str) -> str:
    ast_parsed = ast.parse(assert_statement)
    try:
        call_str = ast_parsed.body[0].test.left
    except Exception:
        call_str = ast_parsed.body[0].test

    if astunparse is not None:
        return astunparse.unparse(call_str).strip()
    return ast.unparse(call_str).strip()


def _execution_is_silent() -> bool:
    return os.environ.get("KVCOMM_CODE_EXEC_SILENT", "1").lower() in {"1", "true", "yes", "y"}


@contextlib.contextmanager
def _maybe_redirect_stdio():
    if not _execution_is_silent():
        yield
        return
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        yield


def _sandbox_worker(conn) -> None:
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg is None:
            return

        kind, payload = msg
        try:
            if kind == "exec":
                with _maybe_redirect_stdio():
                    exec(str(payload), {})
                conn.send(("ok", None))
                continue
            if kind == "eval_call":
                func_src, call_expr = payload
                local_vars: Dict[str, Any] = {}
                with _maybe_redirect_stdio():
                    exec(f"from typing import *\n{func_src}", {}, local_vars)
                    output = eval(str(call_expr), {}, local_vars)
                conn.send(("ok", output))
                continue
            if kind == "exec_get_answer":
                code = str(payload)
                local_vars: Dict[str, Any] = {}
                with _maybe_redirect_stdio():
                    exec(textwrap.dedent(code), {}, local_vars)
                    res = local_vars.get("answer")
                    if callable(res):
                        res = res()
                conn.send(("ok", res))
                continue
            conn.send(("error", f"Unknown sandbox request: {kind!r}"))
        except BaseException:
            conn.send(("error", traceback.format_exc()))


class _Sandbox:
    """Persistent subprocess for safe code execution with hard timeouts.

    Using a long-lived subprocess avoids the heavy cost of spawning a new process
    for every single HumanEval test, while still allowing us to kill the worker
    if user code hangs.
    """

    def __init__(self) -> None:
        self._ctx = mp.get_context("spawn")
        self._lock = threading.Lock()
        self._proc: mp.Process | None = None
        self._conn = None

    def _ensure(self) -> None:
        if self._proc is not None and self._proc.is_alive() and self._conn is not None:
            return
        self._restart()

    def _restart(self) -> None:
        self._close()
        parent_conn, child_conn = self._ctx.Pipe(duplex=True)
        proc = self._ctx.Process(target=_sandbox_worker, args=(child_conn,))
        proc.daemon = True
        proc.start()
        try:
            child_conn.close()
        except Exception:
            pass
        self._proc = proc
        self._conn = parent_conn

    def _close(self) -> None:
        try:
            if self._conn is not None:
                try:
                    self._conn.send(None)
                except Exception:
                    pass
                try:
                    self._conn.close()
                except Exception:
                    pass
        finally:
            self._conn = None
        if self._proc is not None:
            try:
                if self._proc.is_alive():
                    self._proc.terminate()
                self._proc.join(timeout=0.2)
            except Exception:
                pass
            self._proc = None

    def request(self, kind: str, payload: Any, timeout: float) -> Tuple[str, Any]:
        with self._lock:
            self._ensure()
            assert self._conn is not None
            try:
                self._conn.send((kind, payload))
                if not self._conn.poll(timeout):
                    self._restart()
                    return "timeout", None
                status, result = self._conn.recv()
                return status, result
            except Exception as exc:
                self._restart()
                return "error", f"Sandbox communication error: {exc}"


_SANDBOX = _Sandbox()
atexit.register(_SANDBOX._close)


def get_output(func: str, assert_statement: str, timeout: int = 5) -> str:
    try:
        func_call = get_call_str(assert_statement)
    except Exception as exc:
        return f"Unable to extract call: {exc}"
    status, payload = _SANDBOX.request("eval_call", (func, func_call), timeout=timeout)
    if status == "timeout":
        return "TIMEOUT"
    if status != "ok":
        return payload if isinstance(payload, str) else "ERROR"
    return payload


def execute_code_get_return(code: str, timeout: int = 5):
    """在子进程执行 code，超过 timeout 秒直接终止，并返回 answer 变量（或错误信息）。"""

    def _runner(q):
        local_vars = {}
        try:
            with _maybe_redirect_stdio():
                exec(textwrap.dedent(code), {}, local_vars)
                res = local_vars.get("answer")
                if callable(res):
                    res = res()

            q.put(res)
        except Exception:
            q.put(f"Error occurred:\n{traceback.format_exc()}")

    q = mp.Queue()
    p = mp.Process(target=_runner, args=(q,))
    p.start()

    try:
        result = q.get(timeout=timeout)
    except queue.Empty:
        p.terminate()
        p.join()
        return f"Timeout (> {timeout}s)"
    else:
        p.join()
        return result


class PyExecutor(Executor):
    def execute(
        self,
        func: str,
        tests: List[str],
        timeout: int = 5,
        verbose: bool = True,
    ) -> ExecuteResult:

        imports = "from typing import *"
        func_test_list = [f"{imports}\n{func}\n{test}" for test in tests]

        success_tests = []
        failed_tests = []
        is_passing = True
        num_tests = len(func_test_list)
        for i in range(num_tests):
            status, payload = _SANDBOX.request("exec", func_test_list[i], timeout=timeout)
            if status == "ok":
                success_tests.append(tests[i])
                continue
            output = "TIMEOUT" if status == "timeout" else (payload if isinstance(payload, str) else "ERROR")
            # If this is an assert-style test, try to extract the call output for debugging.
            if status != "timeout" and isinstance(tests[i], str) and "assert" in tests[i]:
                call_output = get_output(func, tests[i], timeout=timeout)
                output = call_output if call_output != "TIMEOUT" else output
            failed_tests.append(f"{tests[i]} # output: {output}")
            is_passing = False

        state = [test in success_tests for test in tests]

        feedback = "Tests passed:\n" + "\n".join(success_tests) + "\n\nTests failed:"
        feedback += "\n" + "\n".join(failed_tests)
        return is_passing, feedback, tuple(state)

    def evaluate(self, name: str, func: str, test: str, timeout: int = 5) -> bool:
        """
        Evaluates the implementation on Human-Eval Python.

        probably should be written in a dataset-agnostic way but not now
        """

        code = f"""{func}

{test}

check({name})
    """
        status, _payload = _SANDBOX.request("exec", code, timeout=timeout)
        return status == "ok"
