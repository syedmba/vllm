# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import os
import time

import psutil

from vllm.logger import init_logger

logger = init_logger(__name__)


class EngineProcObserver:
    """For tracking EngineCore orphaned status in background process."""

    @staticmethod
    def track_processes(pids: list[int], parent_pid: int,
                        alive_check_interval: int):
        """
        Check every alive_check_interval seconds
        whether any EngineCore has been orphaned.
        """

        while True:
            if not psutil.pid_exists(parent_pid):
                logger.info(
                    "EngineCores have been orphaned... Proceeding to terminate."
                )

                for pid in pids:
                    with contextlib.suppress(Exception):
                        os.kill(pid, 9)

                return

            time.sleep(alive_check_interval)
