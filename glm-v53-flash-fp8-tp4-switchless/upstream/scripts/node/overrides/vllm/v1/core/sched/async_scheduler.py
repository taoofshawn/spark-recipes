# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SITE OVERRIDE (fp8 recipe, 2026-10-10): exact image copy
# (ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:0d4029b3…) +
# GRAMMAR-PAIRING FIX for the required+thinking-on apply_grammar_bitmask crash:
# the `num_output_placeholders > 0` guard misses a grammar request's FIRST
# scheduled spec-decode step (prefill chunks skip the placeholder increment),
# so that step takes the immediate sampling path and, under async scheduling +
# adaptive-K, sample_tokens(N) can run after execute_model(N+1) rebuilt the
# input batch — bitmask rows from step N vs mapping from step N+1 (measured
# 8 vs 4) trips the row-count assert and kills the engine. Deferring whenever
# a structured-output request is scheduled restores the sample/step pairing
# for every grammar step. Do NOT overwrite wholesale on an image update —
# re-extract and re-apply the pairing patch (see VENDORED-AT.md). Rollback:
# drop the -v pair from EXTRA_DOCKER_ENV.

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class AsyncScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # reusable read-only placeholder list for speculative decoding.
        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens
        self.pp_size = self.parallel_config.pipeline_parallel_size

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_after_schedule(scheduler_output)
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        # Use the latest num of scheduled draft tokens in next step as placeholder.
        self._spec_token_placeholders = [-1] * (
            scheduler_output.resolve_num_spec_tokens_to_schedule(self.num_spec_tokens)
        )
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue

            # SITE GRAMMAR-PAIRING FIX: upstream guards on
            # `num_output_placeholders > 0`, which is zero for a grammar
            # request's first scheduled spec-decode step (prefill chunks skip
            # the increment) — that step then samples immediately and can pair
            # step N's bitmask with step N+1's input batch under async
            # scheduling. Defer on `use_structured_output` alone; for non-spec
            # grammar requests placeholders are >0 from their second step, so
            # this only widens coverage to the first step (no behavior change
            # for non-grammar steps: pending is only consumed by the deferred
            # sampling branch of step_with_batch_queue).
            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output
            )
            # The request will generate num_sampled_tokens_per_step new tokens
            # plus num_spec_tokens in this scheduling step. Diffusion has no AR
            # bonus token (num_sampled_tokens_per_step == 0) — only the canvas
            # (spec) tokens.
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            request.num_output_placeholders += (
                self.num_sampled_tokens_per_step + cur_num_spec_tokens
            )
            # Add placeholders for the new draft/spec tokens.
            # We will update the actual spec token ids in the worker process.
            request.spec_token_ids = self._spec_token_placeholders

            if self.use_v2_model_runner:
                # Set the next step index in which this request is eligible to be
                # scheduled for decode (for PP microbatching).
                request.next_decode_eligible_step = self.current_step + self.pp_size

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int], is_stale: bool = False
    ) -> tuple[list[int], bool]:
        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids
        )

        # Placeholders were zeroed at preemption; a stale delivery must not
        # decrement them (it would underflow).
        if not is_stale:
            request.num_output_placeholders -= len(new_token_ids)
            assert request.num_output_placeholders >= 0

        # Cache the new tokens. Preempted requests should be skipped.
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders
            )
        return new_token_ids, stopped
