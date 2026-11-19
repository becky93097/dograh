"""Switch TTS providers on the primary's first error, without replaying lost text."""

from __future__ import annotations

import asyncio
import contextvars
from typing import TypeVar

from loguru import logger

from api.errors.failure import (
    ErrorSource,
    annotate_failure_metadata,
    failure_metadata_for_processor,
)
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.pipeline.service_switcher import ServiceSwitcher, ServiceSwitcherStrategy
from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
    FrameProcessorSetup,
)
from pipecat.services.tts_service import TTSService

StrategyType = TypeVar("StrategyType", bound=ServiceSwitcherStrategy)

_BACKUP_CONNECT_TIMEOUT_S = 3.0
_TTS_SPEECH_FRAMES = (TTSStartedFrame, TTSAudioRawFrame, TTSTextFrame, TTSStoppedFrame)
# Set on the primary's speech frames as it pushes them, so a frame is known to
# come from the primary by where it was pushed and not by what it carries.
_FROM_PRIMARY = "tts_fallback_from_primary"


class _SwitchOnFirstError(ServiceSwitcherStrategy):
    """Matches Retell/Vapi: the first primary failure moves the visit to the backup."""

    def __init__(self, services: list[FrameProcessor]):
        super().__init__(services)
        self._switched = False
        self.prepare_switch = None
        self.complete_switch = None
        self.visit_over = None

    async def handle_error(self, error: ErrorFrame) -> TTSService | None:
        if self._switched or (self.visit_over and self.visit_over()):
            return None
        primary, backup = self.services
        # An unusable backup is refused up front: waiting for its pending
        # connect would only delay the primary's own error for nothing.
        if (
            self.active_service is not primary
            or error.processor is not primary
            or not backup.is_usable
        ):
            return None
        self._switched = True
        try:
            if self.prepare_switch:
                await self.prepare_switch(error)
            switched = await self._set_active_if_available(backup)
            if not switched:
                self._switched = False
                return None
            if self.complete_switch:
                await self.complete_switch(error)
            return switched
        except BaseException:
            self._switched = False
            raise


class FallbackTTSService(ServiceSwitcher[_SwitchOnFirstError]):
    def __init__(self, primary: TTSService, backup: TTSService):
        super().__init__(services=[primary, backup], strategy_type=_SwitchOnFirstError)
        self.primary = primary
        self.backup = backup
        # Contexts of the primary's speech whose Started frame went downstream
        # and whose Stopped frame has not; the switch closes them (see
        # _close_primary_contexts).
        self._open_primary_contexts: dict[str, TTSStartedFrame] = {}
        self._visit_ending = False
        self._backup_setup: FrameProcessorSetup | None = None
        self._backup_connect_task: asyncio.Task | None = None
        self._backup_connected = asyncio.Event()
        self._backup_unavailable_logged = False
        self._fallback_metrics: dict[str, int] = {"switched": 0}
        self._campaign_handler_registered = False
        # The campaign handler is registered when the backup becomes active, on
        # whichever task pushed the primary's error; the run and organization
        # live in this construction-time context, not necessarily in that task's.
        self._run_context = contextvars.copy_context()
        self.strategy.prepare_switch = self._prepare_switch
        self.strategy.complete_switch = self._complete_switch
        self.strategy.visit_over = lambda: self._visit_ending
        primary.add_event_handler("on_before_push_frame", self._mark_primary_frame)
        backup.add_event_handler("on_usable_changed", self._on_backup_usable_changed)
        backup.add_event_handler("on_error", self._on_backup_error)

        primary_metadata = failure_metadata_for_processor(primary)
        annotate_failure_metadata(
            self,
            source=primary_metadata.source,
            provider=primary_metadata.provider,
            error_owner=primary_metadata.error_owner,
        )

    @property
    def fallback_metrics(self) -> dict:
        return dict(self._fallback_metrics)

    def _mark_primary_frame(self, _primary, frame: Frame) -> None:
        # Everything the primary says passes through here, including the frames
        # of other processors (a recording router's clips) queued behind its
        # playing context, which carry contexts of their own. The event is
        # synchronous, so the mark is there before the frame reaches push_frame.
        if isinstance(frame, _TTS_SPEECH_FRAMES):
            frame.metadata[_FROM_PRIMARY] = True

    def _count_backup_unavailable(self) -> None:
        if self._backup_unavailable_logged:
            return
        self._backup_unavailable_logged = True
        self._fallback_metrics["backup_unavailable"] = (
            self._fallback_metrics.get("backup_unavailable", 0) + 1
        )

    def _on_backup_error(self, _service, _error: ErrorFrame) -> None:
        if self.strategy.active_service is self.backup:
            return
        logger.warning("TTS fallback backup connect failed while inactive")
        self._count_backup_unavailable()

    async def _on_backup_usable_changed(self, _service, is_usable: bool) -> None:
        if not is_usable and self.strategy.active_service is self.backup:
            logger.warning("TTS fallback backup is no longer usable")

    async def _prepare_switch(self, _error: ErrorFrame) -> None:
        await self._ensure_backup_connected()

    def _register_backup_campaign_handler(self) -> None:
        if self._campaign_handler_registered:
            return
        from api.services.campaign.circuit_breaker import (
            register_campaign_provider_error_handler,
        )

        metadata = failure_metadata_for_processor(self.backup)
        self._run_context.run(
            register_campaign_provider_error_handler,
            self.backup,
            provider=metadata.provider,
            source=metadata.source or ErrorSource.TTS,
        )
        self._campaign_handler_registered = True

    async def _close_primary_contexts(self) -> None:
        """Close the primary's open contexts now, ahead of anything the backup says.

        The primary's own Stopped frame is not waited for: it comes only after
        the idle stop timeout (or never, for a provider that does not push stop
        frames), so it would land inside the backup's next utterance, where the
        output transport would end the bot's speech and clear that utterance's
        audio. Every later speech frame of the primary is dropped in push_frame.
        """
        open_contexts, self._open_primary_contexts = self._open_primary_contexts, {}
        for context_id, started in open_contexts.items():
            stopped = TTSStoppedFrame(context_id=context_id)
            stopped.transport_destination = started.transport_destination
            await super().push_frame(stopped)

    async def _complete_switch(self, error: ErrorFrame) -> None:
        await self._close_primary_contexts()
        self._fallback_metrics["switched"] += 1
        logger.warning(
            f"TTS fallback: {self.primary.name} failed ({error.error}); "
            f"continuing on {self.backup.name}"
        )
        self._register_backup_campaign_handler()
        await self.primary.set_usable(False)

    async def _ensure_backup_connected(self) -> None:
        if self._backup_connected.is_set():
            return
        if self._backup_connect_task and not self._backup_connect_task.done():
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._backup_connect_task),
                    timeout=_BACKUP_CONNECT_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                logger.warning("TTS fallback backup connect timed out during switch")

    async def setup(self, setup: FrameProcessorSetup):
        await FrameProcessor.setup(self, setup)
        self._setup_finished()
        self._backup_setup = setup
        backup_service = self.backup
        await FrameProcessor.setup(self._pipelines[1], setup)
        for processor in self._pipelines[1].processors:
            processor._setup_started()
            if processor is backup_service:
                await FrameProcessor.setup(backup_service, setup)
            else:
                await processor.setup(setup)
            processor._setup_finished()
        self._backup_connect_task = self.create_task(
            self._connect_backup_in_background(), f"{self}::backup_connect"
        )
        await self._setup_processors([self._pipelines[0]], setup)

    async def _connect_backup_in_background(self) -> None:
        setup = self._backup_setup
        if setup is None:
            self._backup_connected.set()
            return
        try:
            await asyncio.wait_for(
                self.backup.setup(setup),
                timeout=_BACKUP_CONNECT_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning("TTS fallback backup lazy connect timed out")
        except Exception as exc:
            logger.warning(
                f"TTS fallback backup lazy connect failed: {type(exc).__name__}"
            )
            self._count_backup_unavailable()
            await self.backup.set_usable(False)
        finally:
            self._backup_connected.set()

    async def cleanup(self):
        if self._backup_connect_task and not self._backup_connect_task.done():
            self._backup_connect_task.cancel()
            await self.cancel_task(self._backup_connect_task)
        await super().cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, (EndFrame, CancelFrame)):
            self._visit_ending = True
        await super().process_frame(frame, direction)

    async def push_frame(
        self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ):
        if direction == FrameDirection.DOWNSTREAM and frame.metadata.get(_FROM_PRIMARY):
            if self.strategy.active_service is self.backup:
                # The primary's speech is gone: its audio, text and lifecycle
                # frames are dropped, whatever context they carry, and its
                # Stopped frames are not forwarded because
                # _close_primary_contexts already sent them.
                return
            context_id = getattr(frame, "context_id", None)
            if isinstance(frame, TTSStartedFrame) and context_id:
                self._open_primary_contexts[context_id] = frame
            elif isinstance(frame, TTSStoppedFrame):
                self._open_primary_contexts.pop(context_id, None)
        await super().push_frame(frame, direction)
