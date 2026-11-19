"""Runtime behaviour of the TTS voice-fallback switcher."""

import asyncio
import contextvars
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    StartFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
    FrameProcessorSetup,
)
from pipecat.tests.mock_tts_service import MockTTSService
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.errors import ErrorCategory
from pipecat.utils.run_context import org_id_var, run_id_var

from api.services.pipecat import fallback_tts
from api.services.pipecat.fallback_tts import FallbackTTSService
from api.services.pipecat.murf_tts import MurfTTSService
from api.services.pipecat.termination_funnel_processor import is_terminal_error


class Capture(FrameProcessor):
    def __init__(self):
        super().__init__(enable_direct_mode=True)
        self.frames = []
        self.errors = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, ErrorFrame):
            self.errors.append(frame)
        else:
            self.frames.append(frame)
        await self.push_frame(frame, direction)


def http_error(status):
    request = httpx.Request("GET", "https://example.com")
    return httpx.HTTPStatusError(
        str(status), request=request, response=httpx.Response(status, request=request)
    )


class RecordingPrimary(MockTTSService):
    def __init__(self, **kwargs):
        kwargs.setdefault("stop_frame_timeout_s", 0.2)
        super().__init__(frame_delay=0, **kwargs)
        self.fail_in_setup = False

    async def setup(self, setup):
        await super().setup(setup)
        if self.fail_in_setup:
            await self.push_error("handshake failed", exception=http_error(402))


class RecordingBackup(MockTTSService):
    """A backup that connects after its base setup, as websocket services do."""

    def __init__(self, **kwargs):
        super().__init__(stop_frame_timeout_s=0.2, frame_delay=0, **kwargs)
        self.slow_connect_s = 0.0
        self.connect_gate: asyncio.Event | None = None
        self.connect_error = False
        self.fail_auth_in_setup = False

    async def setup(self, setup):
        await super().setup(setup)
        if self.slow_connect_s:
            await asyncio.sleep(self.slow_connect_s)
        if self.connect_gate:
            await self.connect_gate.wait()
        if self.connect_error:
            raise RuntimeError("connect refused")
        if self.fail_auth_in_setup:
            await self.push_error("unauthorized", exception=http_error(401))
            await self.set_usable(False)


class StreamingPrimary(RecordingPrimary):
    """Like a websocket provider: run_tts only sends the request, the audio comes
    later through the audio context, and the context stays open until it is
    closed. The test closes it (remove_audio_context), which runs the same
    _end_audio_context path as the idle stop timeout, at a moment it picks."""

    def __init__(self, *, push_stop_frames=True, **kwargs):
        # The idle timer is out of play: the test closes the context itself.
        super().__init__(stop_frame_timeout_s=3600.0, **kwargs)
        self._push_stop_frames = push_stop_frames
        self.context_closed = asyncio.Event()
        self.arrived = []
        self.pushed = []

    async def queue_frame(self, frame, *args, **kwargs):
        self.arrived.append(frame)
        await super().queue_frame(frame, *args, **kwargs)

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        await super().push_frame(frame, direction)
        self.pushed.append(frame)

    async def run_tts(self, text, context_id):
        self.received_texts.append(text)
        yield None

    async def on_audio_context_completed(self, context_id):
        self.context_closed.set()


class TwoHalvesBackup(RecordingBackup):
    """Speaks one utterance in two halves, the test holding the gap between them."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.second_half = asyncio.Event()

    def _chunk(self, context_id):
        return TTSAudioRawFrame(
            b"\x00\x01" * 160, self.sample_rate, 1, context_id=context_id
        )

    async def run_tts(self, text, context_id):
        self.received_texts.append(text)
        yield self._chunk(context_id)
        await self.second_half.wait()
        yield self._chunk(context_id)


def make_harness(
    *,
    fail_in_setup=False,
    slow_backup_setup=0.0,
    backup_connect_gate=None,
    fail_backup_auth=False,
    backup_connect_error=False,
    primary=None,
    backup=None,
):
    if primary is None:
        primary = RecordingPrimary(name="PrimaryTTS")
    primary.fail_in_setup = fail_in_setup
    if backup is None:
        backup = RecordingBackup(name="BackupTTS")
        backup.slow_connect_s = slow_backup_setup
        backup.connect_gate = backup_connect_gate
        backup.fail_auth_in_setup = fail_backup_auth
        backup.connect_error = backup_connect_error
    tts = FallbackTTSService(primary, backup)
    upstream, downstream = Capture(), Capture()
    pipeline = Pipeline([upstream, tts, downstream])
    manager = TaskManager()
    processor_setup = FrameProcessorSetup(
        clock=SystemClock(),
        task_manager=manager,
        pipeline_worker=SimpleNamespace(app_resources={}, worker_runner=None),
        enable_metrics=True,
        enable_usage_metrics=True,
    )

    async def setup():
        await pipeline.setup(processor_setup)

    return SimpleNamespace(
        tts=tts,
        primary=primary,
        backup=backup,
        pipeline=pipeline,
        upstream=upstream,
        downstream=downstream,
        manager=manager,
        setup=setup,
    )


async def build_harness(**options):
    harness = make_harness(**options)
    await harness.setup()
    return harness


async def eventually(predicate, *, timeout=3.0, what="the condition"):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        assert loop.time() < deadline, f"timed out waiting for {what}"
        await asyncio.sleep(0.01)


async def start(harness):
    """Queue the StartFrame and wait until it has crossed the switcher."""
    await harness.pipeline.queue_frame(StartFrame())
    await eventually(
        lambda: any(isinstance(f, StartFrame) for f in harness.downstream.frames),
        what="the StartFrame to reach downstream",
    )


def primary_errors(harness):
    return [
        error
        for error in harness.upstream.errors
        if isinstance(error, ErrorFrame) and error.processor is harness.primary
    ]


def frames_of(harness, context_id):
    return [
        frame
        for frame in harness.downstream.frames
        if getattr(frame, "context_id", None) == context_id
    ]


@pytest.mark.asyncio
async def test_healthy_primary_never_calls_backup():
    harness = await build_harness()
    await harness.pipeline.queue_frame(StartFrame())
    await harness.pipeline.queue_frame(TTSSpeakFrame("hello"))
    await asyncio.sleep(0.3)
    assert harness.backup.received_texts == []
    assert harness.tts.fallback_metrics == {"switched": 0}
    await harness.pipeline.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("backup_kind", ["recording", "murf"])
async def test_slow_backup_connect_does_not_delay_setup_start_or_greeting(
    backup_kind, monkeypatch
):
    """A connect that blocks setup or the StartFrame (even for one branch) holds
    the greeting back. The connect stays pending until the test releases it, so
    the greeting arriving first is proved by ordering, not by elapsed time."""
    monkeypatch.setattr(fallback_tts, "_BACKUP_CONNECT_TIMEOUT_S", 30.0)
    connect_gate = asyncio.Event()
    options = {"backup_connect_gate": connect_gate}
    if backup_kind == "murf":
        murf = MurfTTSService(api_key="test-key")

        async def blocked_connect():
            await connect_gate.wait()

        murf._connect_websocket = blocked_connect
        options = {"backup": murf}
    harness = await asyncio.wait_for(build_harness(**options), timeout=10)
    await start(harness)
    await harness.pipeline.queue_frame(TTSSpeakFrame("greeting"))
    await eventually(
        lambda: "greeting" in harness.primary.received_texts, what="the greeting"
    )
    assert not harness.tts._backup_connected.is_set()
    connect_gate.set()
    await eventually(harness.tts._backup_connected.is_set, what="the connect to end")
    await harness.pipeline.cleanup()


@pytest.mark.asyncio
async def test_setup_error_switches_before_greeting():
    harness = await build_harness(fail_in_setup=True)
    await harness.pipeline.queue_frame(StartFrame())
    await harness.pipeline.queue_frame(TTSSpeakFrame("greeting"))
    await asyncio.sleep(0.5)
    assert harness.tts.fallback_metrics["switched"] == 1
    assert "greeting" in harness.backup.received_texts
    assert not harness.upstream.errors
    assert harness.primary.is_usable is False
    await harness.pipeline.cleanup()


@pytest.mark.asyncio
async def test_mid_call_error_switches_and_later_text_uses_backup():
    harness = await build_harness()
    await harness.pipeline.queue_frame(StartFrame())
    await harness.pipeline.queue_frame(TTSSpeakFrame("first"))
    await asyncio.sleep(0.2)
    await harness.primary.push_error(
        "in-stream", fatal=False, category=ErrorCategory.SERVER
    )
    await harness.pipeline.queue_frame(TTSSpeakFrame("second"))
    await asyncio.sleep(0.5)
    assert harness.tts.fallback_metrics["switched"] == 1
    assert "second" in harness.backup.received_texts
    await harness.pipeline.cleanup()


@pytest.mark.asyncio
async def test_two_errors_in_one_tick_switch_once():
    harness = await build_harness()
    await harness.pipeline.queue_frame(StartFrame())
    switched_events = []

    @harness.tts.strategy.event_handler("on_service_switched")
    async def on_switched(_strategy, _service):
        switched_events.append(True)

    await asyncio.gather(
        harness.primary.push_error("one", fatal=False),
        harness.primary.push_error("two", fatal=False),
    )
    await asyncio.sleep(0.2)
    assert harness.tts.fallback_metrics["switched"] == 1
    assert len(switched_events) == 1
    await harness.pipeline.cleanup()


@pytest.mark.asyncio
async def test_latch_blocks_second_error_during_in_flight_switch():
    """Two primaries must reach handle_error while the first switch still awaits."""
    harness = await build_harness()
    await harness.pipeline.queue_frame(StartFrame())
    await asyncio.sleep(0.2)
    assert harness.backup.is_usable

    switched_events = []

    @harness.tts.strategy.event_handler("on_service_switched")
    async def on_switched(_strategy, _service):
        switched_events.append(True)

    gate = asyncio.Event()
    switch_entered = asyncio.Event()
    original_set = harness.tts.strategy._set_active_if_available

    async def held_set_active(service):
        switch_entered.set()
        await gate.wait()
        return await original_set(service)

    harness.tts.strategy._set_active_if_available = held_set_active

    first = asyncio.create_task(
        harness.primary.push_error("one", fatal=False, category=ErrorCategory.SERVER)
    )
    await switch_entered.wait()
    second = asyncio.create_task(
        harness.primary.push_error("two", fatal=False, category=ErrorCategory.SERVER)
    )
    gate.set()
    await asyncio.gather(first, second)
    await asyncio.sleep(0.2)
    assert harness.tts.fallback_metrics["switched"] == 1
    assert len(switched_events) == 1
    await harness.pipeline.cleanup()


@pytest.mark.asyncio
async def test_never_switches_back():
    harness = await build_harness()
    await harness.pipeline.queue_frame(StartFrame())
    await harness.primary.push_error("fail", fatal=False)
    await harness.primary.set_usable(True)
    await harness.backup.push_error("backup blip", fatal=False)
    await harness.pipeline.queue_frame(TTSSpeakFrame("after"))
    await asyncio.sleep(0.3)
    assert harness.tts.strategy.active_service is harness.backup
    await harness.pipeline.cleanup()


@pytest.mark.asyncio
async def test_refused_switch_keeps_primary_usable():
    """Backup unusable before the error: early handle_error guard, no switch."""
    harness = await build_harness(fail_backup_auth=True)
    await harness.pipeline.queue_frame(StartFrame())
    await harness.primary.push_error("upstream", fatal=False, exception=http_error(500))
    await asyncio.sleep(0.5)
    assert harness.tts.fallback_metrics["switched"] == 0
    assert harness.primary.is_usable is True
    assert primary_errors(harness)
    assert not any(is_terminal_error(e) for e in harness.upstream.errors)
    await harness.pipeline.cleanup()


@pytest.mark.asyncio
async def test_unusable_backup_is_refused_without_waiting_for_its_connect(
    monkeypatch,
):
    """Only the early guard avoids the wait: the connect is still pending. Without
    the guard push_error would wait out the connect bound (10 s here), and the
    connect would be over by the time it returned."""
    monkeypatch.setattr(fallback_tts, "_BACKUP_CONNECT_TIMEOUT_S", 10.0)
    harness = await build_harness(backup_connect_gate=asyncio.Event())
    await start(harness)
    assert not harness.tts._backup_connected.is_set()
    await harness.backup.set_usable(False)
    await harness.primary.push_error("upstream", fatal=False, exception=http_error(500))
    assert not harness.tts._backup_connected.is_set()
    assert harness.tts.fallback_metrics["switched"] == 0
    assert harness.tts.strategy.active_service is harness.primary
    assert primary_errors(harness)
    assert not any(is_terminal_error(e) for e in harness.upstream.errors)
    await harness.pipeline.cleanup()


@pytest.mark.asyncio
async def test_refused_in_flight_switch_unlatches_and_keeps_primary():
    """Backup still usable when the error lands; _set_active_if_available refuses."""
    harness = await build_harness()
    await harness.pipeline.queue_frame(StartFrame())
    await asyncio.sleep(0.3)
    assert harness.backup.is_usable

    original_set = harness.tts.strategy._set_active_if_available
    switch_attempts = 0

    async def refuse_first_switch(service):
        nonlocal switch_attempts
        switch_attempts += 1
        if switch_attempts == 1:
            return None
        return await original_set(service)

    harness.tts.strategy._set_active_if_available = refuse_first_switch

    await harness.primary.push_error("upstream", fatal=False, exception=http_error(500))
    await asyncio.sleep(0.3)
    assert harness.tts.fallback_metrics["switched"] == 0
    assert harness.primary.is_usable is True
    assert harness.tts.strategy.active_service is harness.primary
    assert primary_errors(harness)

    await harness.primary.push_error(
        "retry", fatal=False, category=ErrorCategory.SERVER
    )
    await asyncio.sleep(0.3)
    assert harness.tts.fallback_metrics["switched"] == 1
    assert harness.tts.strategy.active_service is harness.backup
    await harness.pipeline.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("category", [ErrorCategory.QUOTA, ErrorCategory.UNKNOWN])
async def test_switch_errors_types_includes_quota_and_unknown(category):
    """Pin the v1 trigger: any non-fatal primary error, like Retell/Vapi failover."""
    harness = await build_harness()
    await harness.pipeline.queue_frame(StartFrame())
    await harness.primary.push_error("err", fatal=False, category=category)
    await asyncio.sleep(0.1)
    assert harness.tts.strategy.active_service is harness.backup
    await harness.pipeline.cleanup()


@pytest.mark.asyncio
async def test_backup_permanent_error_makes_switcher_unusable():
    harness = await build_harness()
    await harness.pipeline.queue_frame(StartFrame())
    await harness.primary.push_error("fail", fatal=False)
    await harness.backup.set_usable(False)
    await harness.backup.push_error("backup dead", fatal=False)
    await asyncio.sleep(0.2)
    assert harness.primary.is_usable is False
    assert harness.backup.is_usable is False
    assert harness.tts.is_usable is False
    terminal = [e for e in harness.upstream.errors if is_terminal_error(e)]
    assert len(terminal) == 1
    assert terminal[0].processor is harness.tts
    await harness.pipeline.cleanup()


async def settle_primary_path(harness):
    """Push a frame that is not speech through the primary. Frames keep their
    order on the way out, so once it is downstream everything the primary pushed
    before it has been forwarded or dropped."""
    marker = TextFrame("settled")
    await harness.primary.push_frame(marker)
    await eventually(
        lambda: any(f is marker for f in harness.downstream.frames),
        what="the primary's frames to settle",
    )


@pytest.mark.asyncio
async def test_late_primary_audio_is_dropped_after_switch():
    harness = await build_harness()
    await harness.pipeline.queue_frame(StartFrame())
    await harness.pipeline.queue_frame(TTSSpeakFrame("x"))
    await eventually(
        lambda: any(isinstance(f, TTSAudioRawFrame) for f in harness.downstream.frames),
        what="the primary's audio",
    )
    context_id = next(
        f.context_id
        for f in harness.downstream.frames
        if isinstance(f, TTSAudioRawFrame)
    )
    await harness.primary.push_error("fail", fatal=False)
    harness.downstream.frames.clear()
    late = TTSAudioRawFrame(b"\x00\x01", 16000, 1, context_id=context_id)
    await harness.primary.push_frame(late, FrameDirection.DOWNSTREAM)
    await settle_primary_path(harness)
    assert not any(
        isinstance(frame, TTSAudioRawFrame) and frame.context_id == context_id
        for frame in harness.downstream.frames
    )
    await harness.pipeline.cleanup()


@pytest.mark.asyncio
async def test_interrupted_utterance_is_closed_by_the_switch_not_by_the_primary():
    """The primary's Started went out before it failed: the switch closes the
    context itself, and the primary's late audio and own Stopped add nothing, so
    downstream sees exactly one Stopped. The context is one no TTS request made,
    as a recording clip passing through the primary is: it is the primary's by
    where its frames were pushed, not by what they carry."""
    harness = await build_harness()
    await start(harness)
    context_id = "clip"
    await harness.primary.push_frame(TTSStartedFrame(context_id=context_id))
    await eventually(
        lambda: frames_of(harness, context_id), what="the clip's Started frame"
    )
    await harness.primary.push_error("fail", fatal=False, category=ErrorCategory.SERVER)
    assert harness.tts.strategy.active_service is harness.backup
    assert [type(frame) for frame in frames_of(harness, context_id)] == [
        TTSStartedFrame,
        TTSStoppedFrame,
    ]
    await harness.primary.push_frame(
        TTSAudioRawFrame(b"\x00\x01", 16000, 1, context_id=context_id)
    )
    await harness.primary.push_frame(TTSStoppedFrame(context_id=context_id))
    await settle_primary_path(harness)
    assert [type(frame) for frame in frames_of(harness, context_id)] == [
        TTSStartedFrame,
        TTSStoppedFrame,
    ]
    await harness.pipeline.cleanup()


@pytest.mark.asyncio
async def test_primary_speech_after_the_switch_is_dropped_and_the_backups_is_not():
    harness = await build_harness()
    await start(harness)
    dead, live = "primary-never-started", "backup-context"
    await harness.primary.push_error("fail", fatal=False, category=ErrorCategory.SERVER)
    for frame in (
        TTSStartedFrame(context_id=dead),
        TTSStoppedFrame(context_id=dead),
    ):
        await harness.primary.push_frame(frame)
    await settle_primary_path(harness)
    for frame in (
        TTSStartedFrame(context_id=live),
        TTSStoppedFrame(context_id=live),
    ):
        await harness.backup.push_frame(frame)
    await eventually(
        lambda: len(frames_of(harness, live)) == 2, what="the backup's frames"
    )
    assert frames_of(harness, dead) == []
    assert [type(frame) for frame in frames_of(harness, live)] == [
        TTSStartedFrame,
        TTSStoppedFrame,
    ]
    await harness.pipeline.cleanup()


MEDIA = (TTSStartedFrame, TTSAudioRawFrame, TTSStoppedFrame)


def lifecycle(harness):
    return [
        (type(frame).__name__, frame.context_id)
        for frame in harness.downstream.frames
        if isinstance(frame, (TTSStartedFrame, TTSStoppedFrame))
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("primary_pushes_stop_frames", [True, False])
async def test_primary_context_is_closed_at_the_switch_and_its_late_close_stays_out(
    primary_pushes_stop_frames,
):
    """A real switch while the primary is mid-context, then a backup utterance
    that outlasts the primary's context: the primary's own close (the idle stop
    timeout in production, closed by the test here) lands in the middle of the
    backup's speech. Downstream must see Stopped(X) before Started(Y), and no
    Stopped inside Y: the output transport ends the bot's speech on any
    TTSStoppedFrame, which would cut Y's audio. A provider that never pushes a
    stop frame must still be closed."""
    primary = StreamingPrimary(
        name="PrimaryTTS", push_stop_frames=primary_pushes_stop_frames
    )
    backup = TwoHalvesBackup(name="BackupTTS")
    harness = await build_harness(primary=primary, backup=backup)
    await start(harness)
    await eventually(harness.tts._backup_connected.is_set, what="the backup connect")

    await harness.pipeline.queue_frame(TTSSpeakFrame("first"))
    await eventually(
        lambda: any(isinstance(f, TTSStartedFrame) for f in harness.downstream.frames),
        what="the primary's context to start",
    )
    x = next(c for kind, c in lifecycle(harness) if kind == "TTSStartedFrame")
    await primary.append_to_audio_context(
        x, TTSAudioRawFrame(b"\x00\x01" * 160, primary.sample_rate, 1, context_id=x)
    )
    await eventually(
        lambda: any(isinstance(f, TTSAudioRawFrame) for f in frames_of(harness, x)),
        what="the primary's audio",
    )

    await primary.push_error("stream died", fatal=False, category=ErrorCategory.SERVER)
    assert harness.tts.strategy.active_service is backup
    assert harness.tts.fallback_metrics["switched"] == 1
    # Closed by the switch itself, before the backup has said anything.
    assert [type(f) for f in frames_of(harness, x) if type(f) in MEDIA] == [
        TTSStartedFrame,
        TTSAudioRawFrame,
        TTSStoppedFrame,
    ]

    await harness.pipeline.queue_frame(TTSSpeakFrame("second"))
    await eventually(
        lambda: any(
            isinstance(f, TTSAudioRawFrame) and f.context_id != x
            for f in harness.downstream.frames
        ),
        what="the first half of the backup's utterance",
    )
    y = next(c for kind, c in lifecycle(harness) if c != x)

    # The primary's context ends now, mid-way through Y.
    await primary.remove_audio_context(x)
    await asyncio.wait_for(primary.context_closed.wait(), timeout=10)
    backup.second_half.set()
    await eventually(
        lambda: ("TTSStoppedFrame", y) in lifecycle(harness),
        what="the backup's utterance to end",
    )

    assert lifecycle(harness) == [
        ("TTSStartedFrame", x),
        ("TTSStoppedFrame", x),
        ("TTSStartedFrame", y),
        ("TTSStoppedFrame", y),
    ]
    frames = harness.downstream.frames
    stopped_x = next(
        i
        for i, f in enumerate(frames)
        if isinstance(f, TTSStoppedFrame) and f.context_id == x
    )
    started_y = next(
        i
        for i, f in enumerate(frames)
        if isinstance(f, TTSStartedFrame) and f.context_id == y
    )
    stopped_y = next(
        i
        for i, f in enumerate(frames)
        if isinstance(f, TTSStoppedFrame) and f.context_id == y
    )
    assert stopped_x < started_y
    audio = [
        (i, f.context_id)
        for i, f in enumerate(frames)
        if isinstance(f, TTSAudioRawFrame)
    ]
    assert [c for _, c in audio] == [x, y, y]
    assert audio[0][0] < stopped_x
    assert all(started_y < i < stopped_y for i, c in audio if c == y)
    await harness.pipeline.cleanup()


@pytest.mark.asyncio
async def test_recording_clip_queued_behind_the_primary_stays_out_of_the_backup_speech():
    """A recording router sits before the TTS and its clips carry contexts of
    their own: queued behind the primary's playing context, they come out only
    when that context ends, which is mid-way through the backup's utterance
    after a switch. None of the clip may play there or end the bot's speech."""
    primary = StreamingPrimary(name="PrimaryTTS")
    backup = TwoHalvesBackup(name="BackupTTS")
    harness = await build_harness(primary=primary, backup=backup)
    await start(harness)
    await eventually(harness.tts._backup_connected.is_set, what="the backup connect")

    await harness.pipeline.queue_frame(TTSSpeakFrame("first"))
    await eventually(
        lambda: any(isinstance(f, TTSStartedFrame) for f in harness.downstream.frames),
        what="the primary's context to start",
    )
    x = next(c for kind, c in lifecycle(harness) if kind == "TTSStartedFrame")

    clip = "recording-clip"
    for frame in (
        TTSStartedFrame(context_id=clip),
        TTSAudioRawFrame(b"\x00\x01" * 160, primary.sample_rate, 1, context_id=clip),
        TTSStoppedFrame(context_id=clip),
    ):
        await harness.pipeline.queue_frame(frame)
    await eventually(
        lambda: any(
            isinstance(f, TTSStoppedFrame) and f.context_id == clip
            for f in primary.arrived
        ),
        what="the clip to reach the primary",
    )

    await primary.push_error("stream died", fatal=False, category=ErrorCategory.SERVER)
    assert harness.tts.strategy.active_service is backup
    await harness.pipeline.queue_frame(TTSSpeakFrame("second"))
    await eventually(
        lambda: any(
            isinstance(f, TTSAudioRawFrame) and f.context_id != x
            for f in harness.downstream.frames
        ),
        what="the first half of the backup's utterance",
    )
    y = next(c for kind, c in lifecycle(harness) if c not in (x, clip))

    # The primary's context ends mid-way through Y and releases the clip.
    await primary.remove_audio_context(x)
    await eventually(
        lambda: any(
            isinstance(f, TTSStoppedFrame) and f.context_id == clip
            for f in primary.pushed
        ),
        what="the primary to release the clip",
    )
    backup.second_half.set()
    await eventually(
        lambda: ("TTSStoppedFrame", y) in lifecycle(harness),
        what="the backup's utterance to end",
    )

    assert frames_of(harness, clip) == []
    assert lifecycle(harness) == [
        ("TTSStartedFrame", x),
        ("TTSStoppedFrame", x),
        ("TTSStartedFrame", y),
        ("TTSStoppedFrame", y),
    ]
    await harness.pipeline.cleanup()


@pytest.mark.asyncio
async def test_latch_cleared_when_switch_cancelled_then_retries():
    """The first error is cancelled while the switch waits for the backup's
    connect: the latch must not stay set, or no later error could switch."""
    harness = await build_harness(slow_backup_setup=2.0)
    await start(harness)
    strategy = harness.tts.strategy
    first = asyncio.create_task(
        harness.primary.push_error("one", fatal=False, category=ErrorCategory.SERVER)
    )
    await eventually(lambda: strategy._switched, what="the switch to start waiting")
    assert not first.done()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert first.cancelled()
    assert strategy._switched is False
    assert strategy.active_service is harness.primary
    assert harness.tts.fallback_metrics["switched"] == 0

    await eventually(
        harness.tts._backup_connected.is_set, timeout=5.0, what="the backup connect"
    )
    await harness.primary.push_error("two", fatal=False, category=ErrorCategory.SERVER)
    assert strategy.active_service is harness.backup
    assert harness.tts.fallback_metrics["switched"] == 1
    await harness.pipeline.cleanup()


@pytest.mark.asyncio
async def test_end_frame_blocks_late_switch():
    harness = await build_harness()
    await start(harness)
    await harness.pipeline.queue_frame(EndFrame())
    await eventually(lambda: harness.tts._visit_ending, what="the EndFrame")
    await harness.primary.push_error("late", fatal=False, category=ErrorCategory.SERVER)
    await eventually(
        lambda: any(isinstance(f, EndFrame) for f in harness.downstream.frames),
        what="the EndFrame to reach downstream",
    )
    assert harness.tts.fallback_metrics["switched"] == 0
    assert harness.tts.strategy.active_service is harness.primary
    assert primary_errors(harness)
    await harness.pipeline.cleanup()


@pytest.mark.asyncio
async def test_backup_connect_timeout_leaves_backup_usable_and_uncounted(monkeypatch):
    """A connect that outlasts the bound is not a failure: the switch still goes
    to the backup, which reconnects when it first has to speak."""
    monkeypatch.setattr(fallback_tts, "_BACKUP_CONNECT_TIMEOUT_S", 0.2)
    harness = await build_harness(slow_backup_setup=30.0)
    await start(harness)
    await eventually(
        harness.tts._backup_connected.is_set, what="the connect to time out"
    )
    assert harness.backup.is_usable
    assert "backup_unavailable" not in harness.tts.fallback_metrics
    await harness.primary.push_error("fail", fatal=False, category=ErrorCategory.SERVER)
    assert harness.tts.strategy.active_service is harness.backup
    await harness.pipeline.queue_frame(TTSSpeakFrame("after"))
    await eventually(
        lambda: "after" in harness.backup.received_texts, what="the backup to speak"
    )
    await harness.pipeline.cleanup()


@pytest.mark.asyncio
async def test_backup_connect_failure_marks_backup_unusable_and_counts_it():
    harness = await build_harness(backup_connect_error=True)
    await eventually(harness.tts._backup_connected.is_set, what="the connect to fail")
    assert harness.backup.is_usable is False
    assert harness.tts.fallback_metrics["backup_unavailable"] == 1
    await start(harness)
    await harness.primary.push_error("fail", fatal=False, category=ErrorCategory.SERVER)
    assert harness.tts.fallback_metrics["switched"] == 0
    assert harness.tts.strategy.active_service is harness.primary
    assert primary_errors(harness)
    assert not any(is_terminal_error(e) for e in harness.upstream.errors)
    await harness.pipeline.cleanup()


@pytest.mark.asyncio
async def test_inactive_backup_auth_error_does_not_pause_campaign(monkeypatch):
    from api.services.campaign import circuit_breaker as cb_module

    pause = AsyncMock()
    monkeypatch.setattr(cb_module, "get_current_run_id", lambda: "7")
    monkeypatch.setattr(cb_module, "get_current_org_id", lambda: "1")
    monkeypatch.setattr(cb_module.db_client, "pause_campaign_for_provider_error", pause)
    harness = await build_harness(fail_backup_auth=True)
    await harness.pipeline.queue_frame(StartFrame())
    await eventually(harness.tts._backup_connected.is_set, what="the backup connect")
    pause.assert_not_awaited()
    assert harness.tts.fallback_metrics["backup_unavailable"] == 1
    await harness.pipeline.cleanup()


@pytest.mark.asyncio
async def test_active_backup_auth_error_pauses_campaign(monkeypatch):
    from api.services.campaign import circuit_breaker as cb_module

    pause = AsyncMock()
    monkeypatch.setattr(cb_module, "get_current_run_id", lambda: "7")
    monkeypatch.setattr(cb_module, "get_current_org_id", lambda: "1")
    monkeypatch.setattr(cb_module.db_client, "pause_campaign_for_provider_error", pause)
    harness = await build_harness()
    await harness.pipeline.queue_frame(StartFrame())
    await harness.primary.push_error("fail", fatal=False, category=ErrorCategory.SERVER)
    await asyncio.sleep(0.3)
    await harness.backup.push_error(
        "unauthorized", fatal=False, exception=http_error(401)
    )
    await asyncio.sleep(0.2)
    pause.assert_awaited()
    await harness.pipeline.cleanup()


@pytest.mark.asyncio
async def test_active_backup_pauses_campaign_when_switch_runs_outside_the_run(
    monkeypatch,
):
    """The campaign handler is registered at switch time, on whichever task
    pushed the primary's error. The run lives in the construction context."""
    from api.services.campaign import circuit_breaker as cb_module

    pause = AsyncMock()
    monkeypatch.setattr(cb_module.db_client, "pause_campaign_for_provider_error", pause)
    run_token, org_token = run_id_var.set("7"), org_id_var.set("1")
    try:
        harness = await build_harness()
        await start(harness)
        await asyncio.create_task(
            harness.primary.push_error(
                "fail", fatal=False, category=ErrorCategory.SERVER
            ),
            context=contextvars.Context(),
        )
        assert harness.tts.strategy.active_service is harness.backup
        await harness.backup.push_error(
            "unauthorized", fatal=False, exception=http_error(401)
        )
        pause.assert_awaited()
        await harness.pipeline.cleanup()
    finally:
        run_id_var.reset(run_token)
        org_id_var.reset(org_token)
